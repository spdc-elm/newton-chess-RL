# -*- coding: utf-8 -*-
"""模型天梯：所有 checkpoint 与基线 bot 循环互殴，按 Elo 排名。

- 每个 NN checkpoint 以「网页同款」形态参赛：纯策略直落（前向→掩码→argmax）。
- 每对选手打 N 局，先后手各半；输出胜负表 + 收敛后的 Elo 排名。
- 另设一场表演赛：冠军带 PUCT 搜索（NNBot sims=48）vs 自己不带搜索，量化「搜索的价值」。

用法:
  python3 rl/arena.py [--games 20] [--out rl/runs/arena_report.md]
"""
import argparse
import glob
import json
import os
import random
import re
import sys

import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import nf_env as nf                                  # noqa: E402
from bots.random_bot import RandomBot                # noqa: E402
from bots.greedy_bot import GreedyBot                # noqa: E402
from bots.nn_bot import NNBot                        # noqa: E402
from training.model import NFNet                     # noqa: E402
from eval_policy_only import PolicyOnlyBot           # noqa: E402


def load_net(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    net = NFNet(planes_in=3,
                channels=ck["net"]["stem.0.weight"].shape[0],
                blocks=sum(1 for k in ck["net"]
                           if k.endswith("c1.weight") and k.startswith("trunk")))
    net.load_state_dict(ck["net"])
    tag = "%s@iter%s" % (os.path.basename(os.path.dirname(ckpt_path)), ck.get("iter"))
    return net, tag


def play_game(mk_a, mk_b, a_first, w, h):
    """mk_a/mk_b 是「返回 bot」的工厂。返回 (a 的边子, b 的边子)。"""
    s = nf.create(w, h, 2)
    for _ in range(w * h + 1):
        if s.phase == "settled":
            break
        bot = mk_a() if ((s.cur == 0) == a_first) else mk_b()
        x, y = bot.select_move(s)
        rec, err = nf.apply_move(s, x, y)
        assert rec is not None, err
    c0, c1 = nf.border_counts(s)
    return (c0, c1) if a_first else (c1, c0)


def match(mk_a, mk_b, games, w, h, seed0=0):
    """a vs b 各先后手一半。返回 (aWin, bWin, draw)。"""
    aw = bw = d = 0
    for g in range(games):
        a_first = (g % 2 == 0)
        a_cnt, b_cnt = play_game(mk_a, mk_b, a_first, w, h)
        if a_cnt < b_cnt:
            aw += 1
        elif a_cnt > b_cnt:
            bw += 1
        else:
            d += 1
    return aw, bw, d


def elo_ratings(results, names, k=32, base=1200, passes=300, seed=7):
    """results: list of (name_i, name_j, score_i)，score_i ∈ {1, .5, 0} 是 name_i 的得分。"""
    r = {n: float(base) for n in names}
    rng = random.Random(seed)
    order = list(range(len(results)))
    for _ in range(passes):
        rng.shuffle(order)
        for idx in order:
            ni, nj, si = results[idx]
            ea = 1.0 / (1.0 + 10 ** ((r[nj] - r[ni]) / 400.0))
            r[ni] += k * (si - ea)
            r[nj] += k * ((1 - si) - (1 - ea))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "arena_report.md"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--include", nargs="*", default=None,
                    help="只保留名字含这些子串的参赛者（冒烟用）")
    ap.add_argument("--exhibition", type=int, default=12,
                    help="表演赛局数：冠军带 PUCT 搜索 vs 纯策略直落（0 关闭）")
    args = ap.parse_args()

    # ---- 集结参赛者 ----
    entrants_all = [("random", lambda: RandomBot(args.seed)),
                    ("greedy", lambda: GreedyBot(args.seed + 1))]
    ckpts = sorted(glob.glob(os.path.join(ROOT, "runs", "*", "latest.pt"))) + \
            sorted(glob.glob(os.path.join(ROOT, "runs", "*", "ckpt_iter*.pt")))
    loaded = {}
    for path in ckpts:
        try:
            net, tag = load_net(path)
        except Exception as e:
            print("跳过 %s: %s" % (path, e))
            continue
        loaded[tag] = (net, path)

    # 只保留每个 run 的关键节点，避免天梯爆炸：latest + 每 run 最多 3 个中间点
    by_run = {}
    for tag, (net, path) in loaded.items():
        run, it = tag.split("@iter")
        by_run.setdefault(run, []).append((int(it), tag))
    chosen_tags = []
    for run, items in sorted(by_run.items()):
        items.sort()
        picks = [items[-1][1]]                          # latest iter
        mids = items[:-1]
        if len(mids) >= 3:
            step = max(1, len(mids) // 2)
            picks += [mids[0][1], mids[step][1]]
        elif mids:
            picks += [m[1] for m in mids]
        chosen_tags.extend(picks)

    for tag in sorted(set(chosen_tags)):
        net, _ = loaded[tag]
        entrants_all.append((tag, (lambda n: lambda: PolicyOnlyBot(n))(net)))

    if args.include:
        entrants_all = [(n, m) for n, m in entrants_all
                        if any(sub in n for sub in args.include)]
    entrants = entrants_all

    print("参赛者 %d 名 · %dx%d 棋盘 · 每对 %d 局\n" %
          (len(entrants), args.size, args.size, args.games))
    for i, (name, _) in enumerate(entrants):
        print("  [%2d] %s" % (i, name))

    # ---- 循环赛 ----
    names = [n for n, _ in entrants]
    makers = [m for _, m in entrants]
    tally = {(i, j): [0, 0, 0] for i in range(len(names))
             for j in range(i + 1, len(names))}       # [i胜, j胜, 平]
    elo_games = []
    total_pairs = len(names) * (len(names) - 1) // 2
    done = 0
    t_start = __import__("time").time()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            aw, bw, d = match(makers[i], makers[j], args.games, args.size, args.size,
                              seed0=args.seed + i * 100 + j)
            tally[(i, j)] = [aw, bw, d]
            for g in range(aw):
                elo_games.append((names[i], names[j], 1.0))
            for g in range(bw):
                elo_games.append((names[i], names[j], 0.0))
            for g in range(d):
                elo_games.append((names[i], names[j], 0.5))
            done += 1
            print("  (%2d/%2d) %-22s vs %-22s → %d:%d:%d" %
                  (done, total_pairs, names[i], names[j], aw, bw, d), flush=True)
    elapsed = __import__("time").time() - t_start

    ratings = elo_ratings(elo_games, names)
    ranking = sorted(ratings.items(), key=lambda kv: -kv[1])

    lines = []
    lines.append("# 牛顿棋 · 模型天梯报告")
    lines.append("")
    lines.append("- 棋盘：%d×%d · 每对 %d 局（先后手各半）· 总耗时 %.0fs" %
                 (args.size, args.size, args.games, elapsed))
    lines.append("- NN 参赛形态：纯策略直落（与网页内置 AI 相同）")
    lines.append("- Elo：初始 1200，K=32，全量对局重放收敛 %d 轮" % 300)
    lines.append("")
    lines.append("## 排名")
    lines.append("")
    lines.append("| 排名 | 选手 | Elo |")
    lines.append("|---|---|---|")
    for rank, (name, r) in enumerate(ranking, 1):
        lines.append("| %d | %s | %.0f |" % (rank, name, r))
    lines.append("")
    lines.append("## 对战矩阵（行视角：胜-负-平）")
    lines.append("")
    header = "| | " + " | ".join(names) + " |"
    sep = "|---" * (len(names) + 1) + "|"
    lines += [header, sep]
    for i in range(len(names)):
        row = ["%s" % names[i]]
        for j in range(len(names)):
            if i == j:
                row.append("—")
            else:
                a, b = min(i, j), max(i, j)
                iw, jw, dr = tally[(a, b)]
                if i == a:
                    row.append("%d-%d-%d" % (iw, jw, dr))
                else:
                    row.append("%d-%d-%d" % (jw, iw, dr))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("报告已生成于本地时间 %s" % __import__("time").strftime("%Y-%m-%d %H:%M"))

    # ---- 表演赛：搜索的价值 ----
    if args.exhibition > 0:
        champ_name = ranking[0][0]
        net, path = loaded[champ_name]
        print("\n表演赛：%s 带 PUCT(sims=48) vs 纯策略直落 ×%d…" %
              (champ_name, args.exhibition), flush=True)
        mk_search = (lambda n: lambda: NNBot(n.to("cpu").eval(), device="cpu",
                                             sims=48, c_puct=1.5, seed=args.seed))(net)
        mk_policy = (lambda n: lambda: PolicyOnlyBot(n))(net)
        sw, pw, d = match(mk_search, mk_policy, args.exhibition, args.size, args.size,
                          seed0=args.seed + 999)
        lines.append("")
        lines.append("## 表演赛：搜索的价值")
        lines.append("")
        lines.append("%s +PUCT(sims=48) vs %s 纯策略直落 → **%d : %d : %d**"
                     "（先手/后手各半；平局多说明两者接近同一水平线）" %
                     (champ_name, champ_name, sw, pw, d))
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("表演赛结果 %d:%d:%d" % (sw, pw, d))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n%s\n" % ("\n".join(lines)))
    print("报告已写入:", args.out)


if __name__ == "__main__":
    main()
