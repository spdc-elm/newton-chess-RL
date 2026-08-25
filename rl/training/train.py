# -*- coding: utf-8 -*-
"""Phase 1/2 训练主循环：自博弈 → 训练 → 评测 → checkpoint。

用法示例：
  python3 rl/training/train.py --name p1_8x8 --size 8 --iters 18 \
      --games-per-iter 240 --sims 48 --workers 8 --device mps
冒烟：
  python3 rl/training/train.py --smoke
"""
import argparse
import collections
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "rl"))

import nf_env as env                                    # noqa: E402
from bots.random_bot import RandomBot                   # noqa: E402
from bots.greedy_bot import GreedyBot                   # noqa: E402
from bots.nn_bot import NNBot                           # noqa: E402
from training.model import NFNet                        # noqa: E402
from training.augment import D4_COUNT, expand_d4_batch, opening_orbit_representatives  # noqa: E402
import training.selfplay as sp                          # noqa: E402


VALUE_TARGET = "normalized_border_margin_v2"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run")
    ap.add_argument("--size", type=int, default=8, help="正方形棋盘边长（8–19）")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--games-per-iter", type=int, default=200)
    ap.add_argument("--sims", type=int, default=48)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--value-loss-weight", type=float, default=1.0,
                    help="margin value MSE 在总 loss 中的权重 λ")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--temp-moves", type=int, default=12)
    ap.add_argument("--buffer", type=int, default=60000)
    ap.add_argument("--device", default="auto", help="auto|mps|cpu（训练用）")
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--eval-games", type=int, default=16)
    ap.add_argument("--eval-sims", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--init", default=None, help="从已有 checkpoint 热启动（全卷积可跨尺寸加载）")
    ap.add_argument("--start-iter", type=int, default=0,
                    help="续训时的起始全局迭代号；与 --init 配合，训练到 --iters")
    ap.add_argument("--no-d4", action="store_true", help="关闭全部 8 种 D4 对称增强（仅供消融）")
    ap.add_argument("--no-save-selfplay", action="store_true",
                    help="不保存每轮基础自博弈样本；默认保存为可复用 npz")
    ap.add_argument("--no-quick-eval", action="store_true",
                    help="关闭训练脚本内置的 random/greedy 快速评测；正式评测另跑 checkpoint arena")
    ap.add_argument("--opening-coverage", action="store_true",
                    help="按 D4 不等价首着循环覆盖开局（9×9 为 15 类）")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def pick_device(arg):
    if arg != "auto":
        return arg
    return "mps" if torch.backends.mps.is_available() else "cpu"


def save_selfplay_samples(out_dir, iteration, samples):
    """保存未增强的基础样本，避免未来再次丢失 replay 数据。"""
    sample_dir = os.path.join(out_dir, "selfplay")
    os.makedirs(sample_dir, exist_ok=True)
    path = os.path.join(sample_dir, "iter%03d.npz" % iteration)
    np.savez_compressed(
        path,
        format=np.asarray(VALUE_TARGET),
        obs=np.stack([b[0] for b in samples]).astype(np.uint8),
        pi=np.stack([b[1] for b in samples]).astype(np.float32),
        value=np.asarray([b[2] for b in samples], dtype=np.float32),
    )
    return path


def train_epochs(net, buffer_, opt, device, epochs, batch, d4=True, value_loss_weight=1.0):
    if value_loss_weight < 0:
        raise ValueError("value_loss_weight 必须 >= 0")
    net.train()
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "steps": 0,
             "train_examples": 0, "base_examples": 0}
    obs = np.stack([b[0] for b in buffer_])
    pi = np.stack([b[1] for b in buffer_])
    value = np.asarray([b[2] for b in buffer_], dtype=np.float32)
    n = len(obs)
    # D4 下每个基础样本展开 8 份；batch 仍保持接近用户配置值。
    base_batch = max(1, batch // D4_COUNT) if d4 else batch
    if n < base_batch:
        base_batch = max(1, n)
    for _ in range(epochs):
        order = np.random.permutation(n)
        for i in range(0, n - base_batch + 1, base_batch):
            idx = order[i:i + base_batch]
            x_np, pi_np, value_np = obs[idx], pi[idx], value[idx]
            if d4:
                x_np, pi_np, value_np = expand_d4_batch(x_np, pi_np, value_np)
            x = torch.from_numpy(x_np).to(device)
            target_pi = torch.from_numpy(pi_np).to(device)
            target_value = torch.from_numpy(value_np).to(device)
            logits, v = net(x)
            logp = F.log_softmax(logits, dim=1)
            loss_pi = -(target_pi * logp).sum(dim=1).mean()
            loss_v = F.mse_loss(v, target_value)
            loss = loss_pi + value_loss_weight * loss_v
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            stats["policy_loss"] += float(loss_pi.detach())
            stats["value_loss"] += float(loss_v.detach())
            stats["steps"] += 1
            stats["train_examples"] += len(x_np)
            stats["base_examples"] += len(idx)
    if stats["steps"]:
        stats["policy_loss"] /= stats["steps"]
        stats["value_loss"] /= stats["steps"]
    return stats


def quick_eval(net, cfg, device, games, sims, seed=123):
    """当前网络（确定性、无噪声）对随机/贪心基线的双色轮换胜率，按先后手拆分。"""
    import copy
    eval_net = copy.deepcopy(net).to("cpu").eval()
    results = {}
    for opp_name, opp in (("random", RandomBot(seed + 1)), ("greedy", GreedyBot(seed + 2))):
        bot = NNBot(eval_net, device="cpu", sims=sims, c_puct=1.5, seed=seed)
        agg = {"as_first": [0, 0, 0], "as_second": [0, 0, 0]}   # W/L/D
        for g in range(games):
            s = env.create(cfg["w"], cfg["h"], cfg["n"])
            first_nn = (g % 2 == 0)
            for _ in range(cfg["w"] * cfg["h"] + 1):
                if s.phase == "settled":
                    break
                cur = s.cur
                mover_is_nn = (cur == 0) == first_nn
                x, y = bot.select_move(s) if mover_is_nn else opp.select_move(s)
                rec, err = env.apply_move(s, x, y)
                assert rec is not None, err
            counts = env.border_counts(s)
            nn_border = counts[0] if first_nn else counts[1]
            opp_border = counts[1] if first_nn else counts[0]
            r = 0 if nn_border < opp_border else 1 if nn_border > opp_border else 2
            agg["as_first" if first_nn else "as_second"][r] += 1
        def pack(a):
            n_ = sum(a)
            return {"win": a[0], "loss": a[1], "draw": a[2],
                    "score": round((a[0] + 0.5 * a[2]) / max(1, n_), 3)}
        total = [agg["as_first"][i] + agg["as_second"][i] for i in range(3)]
        results[opp_name] = {"total": pack(total), "先手": pack(agg["as_first"]), "后手": pack(agg["as_second"])}
    return results


def main():
    args = parse_args()
    if args.smoke:
        args.iters, args.games_per_iter, args.sims = 2, 8, 16
        args.workers, args.epochs, args.eval_games, args.eval_sims = 4, 1, 2, 8
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    out_dir = os.path.join(ROOT, "rl", "runs", args.name)
    os.makedirs(out_dir, exist_ok=True)
    cfg = {
        "w": args.size, "h": args.size, "n": 2,
        "sims": args.sims, "c_puct": 1.5,
        "dirichlet_eps": 0.25, "dirichlet_alpha": 0.2,
        "temp_moves": args.temp_moves,
        "opening_actions": opening_orbit_representatives(args.size) if args.opening_coverage else [],
    }
    net = NFNet(planes_in=3, channels=args.channels, blocks=args.blocks).to(device)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["net"])
        source_target = ck.get("value_target", ck.get("cfg", {}).get("value_target", "outcome_v1"))
        if source_target != VALUE_TARGET:
            # policy/trunk 热启动；旧胜负 value 语义不兼容，清零重训。
            net.reset_value_head(zero_output=True)
            print("value 语义迁移:", source_target, "→", VALUE_TARGET, "（value 头已重置为 0）")
        print("热启动自:", args.init, "(iter", str(ck.get("iter")) + ")")
    print("参数量:", net.num_params(), "· 训练设备:", device, "· value:", VALUE_TARGET,
          "· D4:", "关闭" if args.no_d4 else "8×",
          "· 开局覆盖:", len(cfg["opening_actions"]) or "关闭", "· 输出目录:", out_dir)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    buffer_ = collections.deque(maxlen=args.buffer)
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    rng = random.Random(args.seed)

    for it in range(args.start_iter + 1, args.iters + 1):
        t0 = time.time()
        # ---- 自博弈 ----
        state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        seeds = [rng.randrange(1 << 30) for _ in range(args.games_per_iter)]
        samples = []
        game_metas = []
        if args.workers > 1:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(args.workers, initializer=sp.pool_init,
                          initargs=(state_dict, 3, args.channels, args.blocks, "cpu")) as pool:
                for out, meta in pool.imap_unordered(
                        sp.pool_play, sp.make_pool_jobs(cfg, seeds), chunksize=1):
                    samples.extend(out)
                    game_metas.append(meta)
        else:
            bot_template = (copy_net(net), "cpu")
            for i, sd in enumerate(seeds):
                forced = cfg["opening_actions"][i % len(cfg["opening_actions"])] if cfg["opening_actions"] else None
                out, meta = sp.play_one_game(bot_template, cfg, sd, forced)
                samples.extend(out)
                game_metas.append(meta)
        t_selfplay = time.time() - t0
        sample_path = None
        if samples and not args.no_save_selfplay:
            sample_path = save_selfplay_samples(out_dir, it, samples)

        # ---- 训练 ----
        buffer_.extend(samples)
        t1 = time.time()
        tstats = train_epochs(net, buffer_, opt, device, args.epochs, args.batch,
                              d4=not args.no_d4,
                              value_loss_weight=args.value_loss_weight)
        t_train = time.time() - t1

        # ---- 评测与落盘 ----
        ev = {}
        if not args.no_quick_eval and (it % args.eval_every == 0 or it == args.iters):
            ev = quick_eval(net, cfg, device, args.eval_games, args.eval_sims)
        rec = {
            "iter": it, "games": args.games_per_iter, "samples": len(samples),
            "selfplay_s": round(t_selfplay, 1), "train_s": round(t_train, 1),
            "buffer": len(buffer_),
            "policy_loss": round(tstats["policy_loss"], 4),
            "value_loss": round(tstats["value_loss"], 4),
            "value_loss_weight": args.value_loss_weight,
            "value_target": VALUE_TARGET,
            "d4": not args.no_d4,
            "opening_coverage": cfg["opening_actions"],
            "train_examples": tstats["train_examples"],
            "selfplay_file": os.path.relpath(sample_path, ROOT) if sample_path else None,
            "eval": ev,
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        checkpoint = {"cfg": vars(args), "net": net.state_dict(), "iter": it,
                      "optimizer": opt.state_dict(),
                      "value_loss_weight": args.value_loss_weight,
                      "value_target": VALUE_TARGET,
                      "augmentation": "d4x8" if not args.no_d4 else "none",
                      "opening_coverage": cfg["opening_actions"]}
        torch.save(checkpoint, os.path.join(out_dir, "latest.pt"))
        if it % max(1, args.iters // 6) == 0 or it == args.iters:
            torch.save(checkpoint, os.path.join(out_dir, "ckpt_iter%03d.pt" % it))
        print("iter %3d/%d · %d 基础样本/%d 训练样本 · 自博弈 %.0fs · 训练 %.0fs · π_loss %.3f · margin_loss %.3f%s"
              % (it, args.iters, len(samples), tstats["train_examples"], t_selfplay, t_train,
                 tstats["policy_loss"], tstats["value_loss"],
                 (" · eval " + json.dumps(ev, ensure_ascii=False)) if ev else ""), flush=True)

    print("训练完成 →", out_dir)


def copy_net(net):
    import copy
    return copy.deepcopy(net).to("cpu").eval()


if __name__ == "__main__":
    main()
