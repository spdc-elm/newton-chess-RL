# -*- coding: utf-8 -*-
"""评估「网页同款」AI 强度：纯策略网络直落（无 MCTS），与 random/greedy 对战。

这正是 newton-force.html 内置 AI 的决策方式（前向 → 掩码 → argmax）。
用法: python3 rl/eval_policy_only.py [--ckpt rl/runs/p2_9x9/latest.pt] [--games 100]
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import nf_env as nf                                  # noqa: E402
from bots.random_bot import RandomBot                # noqa: E402
from bots.greedy_bot import GreedyBot                # noqa: E402


class PolicyOnlyBot:
    def __init__(self, net, device="cpu"):
        self.net = net.to(device).eval()
        self.device = device

    def _forward(self, s):
        obs = nf.encode_canonical(s)
        with torch.no_grad():
            logits, v = self.net(torch.from_numpy(obs).unsqueeze(0).to(self.device))
        return logits[0].cpu().numpy(), float(v[0])

    def select_move(self, s):
        lg, _ = self._forward(s)
        legal = nf.legal_mask(s)
        best_x = best_y = -1
        best_v = -1e30
        for y in range(s.h):
            for x in range(s.w):
                if not legal[y * s.w + x]:
                    continue
                if lg[y * s.w + x] > best_v:
                    best_v = lg[y * s.w + x]
                    best_x, best_y = x, y
        return best_x, best_y


class Value1PlyBot:
    """策略先验排序 + 每个候选落点做一次 value 前向，选「对手最难受」的着法。"""
    def __init__(self, net, device="cpu", topk=12):
        self.net = net.to(device).eval()
        self.device = device
        self.topk = topk

    def _forward(self, s):
        obs = nf.encode_canonical(s)
        with torch.no_grad():
            logits, v = self.net(torch.from_numpy(obs).unsqueeze(0).to(self.device))
        return logits[0].cpu().numpy(), float(v[0])

    def select_move(self, s):
        obs = nf.encode_canonical(s)
        with torch.no_grad():
            logits, _ = self.net(torch.from_numpy(obs).unsqueeze(0).to(self.device))
        lg = logits[0].cpu().numpy()
        legal = nf.legal_mask(s)
        cands = [(lg[y * s.w + x], x, y)
                 for y in range(s.h) for x in range(s.w) if legal[y * s.w + x]]
        cands.sort(reverse=True)
        cands = cands[:self.topk]
        best, best_score = cands[0], -1e30
        for _, x, y in cands:
            st = nf.clone(s)
            rec, err = nf.apply_move(st, x, y)
            assert rec is not None, err
            if st.phase == "settled":
                counts = nf.border_counts(st)
                mine = counts[rec["player"]]
                opps = max(c for i, c in enumerate(counts) if i != rec["player"])
                score = 1e6 if mine < opps else (-1e6 if mine > opps else 0.5 * lg[x + y * s.w])
            else:
                _, v2 = self._forward(st)
                score = -v2            # 对手的胜率预期取反 = 我方利益
            if score > best_score:
                best_score, best = score, (_, x, y)
        return best[1], best[2]


def play_game(net, opp, first_nn, seed, mode="policy"):
    import random as rnd
    rng = rnd.Random(seed)
    s = nf.create(9, 9, 2)
    nn = (PolicyOnlyBot(net) if mode == "policy" else Value1PlyBot(net))
    for _ in range(s.w * s.h + 1):
        if s.phase == "settled":
            break
        mover_is_nn = ((s.cur == 0) == first_nn)
        if mover_is_nn:
            x, y = nn.select_move(s)
        else:
            if hasattr(opp, "rng"):
                opp.rng = rng
            x, y = opp.select_move(s)
        rec, err = nf.apply_move(s, x, y)
        assert rec is not None, err
    counts = nf.border_counts(s)
    a, b = (counts[0], counts[1]) if first_nn else (counts[1], counts[0])
    return ("W" if a < b else "L" if a > b else "D")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs", "p1_8x8", "latest.pt"))
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--mode", choices=("policy", "value1"), default="policy")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    from training.model import NFNet
    net = NFNet(planes_in=3, channels=ck["net"]["stem.0.weight"].shape[0],
                blocks=sum(1 for k in ck["net"] if k.endswith("c1.weight") and k.startswith("trunk")))
    net.load_state_dict(ck["net"])

    print("模型: %s (iter %s, 训练尺寸 %s) · 棋盘 %dx%d · %s" %
          (os.path.basename(args.ckpt), ck.get("iter"), ck.get("cfg", {}).get("size"),
           args.size, args.size, "纯策略直落" if args.mode == "policy" else "策略先验+1层价值搜索(top12)"))

    for name, mk in (("random", lambda: RandomBot(7)), ("greedy", lambda: GreedyBot(9))):
        w = l = d = 0
        for g in range(args.games):
            r = play_game(net, mk(), g % 2 == 0, 1000 + g, args.mode)
            w += r == "W"; l += r == "L"; d += r == "D"
        score = (w + 0.5 * d) / max(1, args.games)
        elo = "∞" if l == 0 else ""
        print("vs %-6s: %d胜 %d负 %d平 · 得分率 %.2f %s" % (name, w, l, d, score, elo))


if __name__ == "__main__":
    main()
