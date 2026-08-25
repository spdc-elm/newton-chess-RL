#!/usr/bin/env python3
"""同尺寸 checkpoint 的分色、分差 arena。"""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

import nf_env as env  # noqa: E402
from bots.nn_bot import NNBot  # noqa: E402
from training.model import NFNet  # noqa: E402

_A = _B = None
_CFG = None


def load_net(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = NFNet(planes_in=3,
                channels=ck["net"]["stem.0.weight"].shape[0],
                blocks=sum(1 for k in ck["net"]
                           if k.endswith("c1.weight") and k.startswith("trunk")))
    net.load_state_dict(ck["net"])
    return net.to("cpu").eval(), ck


def init_worker(a_state, b_state, channels, blocks, cfg):
    global _A, _B, _CFG
    torch.set_num_threads(1)
    _A = NFNet(3, channels, blocks).to("cpu").eval()
    _B = NFNet(3, channels, blocks).to("cpu").eval()
    _A.load_state_dict(a_state)
    _B.load_state_dict(b_state)
    _CFG = cfg


def play_one(job):
    game, seed = job
    w = h = _CFG["size"]
    sims = _CFG["sims"]
    c_puct = _CFG["c_puct"]
    a_first = (game % 2 == 0)
    first_sims = _CFG.get("first_sims", sims)
    second_sims = _CFG.get("second_sims", sims)
    a_bot = NNBot(_A, device="cpu", sims=first_sims if a_first else second_sims,
                  c_puct=c_puct, temperature=_CFG["temperature"], seed=seed * 2 + 1)
    b_bot = NNBot(_B, device="cpu", sims=second_sims if a_first else first_sims,
                  c_puct=c_puct, temperature=_CFG["temperature"], seed=seed * 2 + 2)
    s = env.create(w, h, 2)
    for _ in range(w * h + 1):
        if s.phase == "settled":
            break
        a_turn = ((s.cur == 0) == a_first)
        x, y = (a_bot if a_turn else b_bot).select_move(s)
        rec, err = env.apply_move(s, x, y)
        if rec is None:
            raise RuntimeError("非法手 %s: %s" % ((x, y), err))
    counts = env.border_counts(s)
    a_count, b_count = (counts[0], counts[1]) if a_first else (counts[1], counts[0])
    if a_count < b_count:
        result = "A"
    elif a_count > b_count:
        result = "B"
    else:
        result = "D"
    return {"game": game, "a_first": a_first, "a_count": a_count,
            "b_count": b_count, "a_margin": (b_count - a_count) / (2 * (w + h) - 4),
            "result": result}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="候选 checkpoint")
    ap.add_argument("--b", required=True, help="旧 checkpoint")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--sims", type=int, default=96)
    ap.add_argument("--first-sims", type=int, default=None,
                    help="给执先方的模拟数；默认等于 --sims")
    ap.add_argument("--second-sims", type=int, default=None,
                    help="给执后方的模拟数；默认等于 --sims")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a_net, a_ck = load_net(args.a)
    b_net, b_ck = load_net(args.b)
    a_state = {k: v.detach().cpu() for k, v in a_net.state_dict().items()}
    b_state = {k: v.detach().cpu() for k, v in b_net.state_dict().items()}
    channels = a_state["stem.0.weight"].shape[0]
    blocks = sum(1 for k in a_state if k.endswith("c1.weight") and k.startswith("trunk"))
    if channels != b_state["stem.0.weight"].shape[0]:
        raise ValueError("两个 checkpoint 的 channels 不同")
    cfg = {"size": args.size, "sims": args.sims,
           "first_sims": args.first_sims or args.sims,
           "second_sims": args.second_sims or args.sims,
           "c_puct": args.c_puct, "temperature": args.temperature}
    rng = random.Random(args.seed)
    jobs = [(g, rng.randrange(1 << 30)) for g in range(args.games)]
    started = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=init_worker,
                  initargs=(a_state, b_state, channels, blocks, cfg)) as pool:
        details = list(pool.imap_unordered(play_one, jobs, chunksize=1))
    details.sort(key=lambda x: x["game"])

    def stats(items):
        wins = sum(x["result"] == "A" for x in items)
        losses = sum(x["result"] == "B" for x in items)
        draws = sum(x["result"] == "D" for x in items)
        score = (wins + 0.5 * draws) / max(1, len(items))
        margins = np.asarray([x["a_margin"] for x in items], dtype=np.float64)
        return {"games": len(items), "A_wins": wins, "B_wins": losses, "draws": draws,
                "A_score": round(float(score), 5),
                "A_mean_margin": round(float(margins.mean()), 5),
                "A_margin_std": round(float(margins.std()), 5),
                "A_mean_count": round(float(np.mean([x["a_count"] for x in items])), 4),
                "B_mean_count": round(float(np.mean([x["b_count"] for x in items])), 4)}

    first = [x for x in details if x["a_first"]]
    second = [x for x in details if not x["a_first"]]
    report = {
        "a_path": args.a, "b_path": args.b,
        "a_iter": a_ck.get("iter"), "b_iter": b_ck.get("iter"),
        "a_value_target": a_ck.get("value_target", a_ck.get("cfg", {}).get("value_target", "unknown")),
        "b_value_target": b_ck.get("value_target", b_ck.get("cfg", {}).get("value_target", "unknown")),
        "size": args.size, "sims": args.sims,
        "first_sims": cfg["first_sims"], "second_sims": cfg["second_sims"],
        "c_puct": args.c_puct,
        "temperature": args.temperature, "workers": args.workers, "seed": args.seed,
        "elapsed_s": round(time.time() - started, 1),
        "all": stats(details), "A_first": stats(first), "A_second": stats(second),
        "games_detail": details,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "games_detail"}, ensure_ascii=False, indent=2))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
