#!/usr/bin/env python3
"""Gumbel AlphaZero checkpoint 评测闭环。

同一候选 checkpoint 在三种动作协议下分别评测：

* ``det``：temperature=0，评测时根 Gumbel noise=0；
* ``early12``：前 12 手按根访问数 temperature=1，之后 temperature=0；
* ``all1``：全程按根访问数 temperature=1。

每种协议都做：候选 vs opening15/iter210 anchor、候选 vs corrected
puctsqrt/iter210 anchor，以及候选自战。候选使用 Gumbel 搜索，旧 anchor
使用无 Dirichlet 的标准 sqrt- PUＣT；所有对局先后手各半，并报告分色分差。
自战还记录空棋盘 value 与最终「当前行动方」分差的校准。

这个脚本特意把 temperature 采样放在搜索结果之外，避免为了换协议重新
搜索一遍同一局面。Gumbel 评测关闭根噪声是论文在 perfect-information
游戏上的推荐做法；训练时则由 train.py 保持 noise=True。
"""
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
from bots.nn_bot import NNBot, select_by_temperature  # noqa: E402
from training.model import NFNet  # noqa: E402

_CAND = _ANCHOR = None
_CFG = None


def load_net(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["net"]
    channels = state["stem.0.weight"].shape[0]
    blocks = sum(1 for k in state if k.endswith("c1.weight") and k.startswith("trunk"))
    net = NFNet(planes_in=3, channels=channels, blocks=blocks)
    net.load_state_dict(state)
    return net.to("cpu").eval(), ck


def init_worker(cand_state, anchor_state, channels, blocks, cfg):
    global _CAND, _ANCHOR, _CFG
    torch.set_num_threads(1)
    _CAND = NFNet(3, channels, blocks).to("cpu").eval()
    _ANCHOR = NFNet(3, channels, blocks).to("cpu").eval()
    _CAND.load_state_dict(cand_state)
    _ANCHOR.load_state_dict(anchor_state)
    _CFG = cfg


def _make_bot(net, kind, seed):
    return NNBot(
        net, device="cpu", sims=_CFG["sims"], c_puct=_CFG["c_puct"],
        dirichlet_eps=0.0, temperature=0.0, seed=seed,
        gumbel=(kind == "gumbel"),
        gumbel_max_actions=_CFG["gumbel_max_actions"],
        gumbel_c_visit=_CFG["gumbel_c_visit"],
        gumbel_c_scale=_CFG["gumbel_c_scale"],
        full_gumbel=_CFG["full_gumbel"],
        # Tree reuse is safe here because each bot is explicitly advanced after
        # every environment move; missing children simply invalidate the cache.
        reuse_tree=True,
        # Evaluation of perfect-information games follows the paper and uses
        # deterministic gumbel=0; temperature protocols supply any desired
        # stochasticity through visit-count sampling.
        gumbel_noise=False,
    )


def _temperature(protocol, ply):
    if protocol == "det":
        return 0.0
    if protocol == "early12":
        return 1.0 if ply < _CFG["temp_moves"] else 0.0
    if protocol == "all1":
        return 1.0
    raise ValueError(protocol)


def _choose(bot, state, protocol, ply, rng):
    result = bot.root_search(state)
    temperature = _temperature(protocol, ply)
    if temperature > 0:
        try:
            return select_by_temperature(result["visits"], temperature, rng)
        except ValueError:
            pass
    action = result["gumbel_action"] if bot.gumbel else result["action"]
    if action is None:
        action = result["action"]
    return int(action)


def _finish(state):
    counts = env.border_counts(state)
    if counts[0] < counts[1]:
        result = "first"
    elif counts[0] > counts[1]:
        result = "second"
    else:
        result = "draw"
    border = env.border_cell_count(state.w, state.h)
    return counts, result, (counts[1] - counts[0]) / border


def play_match(job):
    """候选 vs anchor；candidate_first 由 game parity 固定。"""
    game, seed = job
    candidate_first = (game % 2 == 0)
    cand = _make_bot(_CAND, "gumbel", seed * 2 + 1)
    anchor = _make_bot(_ANCHOR, "puct", seed * 2 + 2)
    rng = np.random.default_rng(seed + 991)
    state = env.create(_CFG["size"], _CFG["size"], 2)
    for ply in range(_CFG["size"] * _CFG["size"] + 1):
        if state.phase == "settled":
            break
        candidate_turn = ((state.cur == 0) == candidate_first)
        bot = cand if candidate_turn else anchor
        action = _choose(bot, state, _CFG["protocol"], ply, rng)
        rec, err = env.apply_move(state, action % state.w, action // state.w)
        if rec is None:
            raise RuntimeError("非法手 %s: %s" % (action, err))
        cand.advance_root(action)
        anchor.advance_root(action)
    counts, first_result, margin_first = _finish(state)
    cand_count = counts[0] if candidate_first else counts[1]
    other_count = counts[1] if candidate_first else counts[0]
    if cand_count < other_count:
        result = "candidate"
    elif cand_count > other_count:
        result = "anchor"
    else:
        result = "draw"
    border = env.border_cell_count(_CFG["size"], _CFG["size"])
    return {
        "game": game, "candidate_first": candidate_first,
        "candidate_count": cand_count, "anchor_count": other_count,
        "candidate_margin": (other_count - cand_count) / border,
        "first_margin": margin_first, "result": result,
        "first_result": first_result,
    }


def play_self(job):
    """候选自战；root_value 是空棋盘当前行动方视角预测。"""
    game, seed = job
    first_bot = _make_bot(_CAND, "gumbel", seed * 2 + 1)
    second_bot = _make_bot(_CAND, "gumbel", seed * 2 + 2)
    rng = np.random.default_rng(seed + 1991)
    state = env.create(_CFG["size"], _CFG["size"], 2)
    root_value = None
    for ply in range(_CFG["size"] * _CFG["size"] + 1):
        if state.phase == "settled":
            break
        bot = first_bot if state.cur == 0 else second_bot
        if ply == 0:
            # 记录同一个搜索结果的 value；不额外改变动作路径。
            result = bot.root_search(state)
            root_value = float(result["value"])
            temperature = _temperature(_CFG["protocol"], ply)
            if temperature > 0:
                try:
                    action = select_by_temperature(result["visits"], temperature, rng)
                except ValueError:
                    action = result["gumbel_action"]
            else:
                action = result["gumbel_action"]
            if action is None:
                action = result["action"]
        else:
            action = _choose(bot, state, _CFG["protocol"], ply, rng)
        rec, err = env.apply_move(state, int(action) % state.w, int(action) // state.w)
        if rec is None:
            raise RuntimeError("非法手 %s: %s" % (action, err))
        first_bot.advance_root(int(action))
        second_bot.advance_root(int(action))
    counts, first_result, margin_first = _finish(state)
    return {
        "game": game, "first_count": counts[0], "second_count": counts[1],
        "first_margin": margin_first, "first_result": first_result,
        "root_value": root_value,
        "root_error": margin_first - root_value if root_value is not None else None,
    }


def _run_pool(fn, jobs, cand_state, anchor_state, channels, blocks, cfg, workers):
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=init_worker,
                  initargs=(cand_state, anchor_state, channels, blocks, cfg)) as pool:
        rows = list(pool.imap_unordered(fn, jobs, chunksize=1))
    rows.sort(key=lambda row: row["game"])
    return rows


def _match_stats(rows):
    def pack(items):
        wins = sum(row["result"] == "candidate" for row in items)
        losses = sum(row["result"] == "anchor" for row in items)
        draws = sum(row["result"] == "draw" for row in items)
        margins = np.asarray([row["candidate_margin"] for row in items], dtype=np.float64)
        return {
            "games": len(items), "candidate_wins": wins,
            "anchor_wins": losses, "draws": draws,
            "candidate_score": round((wins + 0.5 * draws) / max(1, len(items)), 5),
            "candidate_mean_margin": round(float(margins.mean()), 5) if len(margins) else None,
            "candidate_margin_std": round(float(margins.std()), 5) if len(margins) else None,
            "candidate_mean_count": round(float(np.mean([row["candidate_count"] for row in items])), 4)
            if items else None,
            "anchor_mean_count": round(float(np.mean([row["anchor_count"] for row in items])), 4)
            if items else None,
        }

    return {
        "all": pack(rows),
        "candidate_first": pack([row for row in rows if row["candidate_first"]]),
        "candidate_second": pack([row for row in rows if not row["candidate_first"]]),
    }


def _self_stats(rows):
    def score(items, first=True):
        wins = sum((row["first_count"] < row["second_count"]) if first
                   else (row["second_count"] < row["first_count"]) for row in items)
        losses = sum((row["second_count"] < row["first_count"]) if first
                     else (row["first_count"] < row["second_count"]) for row in items)
        draws = len(items) - wins - losses
        return {"games": len(items), "wins": wins, "losses": losses,
                "draws": draws,
                "score": round((wins + 0.5 * draws) / max(1, len(items)), 5)}

    preds = np.asarray([row["root_value"] for row in rows if row["root_value"] is not None], dtype=np.float64)
    actual = np.asarray([row["first_margin"] for row in rows if row["root_value"] is not None], dtype=np.float64)
    errors = actual - preds
    if len(preds) > 1 and np.std(preds) > 1e-12 and np.std(actual) > 1e-12:
        correlation = float(np.corrcoef(preds, actual)[0, 1])
    else:
        correlation = None
    calibration = {
        "games": int(len(preds)),
        "mean_predicted_value": round(float(preds.mean()), 6) if len(preds) else None,
        "mean_actual_margin": round(float(actual.mean()), 6) if len(actual) else None,
        "bias_actual_minus_predicted": round(float(errors.mean()), 6) if len(errors) else None,
        "mae": round(float(np.abs(errors).mean()), 6) if len(errors) else None,
        "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 6) if len(errors) else None,
        "correlation": round(correlation, 6) if correlation is not None else None,
    }
    return {
        "all": score(rows, first=True),
        "first_player": score(rows, first=True),
        "second_player": score(rows, first=False),
        "first_mean_margin": round(float(np.mean([r["first_margin"] for r in rows])), 6),
        "calibration": calibration,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--opening-anchor", required=True,
                    help="固定 opening15/iter210 checkpoint")
    ap.add_argument("--puct-anchor", required=True,
                    help="固定 corrected PUCT/iter210 checkpoint")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--sims", type=int, default=96)
    ap.add_argument("--temp-moves", type=int, default=12)
    ap.add_argument("--gumbel-max-actions", type=int, default=16)
    ap.add_argument("--gumbel-c-visit", type=float, default=50.0)
    ap.add_argument("--gumbel-c-scale", type=float, default=1.0)
    ap.add_argument("--no-full-gumbel", action="store_true")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cand_net, cand_ck = load_net(args.candidate)
    opening_net, opening_ck = load_net(args.opening_anchor)
    puct_net, puct_ck = load_net(args.puct_anchor)
    channels = cand_net.stem[0].out_channels
    blocks = cand_net.blocks_n
    for name, net in (("opening", opening_net), ("puct", puct_net)):
        if net.stem[0].out_channels != channels or net.blocks_n != blocks:
            raise ValueError("%s anchor 与候选网络结构不一致" % name)

    cand_state = {k: v.detach().cpu() for k, v in cand_net.state_dict().items()}
    base_cfg = {
        "size": args.size, "sims": args.sims, "c_puct": 1.5,
        "temp_moves": args.temp_moves,
        "gumbel_max_actions": args.gumbel_max_actions,
        "gumbel_c_visit": args.gumbel_c_visit,
        "gumbel_c_scale": args.gumbel_c_scale,
        "full_gumbel": not args.no_full_gumbel,
    }
    # Use one deterministic seed stream for every protocol/anchor so differences
    # are paired as far as the differing action policies permit.
    seed_rng = random.Random(args.seed)
    jobs = [(g, seed_rng.randrange(1 << 30)) for g in range(args.games)]
    started = time.time()
    report = {
        "candidate": args.candidate, "candidate_iter": cand_ck.get("iter"),
        "opening_anchor": args.opening_anchor, "opening_anchor_iter": opening_ck.get("iter"),
        "puct_anchor": args.puct_anchor, "puct_anchor_iter": puct_ck.get("iter"),
        "size": args.size, "sims": args.sims, "games": args.games,
        "workers": args.workers, "seed": args.seed,
        "gumbel": {"max_actions": args.gumbel_max_actions,
                   "c_visit": args.gumbel_c_visit,
                   "c_scale": args.gumbel_c_scale,
                   "full": not args.no_full_gumbel,
                   "eval_noise": False},
        "protocols": {},
    }

    for protocol in ("det", "early12", "all1"):
        cfg = dict(base_cfg, protocol=protocol)
        print("[%s] candidate vs opening anchor …" % protocol, flush=True)
        opening_rows = _run_pool(
            play_match, jobs, cand_state,
            {k: v.detach().cpu() for k, v in opening_net.state_dict().items()},
            channels, blocks, cfg, args.workers)
        opening_summary = _match_stats(opening_rows)
        print("[%s] candidate vs puct anchor …" % protocol, flush=True)
        puct_rows = _run_pool(
            play_match, jobs, cand_state,
            {k: v.detach().cpu() for k, v in puct_net.state_dict().items()},
            channels, blocks, cfg, args.workers)
        puct_summary = _match_stats(puct_rows)
        print("[%s] candidate self-play …" % protocol, flush=True)
        self_rows = _run_pool(
            play_self, jobs, cand_state, cand_state,
            channels, blocks, cfg, args.workers)
        self_summary = _self_stats(self_rows)
        report["protocols"][protocol] = {
            "opening_anchor": opening_summary,
            "puct_anchor": puct_summary,
            "selfplay": self_summary,
            "opening_detail": opening_rows,
            "puct_detail": puct_rows,
            "selfplay_detail": self_rows,
        }
        print(json.dumps({"protocol": protocol,
                          "opening": opening_summary,
                          "puct": puct_summary,
                          "self": self_summary}, ensure_ascii=False), flush=True)

    report["elapsed_s"] = round(time.time() - started, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
