# -*- coding: utf-8 -*-
"""对弈评测：两个 bot 双色轮换打 N 局，报告胜负与粗略 Elo。
用法示例：
    from bots.random_bot import RandomBot
    from bots.mcts_bot import MCTSBot
    from evaluate import arena
    arena(RandomBot(0), MCTSBot(sims=64), games=40, w=8, h=8)
"""
import nf_env as env

B_PER_ELO = 400.0


def play_one(bot_a, bot_b, w, h, n):
    """bot_a 执先手。返回 'a'/'b'/'draw'。"""
    s = env.create(w, h, n)
    bots = [bot_a, bot_b]
    for _ in range(w * h + 1):
        if s.phase == "settled":
            break
        x, y = bots[s.cur].select_move(s)
        rec, err = env.apply_move(s, x, y)
        if rec is None:
            raise RuntimeError("bot 给出非法手 %s: %s" % ((x, y), err))
    c = env.border_counts(s)
    a, b = c[0], max(c[i] for i in range(1, s.n)) if s.n > 1 else c[0]
    if a < b:
        return "a"
    if a > b:
        return "b"
    return "draw"


def arena(bot_a, bot_b, games=40, w=8, h=8, n=2, verbose=True, tag=""):
    """双色轮换对局。返回统计 dict 与 Elo 差（正 = A 强）。"""
    stats = {"a": 0, "b": 0, "draw": 0}
    for g in range(games):
        first_is_a = (g % 2 == 0)
        r = play_one(bot_a, bot_b, w, h, n) if first_is_a else _swap(play_one(bot_b, bot_a, w, h, n))
        stats[r] += 1
    score_a = (stats["a"] + 0.5 * stats["draw"]) / max(1, games)
    elo_diff = _score_to_elo(score_a)
    if verbose:
        print("%s  %d 局 · A胜 %d / B胜 %d / 平 %d · A 得分率 %.3f · Elo差 %+.0f"
              % (tag or "arena", games, stats["a"], stats["b"], stats["draw"], score_a, elo_diff))
    return {"stats": stats, "score_a": score_a, "elo_diff": elo_diff}


def _swap(r):
    return {"a": "b", "b": "a", "draw": "draw"}[r]


def _score_to_elo(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return B_PER_ELO * _log10(p / (1 - p))


def _log10(x):
    import math
    return math.log10(x)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--w", type=int, default=8)
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--sims", type=int, default=64)
    args = ap.parse_args()
    from bots.random_bot import RandomBot
    from bots.mcts_bot import MCTSBot
    arena(RandomBot(1), MCTSBot(sims=args.sims, seed=2),
          games=args.games, w=args.w, h=args.h,
          tag="random vs pureMCTS(%d)" % args.sims)
