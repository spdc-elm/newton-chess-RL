# -*- coding: utf-8 -*-
"""贪心 1-ply 基线：选「落子后立刻让 (对方边界子 - 我方边界子) 最大化」的动作。
诊断用途：若它明显强于随机而纯 MCTS(低sims) 不强于随机，则说明是采样强度问题而非实现 bug。"""
import nf_env as env


class GreedyBot:
    def __init__(self, seed=None):
        import random
        self.rng = random.Random(seed)

    def select_move(self, s):
        me = s.cur
        best, best_v = None, None
        for x, y in env.legal_moves(s):
            rec, err = env.apply_move(s, x, y)
            assert rec is not None, err
            c = env.border_counts(s)
            v = c[(me + 1) % s.n] - c[me]     # 对方边界子 − 我方边界子（越大越好）
            env.undo_move(s)
            if best_v is None or v > best_v:
                best_v, best = v, (x, y)
        return best
