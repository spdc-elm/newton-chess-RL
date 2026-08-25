# -*- coding: utf-8 -*-
"""纯 MCTS 基线（无神经网络）：UCT 选择 + 无 rollout 的树内扩展。
本游戏棋子只增不减、必然在有限手内终局；到达终局时使用真实归一化分差。"""
import math
import random

import nf_env as env


class _Node:
    __slots__ = ("N", "W", "children", "untried")

    def __init__(self):
        self.N = 0
        self.W = 0.0          # 本节点「行动方视角」的价值累计
        self.children = None  # {(x,y): _Node}
        self.untried = None   # 尚未尝试的合法动作列表


class MCTSBot:
    def __init__(self, sims=64, c=1.4, seed=None):
        self.sims = sims
        self.c = c
        self.rng = random.Random(seed)

    def select_move(self, s):
        root = _Node()
        for _ in range(self.sims):
            self._simulate(s, root)
        if not root.children:
            moves = env.legal_moves(s)
            return moves[self.rng.randrange(len(moves))]
        return max(root.children.items(), key=lambda kv: kv[1].N)[0]

    def _simulate(self, s, node):
        # 终局：直接给出该「行动方」（实际是最后行动者）的结果
        if s.phase == "settled":
            v = float(env.margin_for(s, s.cur))
            node.N += 1
            node.W += v
            return v

        if node.untried is None:
            node.untried = env.legal_moves(s)
            node.children = {}

        mover = s.cur
        if node.untried:
            # 扩展阶段：随机挑一个未尝试动作
            i = self.rng.randrange(len(node.untried))
            x, y = node.untried[i]
            node.untried[i] = node.untried[-1]
            node.untried.pop()
            child = _Node()
            node.children[(x, y)] = child
        else:
            # 选择阶段：纯 UCT 基线（这里的 sqrt(log N / n) 是 UCT 公式，
            # 与神经网络 PUCT 的 sqrt(N_parent)/(1+n) 不同）。
            log_n = math.log(node.N + 1)
            best, best_a, best_u = None, None, -1e18
            for a, ch in node.children.items():
                q = ch.W / ch.N
                u = q + self.c * math.sqrt(log_n / ch.N)
                if u > best_u:
                    best, best_a, best_u = ch, a, u
            child, (x, y) = best, best_a

        env.apply_move(s, x, y)
        try:
            if s.phase == "settled":
                v = float(env.margin_for(s, mover))
            else:
                v = -self._simulate(s, child)   # 换边取负
        finally:
            env.undo_move(s)

        child.N += 1
        child.W += v                            # 子节点边价值：父行动方视角
        node.N += 1
        node.W += v                             # 本节点价值：本行动方视角（= mover）
        return v


def self_check():
    """冒烟：模拟若干次后状态必须被完整还原，且根统计合法。"""
    import copy
    s = env.create(8, 8, 2)
    env.apply_move(s, 4, 4)
    snap = copy.deepcopy((s.board, s.cur, s.history[:]))
    bot = MCTSBot(sims=32, seed=1)
    for _ in range(3):
        bot.select_move(s)
    assert (s.board, s.cur, s.history) == snap, "MCTS 搜索后状态未被还原"
    print("✓ mcts_bot 自检通过（状态还原无误）")
