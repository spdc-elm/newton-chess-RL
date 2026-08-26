# -*- coding: utf-8 -*-
"""bot 战术冒烟测试。

测试 1（环境语义）：近乎填满的外围上落最后一格，立即终局且按规则判定胜负。
测试 2（贪心战术）：存在唯一能把「对方棋子推进边缘」从而改善边界分差的着法时，
       贪心 bot 必须选中它（该信号正是本游戏的核心战术，也是奖励设计的物理基础）。
说明：纯 MCTS 的「取胜手优先」无法用单空格构造判别场景——若某方边界子已锁胜，
所有动作终局相同；其正确性由 Phase 1 的学习曲线与 NN-MCTS 端到端验证兜底。

运行: python3 rl/tests/test_bots.py"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rl"))
import nf_env as env
from bots.greedy_bot import GreedyBot
from bots.mcts_bot import MCTSBot
from bots.nn_bot import NNBot, _puct_score
from training.model import NFNet


def test_terminal_semantics():
    s = env.create(8, 8, 2)
    cells = [(x, 0) for x in range(1, 8)] + \
            [(7, y) for y in range(1, 8)] + \
            [(x, 7) for x in range(7)] + \
            [(0, y) for y in range(1, 7)]
    assert len(cells) == 27, len(cells)
    colors = [0] * 5 + [1] * 22
    for (x, y), c in zip(cells, colors):
        s.board[y][x] = c
    rec, err = env.apply_move(s, 0, 0)
    assert rec and rec["terminal"], (err,)
    # 6 : 22 → 先手方大胜
    assert env.border_counts(s)[0] == 6 and env.border_counts(s)[1] == 22
    assert math.isclose(env.margin_for(s, 0), 16 / 28)
    assert math.isclose(env.margin_for(s, 1), -16 / 28)
    assert env.outcome_for(s, 0) == 1 and env.outcome_for(s, 1) == -1
    print("✓ 终局语义：(0,0) 填满外围，6:22 分差与胜负判定正确")


def test_greedy_finds_border_push():
    """(4,4) 落子会沿南向射线把 (4,6) 的对方子推进底边 (4,7)；
    其余远端着法均不改变边界计分。贪心必须选中 (4,4)。"""
    s = env.create(8, 8, 2)
    s.board[5][4] = 1   # (4,5) 对方
    s.board[6][4] = 1   # (4,6) 对方
    s.cur = 0
    x, y = GreedyBot().select_move(s)
    assert (x, y) == (4, 4), "贪心未发现推子入边的战术手，选了 %s" % ((x, y),)
    print("✓ 贪心选中推对手入边的战术手 (4,4)")


def test_puct_uses_sqrt_parent_visits():
    """PUCT exploration must scale as sqrt(N_parent), not sqrt(log N_parent)."""
    # With P=.2, c=1.5, N_parent=25 and N_child=0, U must be 1.5.
    # sqrt(log(25)) would only give about 0.538, a materially different search.
    u = _puct_score(0.0, 0.2, 25, 0, 1.5)
    assert math.isclose(u, 1.5, rel_tol=1e-12), u
    print("✓ PUCT 探索项使用 sqrt(N_parent)")


def test_gumbel_root_policy_and_state_restoration():
    """Gumbel 根 target 覆盖合法动作，预算守恒，且搜索不改盘面。"""
    torch.manual_seed(13)
    net = NFNet(3, 8, 1).eval()
    s = env.create(8, 8, 2)
    snap = ([row[:] for row in s.board], s.cur, s.history[:], s.phase)
    bot = NNBot(net, device="cpu", sims=8, seed=5, gumbel=True,
                gumbel_max_actions=8, gumbel_noise=False, full_gumbel=True)
    result = bot.root_search(s)
    assert np.isclose(result["visits"].sum(), 8), result["visits"].sum()
    assert np.count_nonzero(result["visits"]) == 8
    assert np.isclose(result["policy"].sum(), 1.0)
    mask = np.frombuffer(env.legal_mask(s), dtype=np.uint8).astype(bool)
    assert np.all(result["policy"][~mask] == 0)
    assert result["gumbel_action"] in np.flatnonzero(mask)
    assert ([r[:] for r in s.board], s.cur, s.history, s.phase) == snap
    print("✓ Gumbel 根搜索预算守恒、completed-Q target 合法且状态还原")


def test_gumbel_tree_reuse():
    """已展开 child 可作为下一根继续搜索，累计访问不会丢失。"""
    torch.manual_seed(17)
    net = NFNet(3, 8, 1).eval()
    s = env.create(8, 8, 2)
    bot = NNBot(net, device="cpu", sims=8, seed=9, gumbel=True,
                gumbel_max_actions=8, gumbel_noise=False, reuse_tree=True)
    first = bot.root_search(s)
    action = first["gumbel_action"]
    rec, err = env.apply_move(s, action % 8, action // 8)
    assert rec is not None, err
    assert bot.advance_root(action)
    second = bot.root_search(s)
    assert second["visits"].sum() >= 8
    assert second["gumbel_action"] in np.flatnonzero(
        np.frombuffer(env.legal_mask(s), dtype=np.uint8).astype(bool))
    print("✓ Gumbel 已展开子树可安全复用为下一根")


def test_mcts_state_restoration():
    s = env.create(8, 8, 2)
    env.apply_move(s, 4, 4)
    snap = ([row[:] for row in s.board], s.cur, s.history[:])
    bot = MCTSBot(sims=48, seed=11)
    bot.select_move(s)
    assert ([r[:] for r in s.board], s.cur, s.history) == snap, "搜索后状态未还原"
    print("✓ MCTS 搜索后状态完整还原")


if __name__ == "__main__":
    test_terminal_semantics()
    test_greedy_finds_border_push()
    test_puct_uses_sqrt_parent_visits()
    test_gumbel_root_policy_and_state_restoration()
    test_gumbel_tree_reuse()
    test_mcts_state_restoration()
    print("全部通过")
