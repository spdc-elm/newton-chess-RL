# -*- coding: utf-8 -*-
"""v2 分差目标与 D4 8 倍增强测试。"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rl"))
import nf_env as env
from training.augment import D4_COUNT, expand_d4_batch, opening_orbit_representatives, transform_obs_pi
from training.model import NFNet


def test_margin_semantics():
    s = env.create(8, 8, 2)
    cells = ([(x, 0) for x in range(1, 8)] +
             [(7, y) for y in range(1, 8)] +
             [(x, 7) for x in range(7)] +
             [(0, y) for y in range(1, 7)])
    for (x, y), player in zip(cells, [0] * 5 + [1] * 22):
        s.board[y][x] = player
    rec, err = env.apply_move(s, 0, 0)
    assert rec and rec["terminal"] and not err
    assert env.border_cell_count(8, 8) == 28
    assert np.isclose(env.margin_for(s, 0), 16 / 28)
    assert np.isclose(env.margin_for(s, 1), -16 / 28)
    assert env.outcome_for(s, 0) == 1 and env.outcome_for(s, 1) == -1
    print("✓ 分差语义与双方视角正确")


def test_multiplayer_winner_settlement():
    s = env.create(8, 8, 3)
    ring = ([(x, 0) for x in range(8)] +
            [(7, y) for y in range(1, 8)] +
            [(x, 7) for x in range(6, -1, -1)] +
            [(0, y) for y in range(6, 0, -1)])
    for (x, y), player in zip(ring, [0] * 5 + [1] * 5 + [2] * 18):
        s.board[y][x] = player
    values = env.settlement_values(s)
    # 玩家 2 向并列第一的 0/1 各分一半支付 13；统一除以 (3-1)*28。
    assert np.allclose(values, [6.5 / 56, 6.5 / 56, -13 / 56]), values
    assert np.isclose(sum(values), 0.0)
    print("✓ 多人按第一名分账、并列第一均分且总效用为 0")


def test_opening_orbit_coverage():
    reps = opening_orbit_representatives(9)
    assert len(reps) == 15, reps
    assert len(set(reps)) == 15
    assert all(0 <= a < 81 for a in reps)
    print("✓ 9×9 开局覆盖包含 15 个 D4 不等价首着")


def test_all_eight_d4_transforms_align_policy():
    obs = np.zeros((1, 3, 5, 5), dtype=np.float32)
    pi = np.zeros((1, 25), dtype=np.float32)
    obs[0, 0, 0, 1] = 1
    pi[0, 1] = 1
    positions = set()
    for transform in range(D4_COUNT):
        out_obs, out_pi = transform_obs_pi(obs, pi, transform)
        oy, ox = np.unravel_index(np.argmax(out_obs[0, 0]), (5, 5))
        py, px = np.unravel_index(np.argmax(out_pi[0]), (5, 5))
        assert (oy, ox) == (py, px)
        assert np.isclose(out_pi.sum(), 1)
        positions.add((oy, ox))
    assert len(positions) == 8, positions
    print("✓ 8 种 D4 变换均唯一且 obs/π 坐标对齐")


def test_d4_batch_is_exactly_eightfold():
    obs = np.zeros((2, 3, 5, 5), dtype=np.float32)
    pi = np.zeros((2, 25), dtype=np.float32)
    value = np.asarray([0.25, -0.5], dtype=np.float32)
    obs[:, 0, 0, 1] = 1
    pi[:, 1] = 1
    out_obs, out_pi, out_value = expand_d4_batch(obs, pi, value)
    assert out_obs.shape == (16, 3, 5, 5)
    assert out_pi.shape == (16, 25)
    assert np.array_equal(out_value, np.tile(value, 8))
    print("✓ 每个基础样本完整展开 8 倍，value 不变")


def test_value_head_reset_is_neutral():
    torch.manual_seed(7)
    net = NFNet(3, 16, 1).eval()
    policy_before = {k: v.clone() for k, v in net.state_dict().items()
                     if not k.startswith("value.")}
    net.reset_value_head(zero_output=True)
    for k, v in policy_before.items():
        assert torch.equal(v, net.state_dict()[k]), k
    with torch.no_grad():
        _, value = net(torch.randn(4, 3, 9, 9))
    assert torch.equal(value, torch.zeros_like(value)), value
    print("✓ v1→v2 迁移只重置 value 头，初始输出严格为 0")


def test_cpu_channels_last_parity():
    torch.manual_seed(9)
    net = NFNet(3, 16, 1).eval()
    cl = NFNet(3, 16, 1)
    cl.load_state_dict(net.state_dict())
    cl = cl.to(memory_format=torch.channels_last).eval()
    x = torch.randn(2, 3, 9, 9)
    with torch.inference_mode():
        a = net(x)
        b = cl(x.contiguous(memory_format=torch.channels_last))
    err = max(float((a[i] - b[i]).abs().max()) for i in range(2))
    assert err < 1e-4, err
    print("✓ CPU channels_last 前向与标准布局一致（误差 %.2g）" % err)


if __name__ == "__main__":
    test_margin_semantics()
    test_multiplayer_winner_settlement()
    test_opening_orbit_coverage()
    test_all_eight_d4_transforms_align_policy()
    test_d4_batch_is_exactly_eightfold()
    test_value_head_reset_is_neutral()
    test_cpu_channels_last_parity()
    print("全部通过")
