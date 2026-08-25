# -*- coding: utf-8 -*-
"""神经网络引导的 PUCT MCTS（AlphaZero 式搜索）。

与纯 MCTS 的差别：叶子不再随机走到底，而是用网络给出 (先验 π, 价值 v)；
选择用 PUCT：Q + c_puct · P · √N_parent / (1 + N_child)。
价值/观测全部按「当前行动方视角」语义（canonical 观测配套）。

接口与其他 bot 一致：select_move(state) -> (x, y)。
"""
import math

import numpy as np
import torch

import nf_env as env
from nf_env import DIRS  # noqa: F401 (保持与 env 单一来源)


class _Edge:
    __slots__ = ("P", "N", "W", "child")

    def __init__(self, p):
        self.P = p
        self.N = 0
        self.W = 0.0        # 父节点行动方视角的价值累计
        self.child = None


def _puct_score(q, prior, parent_visits, child_visits, c_puct):
    """PUCT score: Q + c_puct * P * sqrt(N_parent) / (1 + N_child).

    parent_visits includes a +1 virtual visit at the root so that the first
    simulation still has a non-zero prior-driven exploration term.
    """
    return q + c_puct * prior * math.sqrt(parent_visits) / (1 + child_visits)


class NNBot:
    def __init__(self, net, device="cpu", sims=64, c_puct=1.5,
                 dirichlet_eps=0.0, dirichlet_alpha=0.2, temperature=0.0,
                 seed=None):
        """temperature>0 时按访问数^(1/temp) 采样；否则取最高访问数。"""
        self.net = net.to(device).eval()
        if device == "cpu":
            # 9×9 batch=1 CPU 推理在 channels-last 下约快 2×；只改内存布局，
            # 不改网络算子或训练目标。跨端误差仍在既有容差内。
            self.net = self.net.to(memory_format=torch.channels_last)
        self.device = device
        self.sims = sims
        self.c_puct = c_puct
        self.dirichlet_eps = dirichlet_eps
        self.dirichlet_alpha = dirichlet_alpha
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    @torch.inference_mode()
    def _eval(self, s):
        """网络评估当前局面 → (先验 dict{(x,y):p}, value_for_cur)。"""
        x = torch.from_numpy(env.encode_canonical(s)).unsqueeze(0).to(self.device)
        if self.device == "cpu":
            x = x.contiguous(memory_format=torch.channels_last)
        logits, v = self.net(x)
        mask = np.frombuffer(env.legal_mask(s), dtype=np.uint8).astype(np.float64)
        lg = logits[0].cpu().numpy().astype(np.float64)
        lg = lg - lg.max()
        e = np.exp(lg) * mask
        total = e.sum()
        if total <= 0:                      # 数值兜底：均匀分布
            e = mask
            total = e.sum()
        priors = e / total
        return priors, float(v[0])

    def select_move(self, s):
        edges = self._search(s)
        if not edges:                       # 无合法手（不应发生）
            moves = env.legal_moves(s)
            return moves[0]

        idx = self._pick_action(edges, self.temperature)
        return idx % s.w, idx // s.w

    def root_visits(self, s):
        """自博弈用：搜索并返回根访问计数向量（w*h）。"""
        edges = self._search(s)
        vis = np.zeros(s.w * s.h, dtype=np.float32)
        for a, e in edges.items():
            vis[a] = e.N
        return vis

    def _search(self, s):
        # 根先验（自博弈时加 Dirichlet 噪声促进探索）
        priors, _ = self._eval(s)
        edges = {i: _Edge(p) for i, p in enumerate(priors) if p > 0}
        if self.dirichlet_eps > 0 and edges:
            legal_idx = [i for i, p in enumerate(priors) if p > 0]
            noise = self.rng.dirichlet([self.dirichlet_alpha] * len(legal_idx))
            for j, i in enumerate(legal_idx):
                edges[i].P = (1 - self.dirichlet_eps) * edges[i].P + self.dirichlet_eps * noise[j]
            total = sum(e.P for e in edges.values())
            for e in edges.values():
                e.P /= total
        for _ in range(self.sims):
            self._simulate(s, edges)
        return edges

    def _pick_action(self, edges, temperature):
        actions = list(edges.keys())
        visits = np.array([edges[a].N for a in actions], dtype=np.float64)
        if temperature and temperature > 1e-6:
            w = visits ** (1.0 / temperature)
            if w.sum() <= 0:
                w = np.ones_like(w)
            probs = w / w.sum()
            return int(self.rng.choice(actions, p=probs))
        return int(actions[int(np.argmax(visits))])

    # ---- 搜索内核 ----

    def _simulate(self, s, edges):
        """返回「当前行动方视角」的价值。edges: 当前节点的 {action_idx: _Edge}"""
        if s.phase == "settled":
            return float(env.margin_for(s, s.cur))      # 当前行动方视角的归一化分差

        N_total = sum(e.N for e in edges.values()) + 1

        # PUCT: use sqrt(N_parent), not sqrt(log(N_parent)).  The latter
        # suppresses exploration substantially and changes the training game.
        parent_visits = N_total

        # 选择：PUCT 最优；未访问边 Q 记为 0，由先验探索项自然竞争
        best_a, best_e, best_u = None, None, -1e18
        for a, e in edges.items():
            q = e.W / e.N if e.N > 0 else 0.0
            u = _puct_score(q, e.P, parent_visits, e.N, self.c_puct)
            if u > best_u:
                best_a, best_e, best_u = a, e, u

        x, y = best_a % s.w, best_a // s.w
        mover = s.cur
        rec, err = env.apply_move(s, x, y)
        assert rec is not None, err
        try:
            if s.phase == "settled":
                v_mover = float(env.margin_for(s, mover))
            elif best_e.child is None:
                # 展开：网络评估新叶子，价值取反回传给行动方
                priors, v_child = self._eval(s)
                best_e.child = {i: _Edge(p) for i, p in enumerate(priors) if p > 0}
                v_mover = -v_child
            else:
                v_mover = -self._simulate(s, best_e.child)
        finally:
            env.undo_move(s)

        best_e.N += 1
        best_e.W += v_mover                 # 边价值：父行动方视角
        return v_mover


def select_by_temperature(visits, temperature, rng):
    """自博弈动作采样：温度>0 按 visits^(1/T)，否则 argmax。visits 为 w*h 向量。"""
    acts = np.nonzero(visits)[0]
    if len(acts) == 0:
        raise ValueError("无访问记录")
    if temperature and temperature > 1e-6:
        w = visits[acts].astype(np.float64) ** (1.0 / temperature)
        probs = w / w.sum()
        return int(rng.choice(acts, p=probs))
    return int(acts[np.argmax(visits[acts])])
