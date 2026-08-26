# -*- coding: utf-8 -*-
"""神经网络引导的 AlphaZero / Gumbel AlphaZero 搜索。

默认 :class:`NNBot` 保持仓库原有的 ``sqrt(N_parent)`` PUCT 行为。
传入 ``gumbel=True`` 时，根节点使用 Gumbel-Top-m + Sequential Halving，
并用 completed-Q policy-improvement target；``full_gumbel=True`` 还会在
非根节点使用 completed-policy 的确定性选择，而不是 PUCT。

所有 value 都是「当前行动方视角」的归一化外围分差。搜索过程中换手时
取反，终局直接使用规则结算值。
"""
import math

import numpy as np
import torch

import nf_env as env
from nf_env import DIRS  # noqa: F401 (保持与 env 单一来源)


class _Edge:
    __slots__ = ("P", "N", "W", "child")

    def __init__(self, p):
        self.P = float(p)
        self.N = 0
        self.W = 0.0        # 父节点行动方视角的价值累计
        self.child = None   # _Node | None


class _Node:
    """搜索节点。

    ``value``/``logits`` 是网络在该节点（当前行动方视角）的预测；edges
    只包含合法动作。保存 logits 是为了 Gumbel 的 completed policy 不必重复
    做一次网络前向。
    """
    __slots__ = ("edges", "value", "logits", "policy", "selected", "gumbels")

    def __init__(self, edges, value, logits):
        self.edges = edges
        self.value = float(value)
        self.logits = logits
        self.policy = None
        self.selected = None
        self.gumbels = None


def _puct_score(q, prior, parent_visits, child_visits, c_puct):
    """PUCT score: Q + c_puct · P · sqrt(N_parent) / (1 + N_child).

    parent_visits includes a +1 virtual visit at the root so that the first
    simulation still has a non-zero prior-driven exploration term.
    """
    return q + c_puct * prior * math.sqrt(parent_visits) / (1 + child_visits)


class NNBot:
    def __init__(self, net, device="cpu", sims=64, c_puct=1.5,
                 dirichlet_eps=0.0, dirichlet_alpha=0.2, temperature=0.0,
                 seed=None, gumbel=False, gumbel_max_actions=16,
                 gumbel_c_visit=50.0, gumbel_c_scale=1.0,
                 full_gumbel=True, gumbel_noise=True, reuse_tree=False):
        """构造网络 bot。

        ``gumbel=False`` 是旧版 PUCT。开启 Gumbel 后：

        * 根节点用 Gumbel-Top-m 无放回抽取候选，再以 Sequential Halving
          分配 ``sims`` 次模拟；
        * 根 policy target 是 ``softmax(logits + sigma(completed_Q))``，其中
          未访问动作的 Q 用根 value 补全；
        * ``full_gumbel`` 为真时，非根节点用 completed-policy 的确定性
          选择（论文 Eq. 5.14）；否则非根节点仍用 PUCT；
        * ``gumbel_noise=False`` 用于确定性评测，令根 Gumbel 向量全零。
        * ``reuse_tree=True`` 配合每次落子后的 ``advance_root(action)``，会把
          已搜索的 child 重新作为下一根节点；若动作没有对应 child 则安全退回
          到新树搜索。

        ``temperature`` 仍只影响 ``select_move``：温度大于 0 时按根访问数
        采样，温度为 0 时执行 Gumbel Sequential-Halving 选出的动作。训练
        侧可以直接使用 ``root_search``，在前若干手按访问数探索。
        """
        self.net = net.to(device).eval()
        if device == "cpu":
            # 9×9 batch=1 CPU 推理在 channels-last 下约快 2×；只改内存布局，
            # 不改网络算子或训练目标。跨端误差仍在既有容差内。
            self.net = self.net.to(memory_format=torch.channels_last)
        self.device = device
        self.sims = max(0, int(sims))
        self.c_puct = float(c_puct)
        self.dirichlet_eps = float(dirichlet_eps)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)

        self.gumbel = bool(gumbel)
        self.gumbel_max_actions = max(1, int(gumbel_max_actions))
        self.gumbel_c_visit = float(gumbel_c_visit)
        self.gumbel_c_scale = float(gumbel_c_scale)
        self.full_gumbel = bool(full_gumbel)
        self.gumbel_noise = bool(gumbel_noise)
        self.reuse_tree = bool(reuse_tree)
        self._root_node = None
        self._reuse_ready = False
        if self.gumbel_c_visit < 0 or self.gumbel_c_scale < 0:
            raise ValueError("Gumbel Q 缩放参数必须 >= 0")

    @torch.inference_mode()
    def _eval_all(self, s):
        """网络评估当前局面 → (合法 logits, 先验, value_for_cur)。"""
        x = torch.from_numpy(env.encode_canonical(s)).unsqueeze(0).to(self.device)
        if self.device == "cpu":
            x = x.contiguous(memory_format=torch.channels_last)
        logits_t, v = self.net(x)
        raw = logits_t[0].cpu().numpy().astype(np.float64)
        raw = np.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6)
        mask = np.frombuffer(env.legal_mask(s), dtype=np.uint8).astype(bool)

        # 非法动作在所有后续根/非根计算中都是 -inf；这样不会被 Gumbel
        # 采样或 completed policy 偷偷重新引入。
        logits = np.full(raw.shape, -np.inf, dtype=np.float64)
        logits[mask] = raw[mask]
        legal = np.flatnonzero(mask)
        priors = np.zeros(raw.shape, dtype=np.float64)
        if len(legal):
            shifted = logits[legal] - np.max(logits[legal])
            e = np.exp(np.clip(shifted, -745.0, 0.0))
            total = float(e.sum())
            if not np.isfinite(total) or total <= 0:
                priors[legal] = 1.0 / len(legal)
            else:
                priors[legal] = e / total
        return logits, priors, float(v[0])

    @torch.inference_mode()
    def _eval(self, s):
        """兼容旧调用：网络评估 → ``(先验, value_for_cur)``。"""
        _, priors, value = self._eval_all(s)
        return priors, value

    def _expand(self, s):
        logits, priors, value = self._eval_all(s)
        edges = {i: _Edge(priors[i]) for i in np.flatnonzero(priors > 0)}
        return _Node(edges, value, logits)

    def select_move(self, s):
        """返回 ``(x, y)``；搜索后不修改传入状态。"""
        result = self.root_search(s)
        actions = list(result["edges"].keys())
        if not actions:
            moves = env.legal_moves(s)
            if not moves:
                raise ValueError("没有合法动作")
            return moves[0]
        idx = result["action"]
        return idx % s.w, idx // s.w

    def root_search(self, s):
        """执行一次根搜索并返回可复用的搜索结果。

        返回字典字段：

        ``visits``
            行主序根访问计数向量；Gumbel 只会对 Top-m 候选产生访问。
        ``policy``
            训练 target。旧 PUCT 是归一化 visit counts；Gumbel 是
            completed-Q policy-improvement distribution，覆盖全部合法动作。
        ``action``
            按 bot 的 ``temperature`` 选择的实际动作。
        ``gumbel_action``
            Gumbel 根算法的确定性（给定 Gumbel 向量）最终动作。
        ``edges`` / ``value``
            供诊断使用的根边和网络根 value。
        """
        if self.reuse_tree and self._root_node is not None and self._reuse_ready:
            # The cached node is already the state after the caller's last move.
            # Reuse its edge statistics and spend one more root budget on it.
            node = self._root_node
            if self.gumbel:
                self._gumbel_root_search(s, node)
            else:
                self._apply_dirichlet(node)
                for _ in range(self.sims):
                    self._simulate(s, node)
                node.policy = self._visit_policy(node)
                node.selected = self._pick_action(node.edges, self.temperature) if node.edges else None
        else:
            node = self._search(s)
        self._root_node = node
        self._reuse_ready = False
        visits = np.zeros(s.w * s.h, dtype=np.float32)
        for action, edge in node.edges.items():
            visits[action] = edge.N

        if node.policy is None:
            # 理论上 _search 总会设置；保底避免未来新增模式时返回空 target。
            total = float(visits.sum())
            if total > 0:
                policy = visits / total
            else:
                policy = self._softmax_legal(node.logits)
        else:
            policy = np.asarray(node.policy, dtype=np.float32)

        if self.gumbel:
            # Gumbel 的训练/部署动作是 SH 最终动作；显式温度则保留旧评测
            # 协议的访问数采样，方便做 temp=0 / temp=1 对照。
            if self.temperature > 1e-6:
                action = select_by_temperature(visits, self.temperature, self.rng)
            else:
                action = node.selected
        else:
            action = node.selected

        if action is None:
            legal = list(node.edges.keys())
            if not legal:
                moves = env.legal_moves(s)
                if not moves:
                    raise ValueError("没有合法动作")
                action = moves[0][1] * s.w + moves[0][0]
            else:
                action = legal[0]
        return {"visits": visits, "policy": policy, "action": int(action),
                "gumbel_action": node.selected, "edges": node.edges,
                "value": float(node.value), "logits": node.logits,
                "gumbels": node.gumbels}

    def advance_root(self, action):
        """在环境执行 ``action`` 后，把对应 child 设为下一次搜索根。

        调用方应在 ``env.apply_move`` 成功后调用。动作若没有被当前树展开，
        或当前 bot 没有可复用根，则清空缓存并返回 False；这不会影响正确性，
        下一次 ``root_search`` 会完整重建搜索树。
        """
        if not self.reuse_tree or self._root_node is None:
            return False
        edge = self._root_node.edges.get(int(action))
        if edge is None or edge.child is None:
            self._root_node = None
            self._reuse_ready = False
            return False
        self._root_node = edge.child
        self._reuse_ready = True
        return True

    def clear_tree(self):
        """显式丢弃可复用搜索树（换棋局/换网络时调用）。"""
        self._root_node = None
        self._reuse_ready = False

    def root_visits(self, s):
        """自博弈/旧调用用：搜索并返回根访问计数向量（w*h）。"""
        return self.root_search(s)["visits"]

    def root_policy(self, s):
        """搜索并返回 policy target；Gumbel 时是 completed-Q target。"""
        return self.root_search(s)["policy"]

    def _search(self, s):
        node = self._expand(s)
        if self.gumbel:
            self._gumbel_root_search(s, node)
        else:
            self._apply_dirichlet(node)
            for _ in range(self.sims):
                self._simulate(s, node)
            node.policy = self._visit_policy(node)
            node.selected = self._pick_action(node.edges, self.temperature) if node.edges else None
        return node

    def _apply_dirichlet(self, node):
        """旧 PUCT 根噪声；完整 Gumbel 路线不会调用。"""
        if self.dirichlet_eps <= 0 or not node.edges:
            return
        legal_idx = list(node.edges.keys())
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(legal_idx))
        for j, action in enumerate(legal_idx):
            edge = node.edges[action]
            edge.P = (1.0 - self.dirichlet_eps) * edge.P + self.dirichlet_eps * noise[j]
        total = sum(edge.P for edge in node.edges.values())
        if total > 0:
            for edge in node.edges.values():
                edge.P /= total

    @staticmethod
    def _softmax_legal(logits):
        out = np.zeros_like(logits, dtype=np.float64)
        legal = np.flatnonzero(np.isfinite(logits))
        if len(legal) == 0:
            return out
        shifted = logits[legal] - np.max(logits[legal])
        e = np.exp(np.clip(shifted, -745.0, 0.0))
        total = float(e.sum())
        out[legal] = e / total if np.isfinite(total) and total > 0 else 1.0 / len(legal)
        return out

    @staticmethod
    def _visit_policy(node):
        out = np.zeros_like(node.logits, dtype=np.float64)
        total = sum(edge.N for edge in node.edges.values())
        if total > 0:
            for action, edge in node.edges.items():
                out[action] = edge.N / total
        else:
            out = NNBot._softmax_legal(node.logits)
        return out

    def _pick_action(self, edges, temperature):
        actions = list(edges.keys())
        if not actions:
            return None
        visits = np.array([edges[a].N for a in actions], dtype=np.float64)
        if temperature and temperature > 1e-6:
            w = visits ** (1.0 / temperature)
            if w.sum() <= 0:
                w = np.ones_like(w)
            probs = w / w.sum()
            return int(self.rng.choice(actions, p=probs))
        return int(actions[int(np.argmax(visits))])

    # ---- Gumbel root -------------------------------------------------

    def _q01(self, q):
        """论文的根 Q 缩放采用 [0,1]；当前 value 是 [-1,1] 分差。"""
        return float(np.clip((float(q) + 1.0) * 0.5, 0.0, 1.0))

    def _sigma_q(self, q, max_visits):
        return (self.gumbel_c_visit + float(max_visits)) * self.gumbel_c_scale * self._q01(q)

    @staticmethod
    def _edge_q(edge, fallback):
        return edge.W / edge.N if edge.N > 0 else fallback

    def _completed_policy(self, node):
        """Equation (5.10–5.12): completed Q → improved policy π′."""
        max_visits = max((edge.N for edge in node.edges.values()), default=0)
        scores = np.full_like(node.logits, -np.inf, dtype=np.float64)
        for action, edge in node.edges.items():
            q = self._edge_q(edge, node.value)
            scores[action] = node.logits[action] + self._sigma_q(q, max_visits)
        return self._softmax_legal(scores)

    def _root_score(self, node, action):
        edge = node.edges[action]
        q = self._edge_q(edge, node.value)
        max_visits = max((e.N for e in node.edges.values()), default=0)
        return (float(node.gumbels[action]) + float(node.logits[action]) +
                self._sigma_q(q, max_visits))

    def _gumbel_root_search(self, s, node):
        """Gumbel-Top-m + Sequential Halving at the root.

        We deliberately keep one Gumbel vector for both candidate selection and
        final selection. This is the policy-improvement construction, rather than
        Dirichlet noise pasted onto an ordinary PUCT tree.
        """
        legal = np.asarray(list(node.edges.keys()), dtype=np.int64)
        if len(legal) == 0:
            node.policy = np.zeros_like(node.logits)
            node.selected = None
            node.gumbels = np.zeros_like(node.logits)
            return

        gumbels = np.zeros_like(node.logits, dtype=np.float64)
        if self.gumbel_noise:
            gumbels[legal] = self.rng.gumbel(size=len(legal))
        node.gumbels = gumbels

        # The paper uses m=min(n,16) on Go. Constraining m<=sims ensures that
        # every candidate can receive at least one fresh visit in small-budget
        # experiments; at sims=0 we still provide a legal policy/action fallback.
        budget = int(self.sims)
        m = min(len(legal), self.gumbel_max_actions, max(1, budget))
        order = np.argsort(-(node.logits[legal] + gumbels[legal]), kind="mergesort")
        active = [int(a) for a in legal[order[:m]]]

        remaining = budget
        phase_count = max(1, int(math.ceil(math.log2(m)))) if m > 1 else 1
        phase = 0
        while active and remaining > 0:
            if len(active) == 1:
                # Spend any rounding remainder on the surviving arm.
                for _ in range(remaining):
                    self._simulate_root_action(s, node, active[0])
                remaining = 0
                break

            phases_left = max(1, phase_count - phase)
            visits_each = remaining // (len(active) * phases_left)
            visits_each = max(1, visits_each)
            if visits_each * len(active) > remaining:
                visits_each = remaining // len(active)
            if visits_each <= 0:
                break
            for _ in range(visits_each):
                for action in active:
                    self._simulate_root_action(s, node, action)
            remaining -= visits_each * len(active)

            # Eliminate the lower half by the same g+logit+sigma(Q) score.
            ranked = sorted(active, key=lambda a: self._root_score(node, a), reverse=True)
            keep = max(1, (len(ranked) + 1) // 2)
            active = ranked[:keep]
            phase += 1
            if phase >= phase_count:
                break

        # If rounding stopped before all scheduled phases, continue with the
        # current candidates only when budget remains. Otherwise the most recent
        # active set is the best information available.
        if remaining > 0 and active:
            for i in range(remaining):
                self._simulate_root_action(s, node, active[i % len(active)])
            remaining = 0

        if not active:
            active = [int(legal[0])]
        ranked = sorted(active, key=lambda a: self._root_score(node, a), reverse=True)
        node.selected = int(ranked[0])
        node.policy = self._completed_policy(node)

    # ---- search kernel ----------------------------------------------

    def _simulate_root_action(self, s, node, action):
        """强制一次根 action，然后在其 child 继续完整搜索。"""
        edge = node.edges[action]
        mover = s.cur
        x, y = action % s.w, action // s.w
        rec, err = env.apply_move(s, x, y)
        assert rec is not None, err
        try:
            v_mover = self._after_edge(s, edge, mover)
        finally:
            env.undo_move(s)
        edge.N += 1
        edge.W += v_mover
        return v_mover

    def _after_edge(self, s, edge, mover):
        if s.phase == "settled":
            return float(env.margin_for(s, mover))
        if edge.child is None:
            edge.child = self._expand(s)
            return -edge.child.value
        return -self._simulate(s, edge.child)

    def _simulate(self, s, node):
        """从已展开节点返回「当前行动方视角」价值。"""
        if s.phase == "settled":
            return float(env.margin_for(s, s.cur))
        if not node.edges:
            return float(env.margin_for(s, s.cur)) if s.phase == "settled" else float(node.value)

        if self.gumbel and self.full_gumbel:
            best_a = self._pick_full_gumbel_action(node)
        else:
            parent_visits = sum(edge.N for edge in node.edges.values()) + 1
            best_a, best_e, best_u = None, None, -1e18
            for action, edge in node.edges.items():
                q = edge.W / edge.N if edge.N > 0 else 0.0
                u = _puct_score(q, edge.P, parent_visits, edge.N, self.c_puct)
                if u > best_u:
                    best_a, best_e, best_u = action, edge, u
            # Keep a single path below; assigning best_e here avoids a second
            # dictionary lookup and documents the PUCT branch.
            edge = best_e

        if self.gumbel and self.full_gumbel:
            edge = node.edges[best_a]
        mover = s.cur
        x, y = best_a % s.w, best_a // s.w
        rec, err = env.apply_move(s, x, y)
        assert rec is not None, err
        try:
            v_mover = self._after_edge(s, edge, mover)
        finally:
            env.undo_move(s)

        edge.N += 1
        edge.W += v_mover
        return v_mover

    def _pick_full_gumbel_action(self, node):
        """Full Gumbel non-root rule, Eq. (5.14)."""
        policy = self._completed_policy(node)
        total = sum(edge.N for edge in node.edges.values())
        best_a, best_score = None, -1e18
        denom = 1.0 + total
        for action, edge in node.edges.items():
            score = float(policy[action]) - edge.N / denom
            if score > best_score:
                best_a, best_score = action, score
        return best_a


def select_by_temperature(visits, temperature, rng):
    """自博弈动作采样：温度>0 按 visits^(1/T)，否则 argmax。"""
    visits = np.asarray(visits)
    acts = np.nonzero(visits)[0]
    if len(acts) == 0:
        raise ValueError("无访问记录")
    if temperature and temperature > 1e-6:
        w = visits[acts].astype(np.float64) ** (1.0 / temperature)
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones_like(w)
        probs = w / w.sum()
        return int(rng.choice(acts, p=probs))
    return int(acts[np.argmax(visits[acts])])
