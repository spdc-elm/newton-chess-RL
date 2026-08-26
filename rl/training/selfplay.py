# -*- coding: utf-8 -*-
"""自博弈：多进程并行生成训练数据。

数据契约（与 docs/rl-interface.md 一致）：
  每个样本 = (canonical 观测 obs, policy target pi, 归一化分差 m)
  传统 PUCT 的 pi 是访问分布；Gumbel 模式的 pi 是 completed-Q policy improvement
  分布。obs/pi/m 均为「该局面当前行动方」视角。
"""
import numpy as np
import torch

import nf_env as env
from bots.nn_bot import NNBot, select_by_temperature


# spawn 子进程本地模板（由 pool_init 注入）
_BOT_TEMPLATE = None


def pool_init(state_dict, planes_in, channels, blocks, device):
    """Pool initializer：每个子进程独立加载一份网络（CPU，单线程）。"""
    global _BOT_TEMPLATE
    torch.set_num_threads(1)
    from training.model import NFNet
    net = NFNet(planes_in, channels, blocks)
    net.load_state_dict(state_dict)
    net = net.to(device).eval()
    _BOT_TEMPLATE = (net, device)


def pool_play(args):
    """Pool 任务：args = (seed, cfg, forced_first_action)。"""
    seed, cfg, forced_first_action = args
    return play_one_game(_BOT_TEMPLATE, cfg, seed, forced_first_action)


def play_one_game(bot_template, cfg, seed, forced_first_action=None):
    """清晰版本：bot 决策时逐手编码当前局面（落子前），返回完整样本。"""
    net, device = bot_template
    w, h, n = cfg["w"], cfg["h"], cfg["n"]
    rng = np.random.default_rng(seed)
    bot = NNBot(net, device=device, sims=cfg["sims"], c_puct=cfg["c_puct"],
                dirichlet_eps=cfg.get("dirichlet_eps", 0.0),
                dirichlet_alpha=cfg.get("dirichlet_alpha", 0.2), temperature=0.0,
                seed=seed, gumbel=cfg.get("gumbel", False),
                gumbel_max_actions=cfg.get("gumbel_max_actions", 16),
                gumbel_c_visit=cfg.get("gumbel_c_visit", 50.0),
                gumbel_c_scale=cfg.get("gumbel_c_scale", 1.0),
                full_gumbel=cfg.get("full_gumbel", True),
                gumbel_noise=cfg.get("gumbel_noise", True),
                reuse_tree=cfg.get("reuse_tree", False))

    s = env.create(w, h, n)
    samples = []
    for ply in range(w * h + 1):
        if s.phase == "settled":
            break
        temp = 1.0 if ply < cfg["temp_moves"] else 0.0
        obs = env.encode_canonical(s)
        if cfg.get("gumbel", False):
            # 一次搜索同时返回 Gumbel improved-policy target 与实际动作，
            # 避免旧版「先 root_visits、再重新搜索」造成 target/action 不一致。
            result = bot.root_search(s)
            vis = result["visits"]
            pi = result["policy"]
            if temp > 0:
                # Gumbel 论文在训练早期仍建议按访问数探索；Gumbel
                # noise/Sequential Halving 则负责根动作的无放回覆盖。
                sampled_action = select_by_temperature(vis, temp, rng)
            else:
                sampled_action = result["gumbel_action"]
                if sampled_action is None:
                    sampled_action = result["action"]
            a = int(sampled_action)
        else:
            vis = bot.root_visits(s)                     # 用「落子前」的局面搜索与编码
            pi = vis / max(1e-6, vis.sum())
            try:
                a = select_by_temperature(vis, temp, rng)
            except ValueError:
                break

        if ply == 0 and forced_first_action is not None:
            # 先完成正常根搜索并保存 π，再覆盖实际第一手；这是数据覆盖干预，
            # 不把指定首着伪装成搜索最优手。
            a = int(forced_first_action)
        rec, err = env.apply_move(s, a % w, a // w)
        assert rec is not None, err
        # 复用这一步已展开的 child；若强制/采样动作未被树访问，
        # NNBot.advance_root 会清缓存，下一手自动走正确的冷树路径。
        bot.advance_root(a)
        samples.append({"obs": obs, "pi": pi.astype(np.float32),
                        "player": rec["player"]})
    counts = env.border_counts(s)
    out = []
    for sm in samples:
        margin = env.margin_for(s, sm["player"])
        out.append((sm["obs"], sm["pi"], np.float32(margin)))
    meta = {"moves": len(samples), "reason": s.reason, "counts": counts,
            "margins": [env.margin_for(s, p) for p in range(n)],
            "forced_first_action": forced_first_action}
    return out, meta


def make_pool_jobs(cfg, seeds):
    openings = cfg.get("opening_actions") or []
    return [(seed, cfg, openings[i % len(openings)] if openings else None)
            for i, seed in enumerate(seeds)]
