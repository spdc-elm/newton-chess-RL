#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在固定 v2 replay 上便宜搜索 value-loss 权重 λ。

不生成新自博弈：从同一 checkpoint、同一训练/验证数据、同一 D4 变换顺序训练多个分支。
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "rl"))

from training.augment import D4_COUNT, expand_d4_batch  # noqa: E402
from training.model import NFNet  # noqa: E402


def pick_device(name):
    if name != "auto":
        return name
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_replay(run_dir, max_samples):
    paths = sorted(glob.glob(os.path.join(run_dir, "selfplay", "iter*.npz")))
    if not paths:
        raise FileNotFoundError("没有找到 selfplay/iter*.npz")
    obs, pi, value = [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as d:
            fmt = str(d["format"])
            if fmt != "normalized_border_margin_v2":
                raise ValueError("数据格式不是 v2 margin: %s (%s)" % (fmt, path))
            obs.append(d["obs"])
            pi.append(d["pi"])
            value.append(d["value"])
    obs = np.concatenate(obs, axis=0)
    pi = np.concatenate(pi, axis=0)
    value = np.concatenate(value, axis=0).astype(np.float32)
    if len(obs) > max_samples:
        obs, pi, value = obs[-max_samples:], pi[-max_samples:], value[-max_samples:]
    return obs, pi, value, paths


def make_net(ckpt):
    state = ckpt["net"]
    net = NFNet(3, state["stem.0.weight"].shape[0],
                sum(1 for k in state if k.endswith("c1.weight") and k.startswith("trunk")))
    net.load_state_dict(state)
    return net


def evaluate(net, obs, pi, value, device, batch=512):
    net.eval()
    policy_sum = value_sum = 0.0
    count = 0
    preds, targets = [], []
    with torch.no_grad():
        for i in range(0, len(obs), batch):
            x = torch.from_numpy(obs[i:i + batch].astype(np.float32)).to(device)
            target_pi = torch.from_numpy(pi[i:i + batch]).to(device)
            target_value = torch.from_numpy(value[i:i + batch]).to(device)
            logits, pred = net(x)
            lp = -(target_pi * F.log_softmax(logits, dim=1)).sum(dim=1)
            lv = (pred - target_value).pow(2)
            policy_sum += float(lp.sum())
            value_sum += float(lv.sum())
            count += len(x)
            preds.append(pred.cpu().numpy())
            targets.append(value[i:i + batch])
    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    decisive = np.abs(target) > 1e-8
    first = obs[:, 0].sum((1, 2)) == obs[:, 1].sum((1, 2))

    def pack(mask):
        if not np.any(mask):
            return None
        t, p = target[mask], pred[mask]
        d = np.abs(t) > 1e-8
        return {
            "n": int(mask.sum()),
            "target_mean": round(float(t.mean()), 6),
            "pred_mean": round(float(p.mean()), 6),
            "rmse": round(float(np.sqrt(np.mean((p - t) ** 2))), 6),
            "corr": round(float(np.corrcoef(p, t)[0, 1]), 6) if len(t) > 1 else None,
            "sign_accuracy": round(float((np.sign(p[d]) == np.sign(t[d])).mean()), 6) if np.any(d) else None,
        }

    return {
        "policy_ce": round(policy_sum / max(1, count), 6),
        "margin_mse": round(value_sum / max(1, count), 6),
        "margin_rmse": round(float(np.sqrt(value_sum / max(1, count))), 6),
        "all": pack(np.ones(len(target), dtype=bool)),
        "first": pack(first),
        "second": pack(~first),
    }


def train_candidate(base_net, obs, pi, value, train_n, device, lam, epochs, batch, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = make_net({"net": base_net.state_dict()})
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    base_batch = max(1, batch // D4_COUNT)
    stats = {"steps": 0, "policy_loss": 0.0, "value_loss": 0.0,
             "preclip_norm": 0.0, "clipped_steps": 0, "train_examples": 0}
    n = train_n
    for _ in range(epochs):
        order = np.random.default_rng(seed).permutation(n)
        for i in range(0, n - base_batch + 1, base_batch):
            idx = order[i:i + base_batch]
            x_np, p_np, v_np = expand_d4_batch(
                obs[idx].astype(np.float32), pi[idx], value[idx])
            x = torch.from_numpy(x_np).to(device)
            target_pi = torch.from_numpy(p_np).to(device)
            target_value = torch.from_numpy(v_np).to(device)
            logits, pred = net(x)
            loss_pi = -(target_pi * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss_v = F.mse_loss(pred, target_value)
            loss = loss_pi + lam * loss_v
            opt.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            stats["steps"] += 1
            stats["policy_loss"] += float(loss_pi.detach())
            stats["value_loss"] += float(loss_v.detach())
            stats["preclip_norm"] += float(norm.detach())
            stats["clipped_steps"] += float(norm > 1.0)
            stats["train_examples"] += len(x_np)
    for key in ("policy_loss", "value_loss", "preclip_norm", "clipped_steps"):
        stats[key] /= max(1, stats["steps"])
    stats["value_loss_weight"] = lam
    return net, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="含 selfplay/ 的 v2 run 目录")
    ap.add_argument("--init", required=True, help="v2 初始 checkpoint")
    ap.add_argument("--lambdas", default="1,4", help="逗号分隔，例如 1,2,4,8")
    ap.add_argument("--max-buffer", type=int, default=200000)
    ap.add_argument("--valid", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = pick_device(args.device)
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    if not lambdas or any(x < 0 for x in lambdas):
        raise ValueError("lambdas 必须是非负数字")

    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    base_net = make_net(ck).to("cpu").eval()
    obs, pi, value, paths = load_replay(args.run, args.max_buffer)
    valid_n = min(args.valid, max(1, len(obs) // 5))
    train_n = len(obs) - valid_n
    if train_n < args.batch // D4_COUNT:
        raise ValueError("训练数据太少")
    print("replay files=%d samples=%d train=%d valid=%d device=%s lambdas=%s" %
          (len(paths), len(obs), train_n, valid_n, device, lambdas), flush=True)
    results = {"config": {"run": args.run, "init": args.init, "lambdas": lambdas,
                           "max_buffer": args.max_buffer, "valid": valid_n,
                           "epochs": args.epochs, "batch": args.batch,
                           "device": device, "seed": args.seed},
               "candidates": {}}
    out_dir = os.path.dirname(args.out)
    os.makedirs(os.path.join(out_dir, "lambda_candidates"), exist_ok=True)
    for j, lam in enumerate(lambdas):
        started = time.time()
        net, train_stats = train_candidate(
            base_net, obs, pi, value, train_n, device, lam, args.epochs, args.batch,
            args.seed + j)
        valid = evaluate(net, obs[train_n:], pi[train_n:], value[train_n:], device)
        tag = ("lambda_%g" % lam).replace(".", "p")
        ck_out = os.path.join(out_dir, "lambda_candidates", tag + ".pt")
        torch.save({"cfg": {"value_loss_weight": lam, "epochs": args.epochs,
                             "batch": args.batch, "source_run": args.run},
                    "net": net.state_dict(), "iter": ck.get("iter"),
                    "value_target": "normalized_border_margin_v2",
                    "augmentation": "d4x8", "lambda_sweep": True,
                    "source_checkpoint": args.init}, ck_out)
        results["candidates"][str(lam)] = {
            "checkpoint": ck_out,
            "train": train_stats,
            "validation": valid,
            "elapsed_s": round(time.time() - started, 1),
        }
        print(json.dumps({"lambda": lam, "train": train_stats, "validation": valid,
                          "elapsed_s": results["candidates"][str(lam)]["elapsed_s"]},
                         ensure_ascii=False), flush=True)
        del net
        if device == "mps":
            torch.mps.empty_cache()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
