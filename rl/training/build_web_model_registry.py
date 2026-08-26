#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建网页内置模型 registry，并把默认模型写入 nf_model.json。

网页仍然保留 ``nf_model.json`` 这个单模型兼容文件；多模型页面使用
``nf_models.json``。四个模型的权重都内联进最终单文件 HTML，CI 不需要
访问外部 checkpoint。
"""
import base64
import datetime
import json
import os
import sys

import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "rl"))

import nf_env as nf  # noqa: E402
from training.model import NFNet  # noqa: E402


SPECS = [
    {
        "id": "gumbel-full-9x9-i380",
        "date": "2026.08.26",
        "label": "2026-08-26 · Gumbel Full",
        "ckpt": "rl/runs/gumbel_full_hotstart_ramp32_96_9x9_7h/ckpt_iter380.pt",
        "card": {
            "summary": "Full Gumbel AlphaZero 训练模型。",
            "details": [
                "训练棋盘：9×9；global iter380",
                "训练搜索：Gumbel-Top-m + Sequential Halving + completed-Q target",
                "训练预算：32→96 渐进，最终 96 sims；D4×8；margin-v2 value",
                "网页端搜索：标准 PUCT Worker",
            ],
            "caution": "这是 Gumbel 训练模型；浏览器原生 Gumbel 搜索留待后续适配。",
        },
    },
    {
        "id": "puct-opening15-9x9-i210",
        "date": "2026.08.25",
        "label": "2026-08-25 · PUCT + opening15",
        "ckpt": "rl/runs/marginv2_lambda4_opening15_d4_9x9_to210/ckpt_iter210.pt",
        "card": {
            "summary": "通过 15 类 D4 不等价首着覆盖改善先手的 PUCT 模型。",
            "details": [
                "训练棋盘：9×9；global iter210",
                "训练搜索：标准 sqrt(N_parent) PUCT，96 sims",
                "训练：15 类 opening coverage；D4×8；margin-v2 value；λ=4",
                "网页搜索：标准 PUCT Worker",
            ],
            "caution": "相较更早版本先手有所改善，但自战中仍有明显后手优势。",
        },
    },
    {
        "id": "puct-corrected-9x9-i210",
        "date": "2026.08.24",
        "label": "2026-08-24 · Corrected PUCT",
        "ckpt": "rl/runs/puctsqrt_9x9_to210/ckpt_iter210.pt",
        "card": {
            "summary": "opening15 之前的修正版 PUCT 历史模型。",
            "details": [
                "训练棋盘：9×9；global iter210",
                "训练搜索：标准 sqrt(N_parent) PUCT，96 sims；没有 opening15",
                "历史 value：outcome-v1（早于 margin-v2 迁移）",
                "网页搜索：标准 PUCT Worker",
            ],
            "caution": "代表严重的先后手失衡阶段；200 局自战先手约 1%、后手约 100%。",
        },
    },
    {
        "id": "stageA-sqrtlog-9x9-i210",
        "date": "2026.08.23",
        "label": "2026-08-23 · Stage A（历史）",
        "ckpt": "rl/runs/stageA_9x9_50k/ckpt_iter210.pt",
        "card": {
            "summary": "早期 Stage A 长训模型，用于展示错误探索公式时期的历史状态。",
            "details": [
                "训练棋盘：9×9；iter210",
                "历史训练搜索：错误的 sqrt(log N_parent) PUCT",
                "历史 value：outcome-v1；没有 margin-v2 迁移",
                "网页端搜索：标准 PUCT Worker（不是历史 sqrt(log N) Worker）",
            ],
            "caution": "探索公式已确认错误，仅建议作为历史对照，不作为默认模型。",
        },
    },
]


def tensor_b64(t):
    a = t.detach().cpu().contiguous().numpy().astype("<f4")
    return base64.b64encode(a.tobytes()).decode("ascii")


def make_model(spec):
    ckpt_path = os.path.join(ROOT, spec["ckpt"])
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["net"]
    channels = state["stem.0.weight"].shape[0]
    blocks = sum(1 for k in state if k.endswith("c1.weight") and k.startswith("trunk"))
    net = NFNet(planes_in=3, channels=channels, blocks=blocks)
    net.load_state_dict(state)
    net.eval()

    tensors = {}
    total = 0
    for name, tensor in net.state_dict().items():
        tensors[name] = {"shape": list(tensor.shape), "b64": tensor_b64(tensor)}
        total += tensor.numel()

    cfg = ck.get("cfg", {})
    value_target = ck.get("value_target") or cfg.get("value_target") or "outcome_v1"
    augmentation = ck.get("augmentation") or "unknown"
    trained_size = cfg.get("size")
    meta = {
        "id": spec["id"],
        "version": spec["date"],
        "label": spec["label"],
        "source": os.path.basename(os.path.dirname(os.path.abspath(ckpt_path))),
        "iter": ck.get("iter"),
        "trained_size": trained_size,
        "params": total,
        "value_target": value_target,
        "augmentation": augmentation,
        "trained_at": spec["date"],
        "exported_at": datetime.datetime.now().replace(microsecond=0).isoformat(timespec="seconds"),
        "card": spec["card"],
    }
    model = {
        "format": "nfnet-web-1",
        "arch": {"planes_in": 3, "channels": net.channels, "blocks": net.blocks_n},
        "meta": meta,
        "tensors": tensors,
    }
    return model, net


def write_vectors(net, path):
    fix_path = os.path.join(ROOT, "rl", "fixtures", "py_games.json")
    with open(fix_path, encoding="utf-8") as f:
        games = json.load(f)["games"]
    seen, cases = set(), []
    with torch.no_grad():
        for game in games:
            if game["n"] != 2:
                continue
            key = (game["w"], game["h"])
            if key in seen:
                continue
            seen.add(key)
            state = nf.replay(nf.parse_save(game["code"]))
            obs = nf.encode_canonical(state)
            logits, value = net(torch.from_numpy(obs).unsqueeze(0))
            cases.append({
                "w": game["w"], "h": game["h"], "n": game["n"],
                "obs": [round(float(x), 5) for x in obs.flatten()],
                "logits": [round(float(x), 5) for x in logits[0].flatten()],
                "value": round(float(value[0]), 5),
            })
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"tolerance": 5e-3, "cases": cases}, f)
    print("✓ 验证向量已写入:", path, "(%d 个尺寸)" % len(cases))


def main():
    models = []
    default_net = None
    for spec in SPECS:
        model, net = make_model(spec)
        models.append(model)
        if default_net is None:
            default_net = net
        print("✓ 已加载:", spec["id"])

    web_dir = os.path.join(ROOT, "rl", "web")
    os.makedirs(web_dir, exist_ok=True)
    registry = {
        "format": "nfnet-web-registry-1",
        "default_id": SPECS[0]["id"],
        "models": models,
    }
    registry_path = os.path.join(web_dir, "nf_models.json")
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, separators=(",", ":"))

    default_model = models[0]
    model_path = os.path.join(web_dir, "nf_model.json")
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(default_model, f, ensure_ascii=False, separators=(",", ":"))
    vectors_path = os.path.join(ROOT, "rl", "fixtures", "web_vectors.json")
    write_vectors(default_net, vectors_path)
    print("✓ 默认模型:", model_path)
    print("✓ 多模型 registry:", registry_path,
          "(%.2f MB)" % (os.path.getsize(registry_path) / 1e6))


if __name__ == "__main__":
    main()
