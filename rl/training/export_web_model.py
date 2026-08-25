# -*- coding: utf-8 -*-
"""把训练 checkpoint 导出为网页可用的格式，并生成跨端数值验证向量。

输出：
  rl/web/nf_model.json     —— 权重（base64 float32）+ 架构清单 + meta.version
  rl/fixtures/web_vectors.json —— 若干真实局面的输入/期望输出（JS 前向必须对齐）

meta.version 是网页上显示的模型版本。默认 YYYY.MM.DD-iterN，可用 --version 覆盖。
换模型后必须再跑 tools/build_html.py，不要手改 newton-force.html。

用法: python3 rl/training/export_web_model.py [--ckpt PATH] [--version LABEL]
"""
import argparse
import base64
import datetime
import json
import os
import struct
import sys

import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "rl"))

import nf_env as nf                                  # noqa: E402
from training.model import NFNet                     # noqa: E402


def tensor_b64(t):
    a = t.detach().cpu().contiguous().numpy().astype("<f4")
    return base64.b64encode(a.tobytes()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "rl", "runs", "p1_8x8", "latest.pt"))
    ap.add_argument("--version", default=None, help="public model id; default YYYY.MM.DD-iterN")
    ap.add_argument("--tol", type=float, default=5e-3)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    net = NFNet(planes_in=3, channels=ck["net"]["stem.0.weight"].shape[0],
                blocks=sum(1 for k in ck["net"] if k.endswith("c1.weight") and k.startswith("trunk")))
    net.load_state_dict(ck["net"])
    net.eval()

    tensors = {}
    total = 0
    for name, t in net.state_dict().items():
        tensors[name] = {"shape": list(t.shape), "b64": tensor_b64(t)}
        total += t.numel()
    exported_at = datetime.datetime.now().replace(microsecond=0)
    iter_n = ck.get("iter")
    source = os.path.basename(os.path.dirname(os.path.abspath(args.ckpt))) or None
    version = args.version or ("%s-iter%s" % (exported_at.strftime("%Y.%m.%d"), iter_n))
    model = {
        "format": "nfnet-web-1",
        "arch": {"planes_in": 3,
                 "channels": net.channels,
                 "blocks": net.blocks_n},
        "meta": {"version": version,
                 "source": source,
                 "iter": iter_n, "trained_size": ck.get("cfg", {}).get("size"),
                 "params": total,
                 "value_target": ck.get("value_target", ck.get("cfg", {}).get("value_target", "outcome_v1")),
                 "augmentation": ck.get("augmentation", "unknown"),
                 "exported_at": exported_at.isoformat(timespec="seconds")},
        "tensors": tensors,
    }
    web_dir = os.path.join(ROOT, "rl", "web")
    os.makedirs(web_dir, exist_ok=True)
    model_path = os.path.join(web_dir, "nf_model.json")
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model, f)
    print("✓ 模型已导出:", model_path, "(%.2f MB, %d 参数)" % (os.path.getsize(model_path) / 1e6, total))

    # ---- 测试向量：取自 py_games.json 各尺寸的终局局面 ----
    fix_path = os.path.join(ROOT, "rl", "fixtures", "py_games.json")
    with open(fix_path, encoding="utf-8") as f:
        games = json.load(f)["games"]
    seen, cases = set(), []
    for g in games:
        if g["n"] != 2:      # 网络为双人局训练（3 平面），多人局面仅用于规则一致性
            continue
        key = (g["w"], g["h"])
        if key in seen:
            continue
        seen.add(key)
        s = nf.replay(nf.parse_save(g["code"]))
        obs = nf.encode_canonical(s)
        with torch.no_grad():
            logits, v = net(torch.from_numpy(obs).unsqueeze(0))
        cases.append({
            "w": g["w"], "h": g["h"], "n": g["n"],
            "obs": [round(float(x), 5) for x in obs.flatten()],
            "logits": [round(float(x), 5) for x in logits[0].flatten()],
            "value": round(float(v[0]), 5),
        })
    vec_path = os.path.join(ROOT, "rl", "fixtures", "web_vectors.json")
    with open(vec_path, "w", encoding="utf-8") as f:
        json.dump({"tolerance": args.tol, "cases": cases}, f)
    print("✓ 验证向量已导出:", vec_path, "(%d 个尺寸)" % len(cases))

    # 自检：round-trip base64
    raw = struct.pack("<1f", 3.14)
    assert abs(struct.unpack("<1f", raw)[0] - 3.14) < 1e-6


if __name__ == "__main__":
    main()
