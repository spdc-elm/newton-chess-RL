#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把模块编译成可离线分发的单文件 newton-force.html。

源模块：
  rl/web/nf_replay.js       复盘树纯逻辑（树/游标/稳定 ID/分支/转场计划）
  rl/web/nf_app.js          主线程游戏/UI/人机与指导适配层
  rl/web/nf_model.json      默认模型参数（兼容单模型工具）
  rl/web/nf_models.json     内置模型 registry（选择器与 model card）
  rl/web/nf_forward.js      模型前向
  rl/web/nf_mcts_worker.js  Worker 内规则模拟 + PUCT

产物：
  newton-force.template.html  静态 HTML/CSS 模板（源码）
  newton-force.html            编译产物（不要手改）

用法：
  python3 tools/build_html.py
  python3 tools/build_html.py --remove   # 不注入模型，但仍保留主应用脚本
"""
import argparse
import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
HTML = os.path.join(ROOT, "newton-force.html")
TEMPLATE = os.path.join(ROOT, "newton-force.template.html")
REPLAY = os.path.join(ROOT, "rl", "web", "nf_replay.js")
APP = os.path.join(ROOT, "rl", "web", "nf_app.js")
MODEL = os.path.join(ROOT, "rl", "web", "nf_model.json")
MODEL_REGISTRY = os.path.join(ROOT, "rl", "web", "nf_models.json")
FWD = os.path.join(ROOT, "rl", "web", "nf_forward.js")
WORKER = os.path.join(ROOT, "rl", "web", "nf_mcts_worker.js")

AI_BEGIN = "<!-- ====== NF_AI:BEGIN（由 tools/build_html.py 维护，勿手改） ====== -->"
AI_END = "<!-- ====== NF_AI:END ====== -->"
APP_BEGIN = "<!-- ====== NF_APP:BEGIN（由 tools/build_html.py 维护，勿手改） ====== -->"
APP_END = "<!-- ====== NF_APP:END ====== -->"


def strip_region(text, begin, end):
    """幂等移除一个注入区；也能容忍上次构建中断留下的孤立 BEGIN。"""
    while begin in text:
        before, after = text.split(begin, 1)
        if end in after:
            after = after.split(end, 1)[1]
            text = before + after
        else:
            # malformed/incomplete region: drop the rest of this marker line
            tail = after.split("\n", 1)
            text = before + (tail[1] if len(tail) == 2 else "")
    return text


def load_model_registry():
    """返回 (registry_or_none, default_model)。兼容旧的单模型文件。"""
    if os.path.exists(MODEL_REGISTRY):
        with open(MODEL_REGISTRY, encoding="utf-8") as f:
            registry = json.load(f)
        models = registry.get("models") or []
        by_id = {m.get("meta", {}).get("id"): m for m in models}
        default = by_id.get(registry.get("default_id")) or (models[0] if models else None)
        if default is None:
            raise ValueError("模型 registry 没有可用模型")
        return registry, default
    with open(MODEL, encoding="utf-8") as f:
        return None, json.load(f)


def model_meta():
    _, model = load_model_registry()
    return model.get("meta") or {}



def build_ai_region():
    registry, default_model = load_model_registry()
    with open(FWD, encoding="utf-8") as f:
        fwd_src = f.read()
    with open(WORKER, encoding="utf-8") as f:
        worker_src = f.read()
    worker_bundle = fwd_src + "\n\n/* ===== NF MCTS worker ===== */\n" + worker_src
    meta = default_model.get("meta") or {}
    stamp = "<!-- NF_MODEL version=%s source=%s iter=%s exported_at=%s -->\n" % (
        meta.get("version") or "unknown",
        meta.get("source") or "",
        meta.get("iter") if meta.get("iter") is not None else "",
        meta.get("exported_at") or "",
    )
    if registry is None:
        model_bootstrap = "window.NF_WEB_MODEL = " + json.dumps(default_model, ensure_ascii=False) + ";\n"
    else:
        model_bootstrap = (
            "window.NF_WEB_MODEL_REGISTRY = " + json.dumps(registry, ensure_ascii=False) + ";\n"
            "window.NF_WEB_MODEL = window.NF_WEB_MODEL_REGISTRY.models.find(function(m){"
            "return m.meta && m.meta.id === window.NF_WEB_MODEL_REGISTRY.default_id;"
            "}) || window.NF_WEB_MODEL_REGISTRY.models[0];\n"
        )
    return (
        AI_BEGIN + "\n"
        + stamp
        + "<script>" + model_bootstrap + "</script>\n"
        + "<script>window.NF_MCTS_WORKER_SOURCE = "
        + json.dumps(worker_bundle, ensure_ascii=False) + ";</script>\n"
        + "<script>\n" + fwd_src + "\n</script>\n"
        + AI_END + "\n"
    )


def build_app_region():
    parts = []
    for path in (REPLAY, APP):
        with open(path, encoding="utf-8") as f:
            parts.append(f.read())
    return APP_BEGIN + "\n<script>\n" + "\n".join(parts) + "\n</script>\n" + APP_END + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", action="store_true", help="不注入模型/Worker，仅生成纯双人主应用")
    args = ap.parse_args()

    for path in (TEMPLATE, REPLAY, APP):
        if not os.path.exists(path):
            sys.exit("缺少 " + path)
    if not args.remove:
        for path in (MODEL, FWD, WORKER):
            if not os.path.exists(path):
                sys.exit("缺少 " + path)

    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    html = strip_region(html, AI_BEGIN, AI_END)
    # 兼容旧版本的 marker 名称
    html = strip_region(html,
                        "<!-- ====== NF_AI:BEGIN（由 tools/inject_web_ai.py 维护，勿手改） ====== -->",
                        AI_END)
    html = strip_region(html, APP_BEGIN, APP_END)

    if not args.remove:
        anchor = '<div id="toast"></div>'
        if anchor not in html:
            sys.exit('HTML 中找不到 AI 注入锚点 ' + anchor)
        html = html.replace(anchor, anchor + "\n\n" + build_ai_region(), 1)

    body = "</body>"
    if body not in html:
        sys.exit("HTML 中找不到 </body>")
    html = html.replace(body, build_app_region() + body, 1)

    html = "<!-- GENERATED FILE: edit newton-force.template.html or rl/web/, then run tools/build_html.py -->\n" + html
    with open(HTML, "w", encoding="utf-8") as f:
        f.write(html)
    extra = ""
    if not args.remove and os.path.exists(MODEL):
        meta = model_meta()
        extra = " · model %s" % (meta.get("version") or "unknown")
    print("已构建单文件 HTML → %s (%.2f MB)%s" % (HTML, os.path.getsize(HTML) / 1e6, extra))


if __name__ == "__main__":
    main()
