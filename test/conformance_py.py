# -*- coding: utf-8 -*-
"""一致性验证（JS→Python 方向）+ 反向夹具生成（Python→JS）。

用法: python3 test/conformance_py.py
  1) 读取 rl/fixtures/js_games.json，用 Python 实现逐手重放，比对每步盘面、
     终局边界计分与 nf1 存档串（含校验和）；
  2) 用 Python 实现打随机对局，导出 rl/fixtures/py_games.json，
     供 JS 端回验（test/conformance_verify_js.js）。
任一不一致即非零退出。
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rl"))
import nf_env as nf

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FIX = os.path.join(ROOT, "rl", "fixtures")

SIZES = [[8, 8, 2], [8, 8, 2], [9, 9, 2], [9, 9, 3], [10, 12, 2], [12, 12, 4], [19, 19, 5]]
GAMES_PER_SIZE = 2


def flatten(board):
    out = []
    for row in board:
        out.extend(row)
    return out


def play_random(w, h, n, rng):
    s = nf.create(w, h, n)
    moves, steps = [], []
    for _ in range(1000):
        if s.phase != "playing":
            break
        empties = nf.legal_moves(s)
        if not empties:
            break
        x, y = empties[rng.randrange(len(empties))]
        rec, err = nf.apply_move(s, x, y)
        assert rec is not None, (w, h, n, x, y, err)
        moves.append([x, y])
        steps.append({"board": flatten(s.board), "cur": s.cur})
    code = nf.serialize(s)
    return {
        "w": w, "h": h, "n": n,
        "moves": moves,
        "steps": steps,
        "terminal_reason": s.reason if s.phase == "settled" else "not-terminal",
        "border_counts": nf.border_counts(s),
        "cur_final": s.cur,
        "code": code,
    }


def verify_games(games, tag):
    fails = []
    for gi, g in enumerate(games):
        w, h, n = g["w"], g["h"], g["n"]
        s = nf.create(w, h, n)
        for k, (x, y) in enumerate(g["moves"]):
            rec, err = nf.apply_move(s, x, y)
            if rec is None:
                fails.append("%s game%d 第 %d 手非法: %s" % (tag, gi, k + 1, err))
                break
            want = g["steps"][k]
            got = {"board": flatten(s.board), "cur": s.cur}
            if got != {"board": want["board"], "cur": want["cur"]}:
                fails.append("%s game%d 第 %d 手后盘面/行动方不一致" % (tag, gi, k + 1))
                break
        else:
            counts = nf.border_counts(s)
            if counts != list(g["border_counts"]):
                fails.append("%s game%d 边界计分不一致 %s != %s" % (tag, gi, counts, g["border_counts"]))
            if s.cur != g["cur_final"]:
                fails.append("%s game%d 终局行动方不一致" % (tag, gi))
            expect_reason = g["terminal_reason"]
            actual = s.reason if s.phase == "settled" else "not-terminal"
            if actual != expect_reason:
                fails.append("%s game%d 终局原因不一致 %s != %s" % (tag, gi, actual, expect_reason))
            if nf.serialize(s) != g["code"]:
                fails.append("%s game%d nf1 存档串不一致\n  py: %s\n  js: %s" % (tag, gi, nf.serialize(s), g["code"]))
            enc = nf.encode_canonical(s)
            want_enc = g.get("encode_final")
            if want_enc is not None:
                if list(map(float, enc.flatten())) != [float(v) for v in want_enc]:
                    fails.append("%s game%d 观测编码不一致" % (tag, gi))
            parsed = nf.parse_save(g["code"])
            if "error" in parsed:
                fails.append("%s game%d 存档解析失败: %s" % (tag, gi, parsed["error"]))
            elif [list(m) for m in parsed["moves"]] != [list(m) for m in g["moves"]]:
                fails.append("%s game%d 存档解析手数不一致" % (tag, gi))
    return fails


def main():
    failures = []

    # ---- 方向一：验证 JS 生成的对局 ----
    js_path = os.path.join(FIX, "js_games.json")
    with open(js_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("generator") == "js"
    failures += verify_games(data["games"], "JS")
    print("✓ JS→PY 方向: %d 局已重放比对" % len(data["games"]))

    # ---- 方向二：Python 生成夹具，交由 JS 回验 ----
    games = []
    seed = 4242
    for w, h, n in SIZES:
        for _ in range(GAMES_PER_SIZE):
            rng = random.Random(seed)
            seed += 1
            games.append(play_random(w, h, n, rng))
    out = {"generator": "py", "version": 1, "games": games}
    dest = os.path.join(FIX, "py_games.json")
    os.makedirs(FIX, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("✓ PY 夹具已生成:", dest, "（%d 局）" % len(games))

    # ---- Python 自身冒烟：parse_save 的拒绝路径 ----
    bad = [
        "hello world",
        "nf1.7.7.2.p..bad",
        "nf1.9.9.2.x..bad",
        "nf1.9.9.2.p.zz.bad",
    ]
    for b in bad:
        r = nf.parse_save(b)
        if "error" not in r:
            failures.append("parse_save 应拒绝: %r" % b)

    if failures:
        print("\n✗ 一致性失败 %d 项:" % len(failures))
        for f_ in failures:
            print(" -", f_)
        sys.exit(1)
    print("✓ 一致性全部通过")


if __name__ == "__main__":
    main()
