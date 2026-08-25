# -*- coding: utf-8 -*-
"""
牛顿棋规则核心 —— docs/rl-interface.md 契约的参考实现。
与 newton-force.html 内联纯逻辑层（nf* 函数）语义逐条对应：
任何分歧都应被 test/ 下的一致性夹具捕获。

状态对象 NFState：
  w, h, n          棋盘尺寸(8–19)与玩家数(2–6)
  cur              当前行动玩家 0..n-1（终局时无意义）
  board            h×w 嵌套列表，-1 = 空，否则玩家索引
  history          每手 dict(player, x, y, pushes:[{fx,fy,tx,ty}])
  phase            'playing' | 'settled'
  reason           None | 'border' | 'manual'
"""
import re

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
SAVE_TAG = "nf1"
_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


class NFState:
    __slots__ = ("w", "h", "n", "cur", "board", "history", "phase", "reason")

    def __init__(self, w, h, n):
        self.w, self.h, self.n = w, h, n
        self.cur = 0
        self.board = [[-1] * w for _ in range(h)]
        self.history = []
        self.phase = "playing"
        self.reason = None


def create(w, h, n):
    assert 8 <= w <= 19 and 8 <= h <= 19 and 2 <= n <= 6
    return NFState(w, h, n)


def clone(s):
    c = NFState.__new__(NFState)
    c.w, c.h, c.n, c.cur = s.w, s.h, s.n, s.cur
    c.board = [row[:] for row in s.board]
    c.history = list(s.history)
    c.phase, c.reason = s.phase, s.reason
    return c


def is_ring(x, y, w, h):
    return x == 0 or y == 0 or x == w - 1 or y == h - 1


def compute_pushes(board, w, h, x, y):
    """牛顿摆推力：每条射线上连续链只有末端一枚移动一格；链贴边则整体不动。
    直接原地修改 board。返回推动列表 [{fx,fy,tx,ty}]。"""
    pushes = []
    for dx, dy in DIRS:
        cx, cy = x + dx, y + dy
        if not (0 <= cx < w and 0 <= cy < h):
            continue                      # 方向指向盘外
        if board[cy][cx] == -1:
            continue                      # 相邻为空，无力可传
        hit_edge = False
        while True:                       # 沿射线找连续链末端
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                hit_edge = True
                break
            if board[ny][nx] == -1:
                break                     # 链末端之外是空格
            cx, cy = nx, ny
        if hit_edge:
            continue                      # 链贴边：整体不动
        pc = board[cy][cx]
        board[cy][cx] = -1
        board[cy + dy][cx + dx] = pc
        pushes.append({"fx": cx, "fy": cy, "tx": cx + dx, "ty": cy + dy})
    return pushes


def board_border_full(board, w, h):
    for x in range(w):
        if board[0][x] == -1 or board[h - 1][x] == -1:
            return False
    for y in range(1, h - 1):
        if board[y][0] == -1 or board[y][w - 1] == -1:
            return False
    return True


def is_border_full(s):
    return board_border_full(s.board, s.w, s.h)


def apply_move(s, x, y):
    """合法手 → 推力 → 落子 → 终局判定或轮转。
    返回 (rec|None, error|None)。rec = {player,x,y,pushes,terminal}"""
    if not (0 <= x < s.w and 0 <= y < s.h):
        return None, "out-of-board"
    if s.board[y][x] != -1:
        return None, "occupied"
    pushes = compute_pushes(s.board, s.w, s.h, x, y)
    rec = {"player": s.cur, "x": x, "y": y, "pushes": pushes}
    s.board[y][x] = s.cur
    s.history.append(rec)
    terminal = False
    if board_border_full(s.board, s.w, s.h):
        s.phase, s.reason, terminal = "settled", "border", True
    else:
        s.cur = (s.cur + 1) % s.n
    rec["terminal"] = terminal
    return rec, None


def undo_move(s):
    """撤销最后一手并恢复行动方/阶段；无可撤返回 None。"""
    if not s.history:
        return None
    rec = s.history.pop()
    s.board[rec["y"]][rec["x"]] = -1
    for m in rec["pushes"]:
        pc = s.board[m["ty"]][m["tx"]]
        s.board[m["ty"]][m["tx"]] = -1
        s.board[m["fy"]][m["fx"]] = pc
    s.cur = rec["player"]
    s.phase, s.reason = "playing", None
    return rec


def legal_moves(s):
    return [(x, y) for y in range(s.h) for x in range(s.w) if s.board[y][x] == -1]


def legal_mask(s):
    out = bytearray(s.w * s.h)
    i = 0
    for y in range(s.h):
        for x in range(s.w):
            out[i] = 1 if s.board[y][x] == -1 else 0
            i += 1
    return bytes(out)


def border_cell_count(w, h):
    """外围格数量；8×8 为 28，9×9 为 32。"""
    return 2 * (w + h) - 4


def border_counts(s):
    c = [0] * s.n
    for y in range(s.h):
        for x in range(s.w):
            if is_ring(x, y, s.w, s.h):
                v = s.board[y][x]
                if v != -1:
                    c[v] += 1
    return c


def encode_canonical(s):
    """观测张量 float32[n+1, H, W]：n 张玩家占子面 + 1 张外围掩码面。
    二人局视角规范化：当前玩家恒在第 0 平面。"""
    import numpy as np
    planes = np.zeros((s.n + 1, s.h, s.w), dtype=np.float32)
    if s.n == 2:
        plane_of = lambda p: 0 if p == s.cur else 1
    else:
        plane_of = lambda p: p
    for y in range(s.h):
        row = s.board[y]
        for x in range(s.w):
            v = row[x]
            if v != -1:
                planes[plane_of(v), y, x] = 1.0
    planes[s.n, 0, :] = 1.0
    planes[s.n, s.h - 1, :] = 1.0
    planes[s.n, :, 0] = 1.0
    planes[s.n, :, s.w - 1] = 1.0
    return planes


# ---------------- 存档格式（nf1）----------------

def _b36(v):
    if v == 0:
        return "0"
    out = ""
    while v > 0:
        out = _B36[v % 36] + out
        v //= 36
    return out


def _from_b36(text):
    v = 0
    for ch in text:
        d = _B36.find(ch)
        if d < 0:
            return -1
        v = v * 36 + d
    return v


def checksum(text):
    hh = 5381
    for ch in text:
        hh = (hh * 33 + ord(ch)) & 0xFFFFFFFF
    return _b36(hh)


def serialize(s):
    moves = "".join(_b36(m["x"]) + _b36(m["y"]) for m in s.history)
    phase = "p" if s.phase == "playing" else ("b" if s.reason == "border" else "m")
    body = ".".join([str(s.w), str(s.h), str(s.n), phase, moves])
    return SAVE_TAG + "." + body + "." + checksum(body)


def validate_replay(w, h, n, moves):
    board = [[-1] * w for _ in range(h)]
    terminal_at = -1
    for k, (x, y) in enumerate(moves):
        if terminal_at >= 0:
            return "第 %d 手起对局已结束，存在多余手数" % (k + 1)
        if not (0 <= x < w and 0 <= y < h):
            return "第 %d 手坐标超出棋盘" % (k + 1)
        if board[y][x] != -1:
            return "第 %d 手落点重复，数据无效" % (k + 1)
        compute_pushes(board, w, h, x, y)
        board[y][x] = 0                       # 占位，推力只看占用
        if board_border_full(board, w, h):
            terminal_at = k
    return None


def parse_save(text):
    """解析 nf1 存档。返回 {'w','h','n','phase','moves'} 或 {'error': ...}
    与 JS 版 parseSave 行为一致：去空白、转小写、七段式、校验和、重放验证。"""
    t = re.sub(r"\s+", "", str(text or "")).lower()
    parts = t.split(".")
    if len(parts) != 7 or parts[0] != SAVE_TAG:
        return {"error": "代码格式不正确（应为 %s.宽.高.人数.状态.手数.校验和）" % SAVE_TAG}
    _, ws, hs, ns, ph, mv, ck = parts
    try:
        w, h, n = int(ws), int(hs), int(ns)
    except ValueError:
        return {"error": "棋盘尺寸无效（应为 8–19）"}
    if not (8 <= w <= 19 and 8 <= h <= 19):
        return {"error": "棋盘尺寸无效（应为 8–19）"}
    if not (2 <= n <= 6):
        return {"error": "玩家数量无效（应为 2–6）"}
    if ph not in ("p", "b", "m"):
        return {"error": "对局状态无效"}
    if checksum(ws + "." + hs + "." + ns + "." + ph + "." + mv) != ck:
        return {"error": "校验和不匹配，代码可能不完整或被改动"}
    if len(mv) % 2:
        return {"error": "手数数据不完整"}
    moves = []
    for i in range(0, len(mv), 2):
        x, y = _from_b36(mv[i]), _from_b36(mv[i + 1])
        if not (0 <= x < w and 0 <= y < h):
            return {"error": "第 %d 手坐标超出棋盘" % (i // 2 + 1)}
        moves.append((x, y))
    err = validate_replay(w, h, n, moves)
    if err:
        return {"error": err}
    return {"w": w, "h": h, "n": n, "phase": ph, "moves": moves}


def replay(save):
    """按存档重放整局并返回状态（信任已通过 parse_save 校验的数据）。"""
    s = create(save["w"], save["h"], save["n"])
    for x, y in save["moves"]:
        apply_move(s, x, y)
    if save.get("phase") == "m":
        s.phase, s.reason = "settled", "manual"
    return s


# ---------------- 对局工具 ----------------

def settlement_values(s):
    """按“所有非第一名向第一名支付分差”计算归一化效用向量。

    每个非第一名支付 own_count - best_count；所有收入由并列第一名均分。
    原始效用之和为 0，再统一除以 (n-1)*外围格数，使每项落在 [-1,1]。
    二人局严格退化为 (对手外围数 - 自己外围数) / 外围格数。
    """
    c = border_counts(s)
    if s.n <= 1:
        return [0.0] * s.n
    best = min(c)
    winners = [i for i, count in enumerate(c) if count == best]
    payments = [0.0 if i in winners else float(c[i] - best) for i in range(s.n)]
    pot = sum(payments)
    utility = [-payment for payment in payments]
    share = pot / len(winners)
    for i in winners:
        utility[i] = share
    scale = (s.n - 1) * border_cell_count(s.w, s.h)
    return [u / scale for u in utility]


def margin_for(s, player):
    """player 视角的归一化结算分差；正数更好。"""
    if not 0 <= player < s.n:
        raise ValueError("player 越界")
    return settlement_values(s)[player]


def outcome_for(s, player):
    """兼容旧调用：分差的符号（胜 +1 / 负 −1 / 平 0）。"""
    margin = margin_for(s, player)
    if margin > 0:
        return 1
    if margin < 0:
        return -1
    return 0
