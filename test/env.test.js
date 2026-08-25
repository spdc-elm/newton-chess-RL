'use strict';
/* 机器接口（纯逻辑层）测试：合法掩码 / 动作编码 / applyMove / undo / 观测编码 / 校验重放 */
const { loadGame } = require('./helpers/load-game');

let pass = 0, fail = 0;
function eq(name, actual, expected){
  const a = JSON.stringify(actual), b = JSON.stringify(expected);
  if(a === b){ pass++; console.log('PASS', name); }
  else { fail++; console.log('FAIL', name, '\n  got     ', a, '\n  expected', b); }
}

const api = loadGame();

/* ---------- 状态创建与动作编码 ---------- */
const st = api.nfCreateState(9, 9, 2);
eq('createState', [st.w, st.h, st.n, st.cur, st.phase, st.history.length],
   [9, 9, 2, 0, 'playing', 0]);
eq('动作编码往返', api.nfActionToXY(st, api.nfXYToAction(st, 7, 3)), [7, 3]);
eq('动作空间大小', api.nfLegalMask(st).length, 81);

/* ---------- 合法掩码 ---------- */
let mask = api.nfLegalMask(st);
eq('空盘全合法', [...mask].every(v => v === 1), true);
api.nfApplyMove(st, 4, 4);                       // 占住中心
mask = api.nfLegalMask(st);
eq('占用格非法', [mask[4 * 9 + 4], mask[0]], [0, 1]);
eq('合法格数量', [...mask].reduce((a, b) => a + b, 0), 80);

/* ---------- 非法输入 ---------- */
eq('越界动作拒绝', api.nfApplyMove(api.nfCreateState(8, 8, 2), -1, 0).ok, false);
eq('重复落子拒绝', api.nfApplyMove(st, 4, 4).ok, false);

/* ---------- applyMove：推力/轮转/终局（复用规则测试的语义，从机器接口再验一遍） ---------- */
{
  const s = api.nfCreateState(9, 9, 3);
  s.board[4][5] = { id: 1, p: 0 };               // 直接布置：东侧一枚
  const r = api.nfApplyMove(s, 4, 4);
  eq('applyMove 推力', r.pushes, [{ id: 1, fx: 5, fy: 4, tx: 6, ty: 4 }]);
  eq('applyMove 轮转', s.cur, 1);
  eq('applyMove 历史', [s.history.length, s.history[0].player, s.history[0].id],
     [1, 0, r.id]);
}
{
  // 外围满 → 终局；终局时 cur 不再前进
  const s = api.nfCreateState(8, 8, 2);
  for(let y = 0; y < 8; y++) for(let x = 0; x < 8; x++){
    if((x === 0 || y === 0 || x === 7 || y === 7) && !(x === 0 && y === 0))
      s.board[y][x] = { id: x * 8 + y, p: (x + y) % 2 };
  }
  s.cur = 1;
  const r = api.nfApplyMove(s, 0, 0);
  eq('外围满终局', [r.terminal, s.phase, s.reason], [true, 'settled', 'border']);
}

/* ---------- undo 恢复到逐字段一致 ---------- */
{
  const s = api.nfCreateState(9, 9, 2);
  api.nfApplyMove(s, 3, 4);
  api.nfApplyMove(s, 8, 8);
  api.nfApplyMove(s, 2, 4);
  api.nfUndo(s);
  const ref = api.nfCreateState(9, 9, 2);        // 独立重放前两手作为基准
  api.nfApplyMove(ref, 3, 4);
  api.nfApplyMove(ref, 8, 8);
  eq('undo 后与重放前两手逐字段一致（除 seq）',
     [JSON.stringify(s.board), s.cur, s.history.length, s.phase],
     [JSON.stringify(ref.board), ref.cur, ref.history.length, 'playing']);
  eq('undo 无手可撤返回 null', api.nfUndo(api.nfCreateState(8, 8, 2)), null);
}

/* ---------- 边界计分 ---------- */
{
  const s = api.nfCreateState(8, 8, 3);
  s.board[0][0] = { id: 1, p: 0 };
  s.board[0][7] = { id: 2, p: 1 };
  s.board[7][3] = { id: 3, p: 0 };
  s.board[4][4] = { id: 4, p: 2 };               // 内部不计入
  eq('边界计分', api.nfBorderCounts(s), [2, 1, 0]);
}

/* ---------- 观测编码 ---------- */
{
  const s = api.nfCreateState(8, 8, 2);
  s.board[3][3] = { id: 1, p: 1 };               // 对手子
  s.board[5][2] = { id: 2, p: 0 };               // 己方子
  const obs = api.nfEncodeCanonical(s);
  eq('观测形状', [obs.w, obs.h, obs.planes, obs.data.length], [8, 8, 3, 192]);
  const planeOf = (p, x, y) => obs.data[((p === s.cur ? 0 : 1) * 64) + y * 8 + x];
  eq('canonical 当前玩家在第 0 平面', planeOf(0, 2, 5), 1);
  eq('canonical 对手在 第 1 平面', planeOf(1, 3, 3), 1);
  eq('外围掩码面', [obs.data[128], obs.data[128 + 4], obs.data[128 + 36]],
     [1, 1, 0]);
  // 三人局不做视角规范化
  const t = api.nfCreateState(8, 8, 3);
  t.cur = 2;
  t.board[2][2] = { id: 1, p: 2 };
  const obs3 = api.nfEncodeCanonical(t);
  eq('三人局恒等映射', obs3.data[(2 * 64) + 2 * 8 + 2], 1);
  eq('三人局面数', obs3.planes, 4);
}

/* ---------- 校验重放：终局后的多余手数 ---------- */
{
  const border = [];
  for(let x = 0; x < 8; x++){ border.push([x, 0]); border.push([x, 7]); }
  for(let y = 1; y < 7; y++){ border.push([0, y]); border.push([7, y]); }
  eq('恰好填满外围合法', api.validateReplay(8, 8, 2, border), null);
  const extra = [...border, [4, 4]];
  eq('终局后多余手被拒',
     /多余手数/.test(api.validateReplay(8, 8, 2, extra) || ''), true);
}

/* ---------- 确定性：两个独立沙盒同序列 → 完全一致的存档 ---------- */
{
  const moves = [[4,4],[0,0],[8,8],[4,0],[2,2],[6,6],[2,6],[6,2],[1,7],[7,1]];
  const run = () => {
    const g = loadGame();
    g.newGame(9, 9, 2);
    moves.forEach(([x, y]) => g.tapCell(x, y));
    return g.serialize(g.getG());
  };
  eq('确定性重放', run(), run());
}

console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
