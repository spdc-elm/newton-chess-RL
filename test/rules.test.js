'use strict';
/* 核心规则测试：牛顿摆推力 / 边界 / 八向 / 悔棋 / 结算（直接驱动真实游戏 API） */
const { loadGame } = require('./helpers/load-game');

let pass = 0, fail = 0;
function eq(name, actual, expected){
  const a = JSON.stringify(actual), b = JSON.stringify(expected);
  if(a === b){ pass++; console.log('PASS', name); }
  else { fail++; console.log('FAIL', name, '\n  got     ', a, '\n  expected', b); }
}

const api = loadGame();
const G = () => api.getG();
const cell = (x, y) => G().board[y][x];
const lastPushes = () => G().history[G().history.length - 1].pushes;
/* 直接在盘上放置棋子（仅用于搭建测试局面，绕过回合流程） */
function put(x, y, p){
  G().board[y][x] = { id: 1000 + x * 100 + y + Math.floor(Math.random() * 50), p };
}
const shape = g => JSON.stringify(g.board.map(row => row.map(c => c ? c.p : null)));

/* ---------- 初始状态 ---------- */
api.newGame(9, 9, 2);
eq('初始状态', [G().phase, G().cur, G().history.length], ['playing', 0, 0]);

/* ---------- 单枚相邻棋子被推一格 ---------- */
api.newGame(9, 9, 2);
api.tapCell(3, 4);          // P0
api.tapCell(8, 8);          // P1（远处，无互动）
api.tapCell(2, 4);          // P0 落在左侧 → 东推
eq('单枚东推', lastPushes(), [{ id: cell(4, 4).id, fx: 3, fy: 4, tx: 4, ty: 4 }]);
eq('单枚东推后原格空', cell(3, 4), null);
eq('单枚东推归属不变', cell(4, 4).p, 0);

/* ---------- 三连链：只有末端动，中间不动（牛顿摆） ---------- */
api.newGame(9, 9, 2);
put(3, 4, 0); put(4, 4, 1); put(5, 4, 0);
const idEnd = cell(5, 4).id, idMid = cell(4, 4).id, idNear = cell(3, 4).id;
api.tapCell(2, 4);
eq('三连链只推末端', lastPushes(), [{ id: idEnd, fx: 5, fy: 4, tx: 6, ty: 4 }]);
eq('中间不动', [!!cell(4, 4), cell(4, 4).id], [true, idMid]);
eq('近端不动', [!!cell(3, 4), cell(3, 4).id], [true, idNear]);
eq('末端到位', cell(6, 4).id, idEnd);

/* ---------- 链贴边：整体不动 ---------- */
api.newGame(9, 9, 2);
put(7, 4, 0); put(8, 4, 1);
api.tapCell(6, 4);
eq('贴边链不动', lastPushes(), []);
eq('贴边链原位', [!!cell(7, 4), !!cell(8, 4)], [true, true]);

/* ---------- 隔空不相干 ---------- */
api.newGame(9, 9, 2);
put(3, 4, 0); put(5, 4, 1);
const idFar = cell(5, 4).id;
api.tapCell(2, 4);
eq('隔空只推动邻居', lastPushes().length, 1);
eq('远处棋子未动', [cell(5, 4).id, !!cell(4, 4)], [idFar, true]);

/* ---------- 八向同时推（直接布置包围圈） ---------- */
api.newGame(9, 9, 2);
const ring = [[1,3],[2,3],[3,3],[1,4],[3,4],[1,5],[2,5],[3,5]];
ring.forEach(([x, y], i) => put(x, y, i % 2));
api.tapCell(2, 4);
const pushes8 = lastPushes();
eq('八向各推一枚', pushes8.length, 8);
eq('目标格互不冲突', new Set(pushes8.map(p => p.tx + ',' + p.ty)).size, 8);
let ringEmpty = true;
for(let dy = -1; dy <= 1; dy++) for(let dx = -1; dx <= 1; dx++){
  if(!dx && !dy) continue;
  if(cell(2 + dx, 4 + dy)) ringEmpty = false;
}
eq('八邻全被推离', ringEmpty, true);

/* ---------- 角落落子：指向盘外的方向安全跳过 ---------- */
api.newGame(9, 9, 2);
api.tapCell(0, 1); api.tapCell(8, 8);
api.tapCell(0, 0);
eq('角落落子推北侧', lastPushes(), [{ id: cell(0, 2).id, fx: 0, fy: 1, tx: 0, ty: 2 }]);

/* ---------- 对角链传递 ---------- */
api.newGame(9, 9, 2);
put(3, 3, 0); put(4, 2, 1);
const diagId = cell(4, 2).id;
api.tapCell(2, 4);
eq('对角链推末端', lastPushes(), [{ id: diagId, fx: 4, fy: 2, tx: 5, ty: 1 }]);

/* ---------- 占用格拒绝落子 ---------- */
api.newGame(9, 9, 2);
api.tapCell(4, 4);
const nHist = G().history.length;
api.tapCell(4, 4);
eq('占用格不产生手数', [G().history.length, cell(4, 4).p], [nHist, 0]);

/* ---------- 回合轮转 ---------- */
api.newGame(9, 9, 3);
api.tapCell(4, 4); api.tapCell(5, 5); api.tapCell(6, 6); api.tapCell(7, 7);
eq('三人轮转', [G().history.map(h => h.player)], [[0, 1, 2, 0]]);

/* ---------- 悔棋完整还原 ---------- */
api.newGame(9, 9, 2);
api.tapCell(3, 4); api.tapCell(8, 8); api.tapCell(2, 4);
const ref = loadGame();          // 独立沙盒重放前两手作为期望基准
ref.newGame(9, 9, 2);
ref.tapCell(3, 4); ref.tapCell(8, 8);
api.undo();
eq('悔棋回到上一手', [shape(G()), G().history.length, G().cur],
   [shape(ref.getG()), 2, 0]);
api.undo();
eq('连悔两手', G().history.length, 1);

/* ---------- 外围占满自动结算 & 悔棋解锁 ---------- */
api.newGame(8, 8, 2);
for(let y = 0; y < 8; y++) for(let x = 0; x < 8; x++){
  const ringCell = x === 0 || y === 0 || x === 7 || y === 7;
  if(ringCell && !(x === 0 && y === 0)) put(x, y, (x + y) % 2);
}
api.tapCell(0, 0);           // 补上最后一格外围
eq('外围满自动结算', [G().phase, G().reason], ['settled', 'border']);
api.undo();
eq('悔棋解除结算', [G().phase, G().reason, cell(0, 0)], ['playing', null, null]);

/* ---------- 手动结算 & 点棋盘继续 ---------- */
api.newGame(9, 9, 2);
api.tapCell(4, 4);
api.settle('manual');
eq('手动结算', [G().phase, G().reason], ['settled', 'manual']);
api.tapCell(5, 5);           // 手动结算后点空格 = 继续对局并落子
eq('继续对局并落子', [G().phase, !!cell(5, 5)], ['playing', true]);

console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
