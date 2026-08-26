'use strict';
/* 导入/导出序列化测试：往返、重放一致性、篡改检测、容错（直接驱动真实游戏 API） */
const { loadGame } = require('./helpers/load-game');

let pass = 0, fail = 0;
function eq(name, actual, expected){
  const a = JSON.stringify(actual), b = JSON.stringify(expected);
  if(a === b){ pass++; console.log('PASS', name); }
  else { fail++; console.log('FAIL', name, '\n  got     ', a, '\n  expected', b); }
}

const api = loadGame();
const G = () => api.getG();
const shape = g => JSON.stringify(g.board.map(row => row.map(c => c ? c.p : null)));
const coordsOf = g => g.history.map(m => [m.x, m.y]);

/* 贪心落子：候选点若已被（推入的）棋子占据则跳过，直到下满 count 手 */
function playN(w, h, n, candidates, count){
  api.newGame(w, h, n);
  const played = [];
  for(const [x, y] of candidates){
    if(played.length >= count) break;
    if(G().board[y][x]) continue;
    api.tapCell(x, y);
    played.push([x, y]);
  }
  if(played.length < count) throw new Error('候选落点不足：仅成功 ' + played.length + '/' + count);
  return played;
}
const errOf = res => res.error || '';

/* ---------- 往返 + 重放一致 ---------- */
const played = playN(9, 9, 2, [
  [4,4],[0,0],[8,8],[4,0],[0,4],[8,4],[2,2],[6,6],[2,6],[6,2],
  [1,7],[7,1],[3,0],[5,8],[0,2],[8,6],[3,3],[5,5],
], 13);
const g1 = G();
const code1 = api.serialize(api.getG());
const parsed1 = api.parseSave(code1);
eq('解析成功', parsed1.error, undefined);
eq('坐标一致', parsed1.moves, coordsOf(g1));
eq('元信息一致', [parsed1.w, parsed1.h, parsed1.n, parsed1.phase], [9, 9, 2, 'p']);
api.importGame(parsed1);
eq('导入后盘面一致', shape(G()), shape(g1));
eq('导入后手数一致', G().history.length, 13);
eq('导入后行动方一致', G().cur, 13 % 2);

/* ---------- 导入后可继续悔棋 ---------- */
api.undo();
const ref = loadGame();
ref.newGame(9, 9, 2);
played.slice(0, -1).forEach(([x, y]) => ref.tapCell(x, y));
eq('导入后悔棋还原上一手', api.serialize(api.getG()), ref.serialize(ref.getG()));

/* ---------- 三人局 / 大棋盘 / 结算状态 ---------- */
const bigCands = [];
for(let y = 0; y < 19; y++) for(let x = 0; x < 19; x++) bigCands.push([x, y]);
playN(19, 19, 3, bigCands, 30);
api.settle('manual');
const parsed3 = api.parseSave(api.serialize(api.getG()));
eq('大盘三人局往返', [parsed3.error, parsed3.w, parsed3.h, parsed3.n, parsed3.phase, parsed3.moves.length],
   [undefined, 19, 19, 3, 'm', 30]);

/* 外围占满状态导出 */
api.newGame(8, 8, 2);
for(let y = 0; y < 8; y++) for(let x = 0; x < 8; x++){
  const ringCell = x === 0 || y === 0 || x === 7 || y === 7;
  if(ringCell && !(x === 0 && y === 0)) G().board[y][x] = { id: 5000 + x * 8 + y, p: (x + y) % 2 };
}
api.tapCell(0, 0);
eq('外围满状态往返', api.parseSave(api.serialize(api.getG())).phase, 'b');

/* ---------- 篡改 / 截断 / 非法数据 ---------- */
const tampered = code1.slice(0, -3) + (code1.slice(-3) === 'aaa' ? 'bbb' : 'aaa');
eq('篡改被拦截', /校验和/.test(errOf(api.parseSave(tampered))), true);
eq('截断被拦截', /校验和|格式/.test(errOf(api.parseSave(code1.slice(0, -4)))), true);

const dupBody = '9.9.2.p.4444';
const dupCode = 'nf1.' + dupBody + '.' + api.checksum(dupBody);
eq('重复落点被拦截', /落点重复/.test(errOf(api.parseSave(dupCode))), true);

const oobBody = '9.9.2.p.zz';
const oobCode = 'nf1.' + oobBody + '.' + api.checksum(oobBody);
eq('越界坐标被拦截', /超出棋盘/.test(errOf(api.parseSave(oobCode))), true);

eq('垃圾输入被拦截', errOf(api.parseSave('hello world')).length > 0, true);

/* ---------- 容错：大小写 / 空白换行 ---------- */
eq('大写容错', api.parseSave(code1.toUpperCase()).moves, coordsOf(g1));
eq('换行空格容错', api.parseSave('  ' + code1.split('.').join('.\n  ') + '\n').moves, coordsOf(g1));

/* ---------- nf1 golden：canonical (4,2) 显示为 E3，存档字节不变 ---------- */
api.newGame(9, 9, 2);
api.tapCell(4, 2);
eq('coordName(4,2) 为 E3', api.coordName(4, 2), 'E3');
eq('parseSave golden moves', api.parseSave('nf1.9.9.2.p.42.1v6osvr').moves, [[4, 2]]);
eq('serialize (4,2) golden', api.serialize(api.getG()), 'nf1.9.9.2.p.42.1v6osvr');

/* ---------- 空对局往返 ---------- */
api.newGame(8, 8, 6);
const parsed0 = api.parseSave(api.serialize(api.getG()));
eq('空对局往返', [parsed0.error, parsed0.w, parsed0.h, parsed0.n, parsed0.moves.length],
   [undefined, 8, 8, 6, 0]);

console.log('\n示例存档代码:', code1);
console.log(pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
