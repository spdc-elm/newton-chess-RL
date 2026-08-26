'use strict';
/* 交点棋盘几何与 canonical 坐标契约：
 *   人类标签、点击热区、坐标标签 DOM、cell 边界 class。
 * 不引入完整 DOM 库；点击映射直接测纯函数 boardCoordFromClient。 */
const { loadGame } = require('./helpers/load-game');

let pass = 0, fail = 0;
function T(name, cond){
  if(cond){ pass++; console.log('PASS', name); }
  else { fail++; console.log('FAIL', name); }
}
function eq(name, actual, expected){
  const a = JSON.stringify(actual), b = JSON.stringify(expected);
  if(a === b){ pass++; console.log('PASS', name); }
  else { fail++; console.log('FAIL', name, '\n  got     ', a, '\n  expected', b); }
}

const api = loadGame();
const xy = (x, y) => ({ x, y });

/* ---------- 人类坐标 ---------- */
eq('9×9 A1', api.coordName(0, 0), 'A1');
eq('9×9 E3', api.coordName(4, 2), 'E3');
eq('9×9 E5', api.coordName(4, 4), 'E5');
eq('9×9 I9', api.coordName(8, 8), 'I9');
eq('8×19 A1', api.coordName(0, 0), 'A1');
eq('8×19 H19', api.coordName(7, 18), 'H19');
eq('19×8 A1', api.coordName(0, 0), 'A1');
eq('19×8 S8', api.coordName(18, 7), 'S8');
eq('coordFile/coordRank 派生', api.coordFile(4) + api.coordRank(2), 'E3');

/* ---------- 点击映射 450×450 9×9 ---------- */
const r0 = { left: 0, top: 0, width: 450, height: 450 };
eq('A1 中心', api.boardCoordFromClient(25, 25, r0, 9, 9), xy(0, 0));
eq('E3 中心', api.boardCoordFromClient(225, 125, r0, 9, 9), xy(4, 2));
eq('E5 中心', api.boardCoordFromClient(225, 225, r0, 9, 9), xy(4, 4));
eq('I9 中心', api.boardCoordFromClient(425, 425, r0, 9, 9), xy(8, 8));

const rOff = { left: 100, top: 40, width: 450, height: 450 };
eq('非零 origin A1', api.boardCoordFromClient(125, 65, rOff, 9, 9), xy(0, 0));
eq('非零 origin E3', api.boardCoordFromClient(325, 165, rOff, 9, 9), xy(4, 2));
eq('左上像素边缘', api.boardCoordFromClient(100, 40, rOff, 9, 9), xy(0, 0));
eq('右下像素边缘 clamp', api.boardCoordFromClient(550, 490, rOff, 9, 9), xy(8, 8));

eq('热区左缘属本格', api.boardCoordFromClient(200, 100, r0, 9, 9), xy(4, 2));
eq('热区右缘属下一格', api.boardCoordFromClient(250, 100, r0, 9, 9), xy(5, 2));
eq('热区上缘属本格', api.boardCoordFromClient(225, 100, r0, 9, 9), xy(4, 2));
eq('热区下缘属下一格', api.boardCoordFromClient(225, 150, r0, 9, 9), xy(4, 3));

eq('越左 clamp', api.boardCoordFromClient(-20, 25, r0, 9, 9), xy(0, 0));
eq('越上 clamp', api.boardCoordFromClient(25, -8, r0, 9, 9), xy(0, 0));
eq('越右 clamp', api.boardCoordFromClient(900, 225, r0, 9, 9), xy(8, 4));
eq('越下 clamp', api.boardCoordFromClient(225, 900, r0, 9, 9), xy(4, 8));

/* 非正方形：8×19，400×475，格子 50×25 */
const rTall = { left: 10, top: 20, width: 400, height: 475 };
eq('非方盘 A1 中心', api.boardCoordFromClient(10 + 25, 20 + 12.5, rTall, 8, 19), xy(0, 0));
eq('非方盘 H19 中心', api.boardCoordFromClient(10 + 375, 20 + 462.5, rTall, 8, 19), xy(7, 18));
eq('非方盘右下 clamp', api.boardCoordFromClient(10 + 400, 20 + 475, rTall, 8, 19), xy(7, 18));

const rWide = { left: 0, top: 0, width: 475, height: 400 };
eq('19×8 A1', api.boardCoordFromClient(12.5, 25, rWide, 19, 8), xy(0, 0));
eq('19×8 S8', api.boardCoordFromClient(462.5, 375, rWide, 19, 8), xy(18, 7));

/* ---------- 标签与 cell class ---------- */
function classCount(names, token){
  const re = new RegExp('\\b' + token + '\\b');
  return names.filter(s => re.test(String(s))).length;
}

api.newGame(9, 9, 2);
eq('9×9 file 数量', api.getFileLabels().length, 9);
eq('9×9 rank 数量', api.getRankLabels().length, 9);
eq('9×9 file 首尾', [api.getFileLabels()[0], api.getFileLabels()[8]], ['A', 'I']);
eq('9×9 rank 首尾', [api.getRankLabels()[0], api.getRankLabels()[8]], ['1', '9']);
eq('9×9 cell 总数', api.getCellClassNames().length, 81);
T('9×9 无 alt 棋盘格', api.getCellClassNames().every(s => !/\balt\b/.test(String(s))));
eq('9×9 first-col', classCount(api.getCellClassNames(), 'first-col'), 9);
eq('9×9 last-col', classCount(api.getCellClassNames(), 'last-col'), 9);
eq('9×9 first-row', classCount(api.getCellClassNames(), 'first-row'), 9);
eq('9×9 last-row', classCount(api.getCellClassNames(), 'last-row'), 9);
eq('9×9 ring', classCount(api.getCellClassNames(), 'ring'), 32);

api.tapCell(4, 2);
T('落子 (4,2) 后棋谱显示 E3', api.getMoveTreeHTML().includes('E3'));
eq('nf1 仍编码 canonical 42', api.serialize(api.getG()), 'nf1.9.9.2.p.42.1v6osvr');

api.navPrev();
api.tapCell(0, 0);
api.navPrev();
T('fork 按钮显示 E3 与 A1',
  api.getForkHTML().includes('E3') && api.getForkHTML().includes('A1'));

api.newGame(19, 8, 2);
eq('19×8 file 数量', api.getFileLabels().length, 19);
eq('19×8 rank 数量', api.getRankLabels().length, 8);
eq('19×8 最右为 S', api.getFileLabels()[18], 'S');
eq('19×8 行号首尾', [api.getRankLabels()[0], api.getRankLabels()[7]], ['1', '8']);
eq('19×8 cell 总数', api.getCellClassNames().length, 152);
eq('19×8 first-col', classCount(api.getCellClassNames(), 'first-col'), 8);
eq('19×8 last-row', classCount(api.getCellClassNames(), 'last-row'), 19);
T('换尺寸后旧 9×9 标签不残留',
  api.getFileLabels().length === 19 && !api.getFileLabels().includes('undefined'));

api.newGame(8, 19, 2);
eq('8×19 file 数量', api.getFileLabels().length, 8);
eq('8×19 rank 数量', api.getRankLabels().length, 19);
eq('8×19 最右为 H', api.getFileLabels()[7], 'H');
eq('8×19 最下行 19', api.getRankLabels()[18], '19');
eq('8×19 cell 总数', api.getCellClassNames().length, 152);
eq('8×19 last-col', classCount(api.getCellClassNames(), 'last-col'), 19);
eq('8×19 first-row', classCount(api.getCellClassNames(), 'first-row'), 8);

console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
