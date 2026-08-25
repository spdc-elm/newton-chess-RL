'use strict';
/* 复盘树核心场景测试（按重构后的最小测试计划）：
 *   场景 1：三分支 + 嵌套分支，五个节点全部可通过明确 node ID 进入
 *   场景 2：首着分支，root 可有 3 个 children 且全部可选择
 *   场景 3：已有分支重入，tree 与 G 永不脱同步
 *   场景 4：transition plan 严格互逆（forward/backward）
 * 另保留：序列化只导出当前 cursor 路径的回归。
 * 驱动真实游戏 API（DOM 桩沙盒里跑发布产物里的逻辑）。 */
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
/* 测试绕过几何动画锁后落子（浏览器中锁由时钟自然过期） */
const play = (x, y) => { api.flushAnimLock(); api.tapCell(x, y); };
const G = () => api.getG();
const cursor = () => api.getCursor();
const coordsOf = g => g.history.map(m => [m.x, m.y]);
/* 全程不变量：cursor 深度 === G.history.length（G 与树永不脱同步） */
function Tsync(name){
  T(name + '（不变量：depth === G.history.length）',
    api.depthOf(cursor()) === G().history.length);
}

/* ---------- 场景 1：三分支 + 嵌套分支 ---------- */
api.newGame(9, 9, 2);
play(4, 4);                        // A
const A = cursor();
/* 在 A 下建三个平级分支 */
play(0, 0); api.navPrev();
play(5, 5); api.navPrev();
play(8, 8);                        // 当前在第三条分支末端
eq('同节点 3 个 children', A.children.length, 3);

/* 第一条分支内部再建 2 个 children */
const B1 = api.nodeById(A.children[0].id);
api.gotoNodeById(B1.id);
Tsync('进入第一条分支');
play(2, 2); api.navPrev();
play(3, 3);
eq('第一条分支内嵌套 2 个 children', B1.children.length, 2);
eq('嵌套不改变外层 children 数量', A.children.length, 3);

/* 五个节点全部仍可通过明确 node ID 点击进入 */
const ids = [B1.id, ...A.children.map(c => c.id), ...B1.children.map(c => c.id)];
let allReachable = true;
for(const id of ids){
  if(!api.gotoNodeById(id)){ allReachable = false; continue; }
  const nd = api.nodeById(id);
  if(cursor() !== nd) allReachable = false;
  if(coordsOf(G()).length !== api.depthOf(nd)) allReachable = false;
}
T('五个节点全部可通过 node ID 直达且 G 同步', allReachable);
Tsync('遍历后回到有效节点');

/* 点击深层节点会同步整条 selected path，因此主线分母应包含该节点：root→A→B1→nested */
eq('mainLineLength 跟随 selected 路径', api.mainLineLength(), 3);
/* 回到 B1 后，A 的选择更新为 B1，主线应延伸到嵌套分支的当前选择 */
api.gotoNodeById(B1.id);
eq('选中 B1 后主线覆盖嵌套选择', api.mainLineLength(), 3);

/* ---------- 场景 2：首着分支 ---------- */
api.newGame(9, 9, 2);
const root = cursor();
T('root 无 move、深度 0', root.move === null && api.depthOf(root) === 0);
play(4, 4); api.navPrev();
play(5, 5); api.navPrev();
play(6, 6);
eq('root 有 3 个首着分支', root.children.length, 3);
let firstMovesOK = true;
for(const c of root.children){
  api.gotoNodeById(c.id);
  if(coordsOf(G()).length !== 1) firstMovesOK = false;
  if(G().history[0].x !== c.move.x || G().history[0].y !== c.move.y) firstMovesOK = false;
}
T('三条首着分支全部可进入', firstMovesOK);
/* Next 跟随 selectedChildId：最后创建的是 (6,6) */
api.gotoNodeById(root.id);
api.navNext();
eq('Next 进入最近选择的分支 (6,6)', coordsOf(G()), [[6,6]]);
Tsync('root→Next');

/* ---------- 场景 3：已有分支重入 ---------- */
api.newGame(9, 9, 2);
play(4, 4);                        // A
const A3 = cursor();
play(0, 0);                        // B
play(8, 8);                        // C（当前叶）
api.navPrev(); api.navPrev();             // 回到 A
T('回到 A', cursor() === A3);
play(5, 5);                        // 从 A 走出新分支 D
play(2, 2);                        // E
const D = cursor().parent;
eq('D 是 (5,5)', [D.move.x, D.move.y], [5,5]);
eq('A 下现有两条分支', A3.children.length, 2);

/* 直接跳回 A 后重入已有分支 D */
api.gotoNodeById(A3.id);
api.gotoNodeById(D.id);
T('cursor === D', cursor() === D);
eq('G.history === [A,D]', coordsOf(G()), [[4,4],[5,5]]);
Tsync('重入 D');

/* 继续 Next 应沿 D 的选择走到 E，而不是停在半路 */
api.navNext();
T('cursor === E', cursor() === api.nodeById(D.children[0].id));
eq('G.history === [A,D,E]', coordsOf(G()), [[4,4],[5,5],[2,2]]);
Tsync('Next 到 E');

/* 分支切换是 jump，不是伪装的倒带 */
const planSwitch = api.planOf(A3.children[0], D);
eq('跨分支切换计划为 jump', planSwitch.mode, 'jump');
eq('直系子推进计划为 forward', api.planOf(A3, D).mode, 'forward');
eq('直系父后退计划为 backward', api.planOf(D, A3).mode, 'backward');

/* ---------- 场景 4：transition plan 可逆 ---------- */
api.newGame(9, 9, 3);                     // 三人局同样适用
play(5, 4);                        // 一颗孤立棋子（同一支线上）
play(4, 4);                        // 向东推：链末端被推一格
const M = cursor();
const mv = M.move;
T('构造出一手带 pushes 的 move', mv.pushes.length >= 1);
eq('push 几何正确 (5,4)→(6,4)', mv.pushes.map(p => [p.fx,p.fy,p.tx,p.ty]), [[5,4,6,4]]);
const fwd = api.planOf(M.parent, M);
const bwd = api.planOf(M, M.parent);
eq('forward placed = move.id', fwd.move.id, mv.id);
eq('backward removed = move.id', bwd.move.id, mv.id);
eq('forward push 目标 = move.pushes', fwd.move.pushes.map(p => [p.fx,p.fy,p.tx,p.ty]),
   mv.pushes.map(p => [p.fx,p.fy,p.tx,p.ty]));
eq('backward 与 forward 严格互逆', bwd.move.pushes.map(p => [p.tx,p.ty,p.fx,p.fy]),
   [[6,4,5,4]]);                     // 即 forward 的 (from,to) 镜像

/* ---------- 序列化契约：只导出当前 cursor 路径 ---------- */
api.newGame(9, 9, 2);
play(4, 4);
play(5, 5);
const code1 = api.serialize(G());
api.navPrev();
play(1, 1);                        // 变招
const code2 = api.serialize(G());
T('变招后存档不同', code1 !== code2);
T('变招后存档可往返解析', !api.parseSave(code2).error);
eq('存档反映当前路径 2 手', api.parseSave(code2).moves.length, 2);

/* ---------- 高价值集成回归：动画事务 + chrome 刷新 ---------- */
api.newGame(9, 9, 2);
play(4, 4);
let animState = api.getAnimationState();
T('第一手 forward 在完成前保留 drop class',
  animState.pieces.some(p => p.classes.includes('drop')));
T('第一手后棋谱立即刷新',
  api.getMoveTreeHTML().includes('data-nid=') && !api.getMoveTreeHTML().includes('尚无着法'));

api.navPrev();
animState = api.getAnimationState();
T('backward 在完成前保留 fading drop-out',
  animState.fadingCount === 1 && animState.fading.some(classes => classes.includes('drop-out')));

/* Prev/Next 之后 current token 必须随 cursor 改变 */
play(4, 4);
play(0, 0);
const currentBefore = (api.getMoveTreeHTML().match(/class="mtok[^\"]* cur[^\"]*" data-nid="(\d+)"/) || [])[1];
api.navPrev();
const currentAfter = (api.getMoveTreeHTML().match(/class="mtok[^\"]* cur[^\"]*" data-nid="(\d+)"/) || [])[1];
T('Prev 后棋谱 current token 改变', currentBefore && currentAfter && currentBefore !== currentAfter);

/* 当前节点有多个后继时 fork row 也必须立即出现 */
api.navPrev();                              // root
play(5, 5);                                // root 第二首着
api.navPrev();                              // 回 root
T('fork row 在分叉点刷新', api.getForkHTML().includes('forkbtn'));
const firstTreeToken = (api.getMoveTreeHTML().match(/data-nid="(\d+)"/) || [])[1];
T('首着主线 token 排在 root siblings 之前', firstTreeToken === String(api.getTree().root.children[0].id));

/* ---------- 深层节点点击：整条 selected path 同步 ---------- */
api.newGame(9, 9, 2);
play(4, 4); const Adeep = cursor();
play(0, 0);
play(8, 8); const Cdeep = cursor();
api.gotoNodeById(Adeep.id);
play(1, 7); const Ddeep = cursor();
play(7, 1); const Edeep = cursor();
play(2, 5); const Fdeep = cursor();
api.gotoNodeById(Cdeep.id);              // 先切到旧分支
api.gotoNodeById(Fdeep.id);              // 再直接点击另一条深层变例
T('深层点击 cursor = F', cursor() === Fdeep);
eq('深层点击 G.history 跟随 F 路径', coordsOf(G()), [[4,4],[1,7],[7,1],[2,5]]);
T('深层点击后 selected line 包含 cursor', api.depthOf(cursor()) <= api.mainLineLength());
T('深层点击后主线长度至少 4', api.mainLineLength() >= 4);
T('深层点击后 root selected path 指向 F', api.getTree().root.selectedChildId === Adeep.id);
T('深层点击后 A selected path 指向 D', Adeep.selectedChildId === Ddeep.id);

console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
