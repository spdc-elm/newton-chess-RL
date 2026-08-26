'use strict';
/* ================= 常量与状态 ================= */
const DIRS = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]];
const NAMES = ['红方','蓝方','绿方','黄方','紫方','橙方'];
const NUM   = ['一','二','三','四','五','六'];

const $ = s => document.querySelector(s);
const boardEl = $('#board'), cellsEl = $('#cells'), piecesEl = $('#pieces'),
      fileLabelsEl = $('#fileLabels'), rankLabelsEl = $('#rankLabels'),
      markerEl = $('#marker'), hintMarkerEl = $('#hintMarker'), scorebarEl = $('#scorebar'),
      statusTextEl = $('#statusText'), fillBarEl = $('#fillBar'), fillTextEl = $('#fillText'),
      hintBtnEl = $('#btnHint'), hintStatusEl = $('#hintStatus'),
      aiProgressRowEl = $('#aiProgressRow'), aiProgressTextEl = $('#aiProgressText'),
      aiProgressFillEl = $('#aiProgressFill'),
      aiValueRowEl = $('#aiValueRow'), aiValueTextEl = $('#aiValueText'),
      overlayEl = $('#overlay'), toastEl = $('#toast'),
      moveTreeEl = $('#moveTreeScroll'), forkRowEl = $('#forkRow'), replayPosEl = $('#replayPos'),
      btnFirstEl = $('#btnFirst'), btnPrevEl = $('#btnPrev'),
      btnNextEl = $('#btnNext'), btnLastEl = $('#btnLast'),
      modelSelectEl = $('#modelSelect'), modelCardEl = $('#modelCard');

let cfg = { w:9, h:9, players:2, sound:true, aiSims:96, modelId:null };   // 待应用设置（新对局生效）
let tree = null;                                  // 复盘树（结构见 nf_replay.js）
let G = null;                                     // 恒等于 tree.cursor.state（唯一 canonical 状态）
const pieceEls = new Map();                       // 稳定 pieceId -> DOM

/* ================= 人机对战状态 ================= */
let aiSeat = -1;          // -1 无 AI；0/1 = AI 执先手/后手
let aiThinking = false;
let aiModel = null;
let aiWorker = null;
let aiWorkerReady = false;
let aiSearchPending = null;
let aiSearchId = 0;
const WEB_MCTS_SIMS = 96;  // 兼容默认值；实际由设置中的 cfg.aiSims 决定
let aiSearchProgress = null;
let aiValue = null;       // { value, player, source }，value 为当前行动方视角的原始网络输出
function webMctsSims(st){
  return Math.max(8, Math.min(1024, cfg.aiSims | 0 || WEB_MCTS_SIMS));
}
let aiHint = null;        // { x, y, player }，仅双人模式的当前回合建议
let aiHintBusy = false;
let aiHintToken = 0;

/* ================= 纯逻辑层（机器接口） =================
 * 本层与 DOM 完全解耦：UI、测试、以及未来的 RL 环境（Python/C++）都以这一层的语义为准。
 * 详细契约见 docs/rl-interface.md。
 *
 * 状态对象 st = {
 *   w, h, n,        棋盘宽高(8–19)与玩家数(2–6)
 *   cur,            当前行动玩家索引 0..n-1（终局时无意义）
 *   seq,            棋子自增 id（仅动画用，RL 可忽略）
 *   phase,          'playing' | 'settled'
 *   reason,         终局原因：'border'(外围满，真正的终局) | 'manual'(人为结算，仅人类界面)
 *   board,          h×w 二维数组，格子为 null 或 { id, p }
 *   history,        每手 { player, x, y, id, pushes:[{id,fx,fy,tx,ty}] }
 * }
 */
function nfCreateState(w, h, n){
  return {
    w, h, n, cur: 0, seq: 0,
    phase: 'playing', reason: null,
    board: Array.from({ length: h }, () => Array(w).fill(null)),
    history: [],
  };
}
function nfCloneState(st){ return JSON.parse(JSON.stringify(st)); }

/* 动作编码：a = y*w + x（行主序）；动作空间恒为 w*h，非法动作由掩码处理 */
function nfXYToAction(st, x, y){ return y * st.w + x; }
function nfActionToXY(st, a){ return [a % st.w, Math.floor(a / st.w)]; }
function nfLegalMask(st){
  const m = new Uint8Array(st.w * st.h);
  for(let y = 0; y < st.h; y++)
    for(let x = 0; x < st.w; x++)
      m[y * st.w + x] = st.board[y][x] ? 0 : 1;
  return m;
}

/* 核心规则：落子推力（牛顿摆）。只读入 board，返回移动列表，不碰其他字段 */
function computePushes(board, w, h, x, y){
  const pushes = [];
  for(const [dx, dy] of DIRS){
    let cx = x + dx, cy = y + dy;
    if(cx < 0 || cy < 0 || cx >= w || cy >= h) continue;     // 方向指向盘外
    if(!board[cy][cx]) continue;                             // 相邻为空，无力可传
    let hitEdge = false;
    for(;;){                                                 // 沿射线找连续链末端
      const nx = cx + dx, ny = cy + dy;
      if(nx < 0 || ny < 0 || nx >= w || ny >= h){ hitEdge = true; break; }
      if(!board[ny][nx]) break;                              // 链末端之外是空格
      cx = nx; cy = ny;
    }
    if(hitEdge) continue;                                    // 链贴边：整体不动
    const pc = board[cy][cx];                                // 牛顿摆：只有末端动
    board[cy][cx] = null;
    board[cy + dy][cx + dx] = pc;
    pushes.push({ id: pc.id, fx: cx, fy: cy, tx: cx + dx, ty: cy + dy });
  }
  return pushes;
}

function nfBoardBorderFull(board, w, h){
  for(let x = 0; x < w; x++){ if(!board[0][x] || !board[h - 1][x]) return false; }
  for(let y = 1; y < h - 1; y++){ if(!board[y][0] || !board[y][w - 1]) return false; }
  return true;
}
function nfIsBorderFull(st){ return nfBoardBorderFull(st.board, st.w, st.h); }
function nfBorderCounts(st){
  const c = Array(st.n).fill(0);
  for(let y = 0; y < st.h; y++)
    for(let x = 0; x < st.w; x++){
      if(x === 0 || y === 0 || x === st.w - 1 || y === st.h - 1){
        const pc = st.board[y][x]; if(pc) c[pc.p]++;
      }
    }
  return c;
}

/* 合法手 → 应用推力 → 落子 → （若外围满）进入终局；否则轮转到下一位玩家。
 * 注意：本函数不检查 phase，阶段门控由调用方负责。 */
function nfApplyMove(st, x, y){
  if(!(x >= 0 && x < st.w && y >= 0 && y < st.h)) return { ok: false, error: 'out-of-board' };
  if(st.board[y][x]) return { ok: false, error: 'occupied' };
  const pushes = computePushes(st.board, st.w, st.h, x, y);
  const id = ++st.seq;
  st.board[y][x] = { id, p: st.cur };
  st.history.push({ player: st.cur, x, y, id, pushes });
  let terminal = false;
  if(nfIsBorderFull(st)){ st.phase = 'settled'; st.reason = 'border'; terminal = true; }
  else st.cur = (st.cur + 1) % st.n;
  return { ok: true, id, pushes, terminal };
}
/* 撤销最后一手并恢复行动方/阶段；无可撤时返回 null */
function nfUndo(st){
  const rec = st.history.pop();
  if(!rec) return null;
  st.board[rec.y][rec.x] = null;
  for(const m of rec.pushes){
    const pc = st.board[m.ty][m.tx];
    st.board[m.ty][m.tx] = null;
    st.board[m.fy][m.fx] = pc;
  }
  st.cur = rec.player;
  st.phase = 'playing'; st.reason = null;
  return rec;
}

/* 观测编码（张量友好）：planes = n 张玩家占子面 + 1 张外围掩码面，行主序展平。
 * 二人局做视角规范化：当前玩家的棋子恒在第 0 平面。
 * 返回 { w, h, planes, data:Float32Array }，与 Python 端 nf_env.encode_canonical 数值一致。 */
function nfEncodeCanonical(st){
  const H = st.h, W = st.w, planes = st.n + 1;
  const data = new Float32Array(planes * H * W);
  const planeOf = st.n === 2 ? (p => (p === st.cur ? 0 : 1)) : (p => p);
  for(let y = 0; y < H; y++)
    for(let x = 0; x < W; x++){
      const pc = st.board[y][x];
      if(pc) data[(planeOf(pc.p) * H + y) * W + x] = 1;
      if(x === 0 || y === 0 || x === W - 1 || y === H - 1)
        data[(st.n * H + y) * W + x] = 1;
    }
  return { w: W, h: H, planes, data };
}

/* ================= 音效（WebAudio，无外部资源） ================= */
let AC = null;
function beep(freq, dur = .07, type = 'triangle', gain = .07, delay = 0){
  if(!cfg.sound) return;
  try{
    AC = AC || new (window.AudioContext || window.webkitAudioContext)();
    if(AC.state === 'suspended') AC.resume();
    const t = AC.currentTime + delay;
    const o = AC.createOscillator(), g = AC.createGain();
    o.type = type;
    o.frequency.setValueAtTime(freq, t);
    o.frequency.exponentialRampToValueAtTime(Math.max(40, freq * .55), t + dur);
    g.gain.setValueAtTime(gain, t);
    g.gain.exponentialRampToValueAtTime(.0001, t + dur);
    o.connect(g).connect(AC.destination);
    o.start(t); o.stop(t + dur + .03);
  }catch(e){}
}
const sfxPlace = () => { beep(300, .06, 'triangle', .09); beep(130, .09, 'sine', .08); };
const sfxPush  = () => beep(210, .07, 'triangle', .06, .05);
const sfxUndo  = () => beep(180, .08, 'sine', .06);
const sfxSettle= () => { beep(392,.12,'triangle',.07); beep(494,.12,'triangle',.07,.13); beep(587,.2,'triangle',.08,.26); };

/* ================= 工具 ================= */
let toastT = 0;
function toast(msg){
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(toastT);
  toastT = setTimeout(() => toastEl.classList.remove('show'), 2100);
}
function saveCfg(){ try{ localStorage.setItem('newtonForceCfg', JSON.stringify(cfg)); }catch(e){} }
function loadCfg(){
  try{
    const c = JSON.parse(localStorage.getItem('newtonForceCfg') || 'null');
    if(c && typeof c === 'object'){
      cfg = { ...cfg, ...c };
      cfg.w = Math.min(19, Math.max(8, cfg.w | 0));
      cfg.h = Math.min(19, Math.max(8, cfg.h | 0));
      cfg.players = Math.min(6, Math.max(2, cfg.players | 0));
      cfg.sound = !!cfg.sound;
      cfg.aiSims = Math.min(1024, Math.max(8, Math.round(Number(cfg.aiSims) || 96)));
      if(cfg.modelId != null) cfg.modelId = String(cfg.modelId);
    }
  }catch(e){}
}
/* 危险按钮：两段式确认 */
function armConfirm(btn, fn){
  if(btn.dataset.armed){
    delete btn.dataset.armed;
    btn.classList.remove('warn');
    btn.innerHTML = btn.dataset.label;
    fn();
    return;
  }
  btn.dataset.armed = '1';
  btn.dataset.label = btn.innerHTML;
  btn.classList.add('warn');
  btn.textContent = '确认？';
  setTimeout(() => {
    if(btn.dataset.armed){
      delete btn.dataset.armed;
      btn.classList.remove('warn');
      btn.innerHTML = btn.dataset.label;
    }
  }, 2300);
}

/* 人类可读坐标：列 A 起、行从上向下（如 (4,2)@9×9 → E3）。
 * canonical：(x 左→右, y 上→下)；此处只做纯函数投影，不解析用户输入。 */
function coordFile(x){
  return String.fromCharCode(65 + x);
}
function coordRank(y){
  return String(y + 1);
}
function coordName(x, y){
  return coordFile(x) + coordRank(y);
}
function boardCoordFromClient(clientX, clientY, rect, w, h){
  const x = Math.min(w - 1, Math.max(
    0,
    Math.floor((clientX - rect.left) / rect.width * w)
  ));
  const y = Math.min(h - 1, Math.max(
    0,
    Math.floor((clientY - rect.top) / rect.height * h)
  ));
  return { x, y };
}
function setSlotPosition(el, x, y, w, h){
  el.style.left = 'calc(' + x + ' * 100% / ' + w + ')';
  el.style.top  = 'calc(' + y + ' * 100% / ' + h + ')';
}
function setSlotBox(el, x, y, w, h){
  setSlotPosition(el, x, y, w, h);
  el.style.width  = 'calc(100% / ' + w + ')';
  el.style.height = 'calc(100% / ' + h + ')';
}

/* ================= 动画事务 =================
 * 全部几何动画（forward/backward）都在一个事务里：
 *   - 开始前 finishAnim() 提交并清理上一个事务（旧 drop/drop-out/pulse、ghost 元素、timeout）；
 *   - 动画期间 navLockedUntil 锁定棋盘与复盘控件约 LOCK_MS，防止快速点击混入新 DOM；
 *   - jump 可随时打断：先强制提交 canonical 状态，再瞬切；
 *   - 每个 geometric 事务只有一个 fallback 定时器，结束时自愈式对齐 canonical。 */
const MOVE_MS = 300, DROP_MS = 340, PULSE_MS = 650, LOCK_MS = 360;
const anim = { token: 0, timer: 0 };
const fadingEls = new Set();
let navLockedUntil = 0;

function purgeFading(){
  for(const el of fadingEls){ try{ el.remove(); }catch(e){} }
  fadingEls.clear();
}
/* 终止当前事务：token 失效 + 清理残留动画类 + 强制对齐 canonical 盘面 */
function finishAnim(){
  anim.token++;
  if(anim.timer){ clearTimeout(anim.timer); anim.timer = 0; }
  navLockedUntil = 0;
  purgeFading();
  for(const el of pieceEls.values())
    el.classList.remove('drop', 'drop-out', 'slide-pulse', 'gone');
  if(G) syncPieces(G, { mode: 'none', move: null });
}
function restartClass(el, cls){
  el.classList.remove(cls);
  void el.offsetWidth;                     // 重启动画
  el.classList.add(cls);
}
/* token 守卫的延时：事务被终止后回调自动失效 */
function animLater(fn, ms){
  const t = anim.token;
  setTimeout(() => { if(anim.token === t) fn(); }, ms);
}

/* ================= 棋盘渲染（唯一 renderer） =================
 * 由转场计划驱动，绝不对比两份重新生成的快照：
 *   forward  : 新落子 = plan.move.id 加 .drop；pushes 从 fx,fy → tx,ty（CSS 过渡）+ pulse；
 *   backward : plan.move.id 反向消失（.drop-out 后移除）；pushes tx,ty → fx,fy + pulse；
 *   jump/none: 无动画快照对齐（缺则建、多则删、错位即贴齐），用于跳步/分支切换/自愈。
 * 稳定 piece id 保证同一逻辑棋子在 live/replay/共享前缀分支间是同一个 DOM 元素。 */
function syncPieces(st, plan){
  plan = plan || { mode: 'jump', move: null };
  const snap = plan.mode === 'none' || plan.mode === 'jump';
  if(snap) piecesEl.classList.add('no-anim');
  const onBoard = new Set();
  const pushes = plan.move ? plan.move.pushes : null;
  for(let y = 0; y < st.h; y++){
    for(let x = 0; x < st.w; x++){
      const pc = st.board[y][x];
      if(!pc) continue;
      onBoard.add(pc.id);
      let el = pieceEls.get(pc.id);
      if(!el){
        el = makePieceEl(pc.p);
        pieceEls.set(pc.id, el);
        piecesEl.appendChild(el);
        setPos(el, x, y);
        if(plan.mode === 'forward' && pc.id === plan.move.id){
          restartClass(el, 'drop');
          animLater(() => el.classList.remove('drop'), DROP_MS);
        }
        continue;
      }
      if(pushes){
        const push = pushes.find(q => q.id === pc.id);
        if(push){
          if(plan.mode === 'forward') setPos(el, push.tx, push.ty);
          else if(plan.mode === 'backward') setPos(el, push.fx, push.fy);
          if(plan.mode === 'forward' || plan.mode === 'backward'){
            restartClass(el, 'slide-pulse');
            animLater(() => el.classList.remove('slide-pulse'), PULSE_MS);
          }
          continue;
        }
      }
      setPos(el, x, y);                    // 未被本手推动：对齐坐标（无位移时为 no-op）
    }
  }
  for(const [id, el] of [...pieceEls]){
    if(onBoard.has(id)) continue;
    pieceEls.delete(id);
    if(plan.mode === 'backward' && plan.move && id === plan.move.id){
      restartClass(el, 'drop-out');
      fadingEls.add(el);
      animLater(() => { fadingEls.delete(el); el.remove(); }, DROP_MS);
    }else{
      el.remove();                          // jump/数据漂移：立即移除，不留残影
    }
  }
  if(snap){ void piecesEl.offsetWidth; piecesEl.classList.remove('no-anim'); }
}

/* 应用一次转场：调用方必须先在 source G 上 finishAnim()，这里不再隐式提交 destination。
 * 这样 forward/backward 的 renderer 仍能从 source DOM 位置开始执行真实几何动画。 */
function startRender(plan){
  if(plan && (plan.mode === 'forward' || plan.mode === 'backward')){
    navLockedUntil = Date.now() + LOCK_MS;
    const t = ++anim.token;
    if(anim.timer) clearTimeout(anim.timer);
    anim.timer = setTimeout(() => {
      if(anim.token !== t) return;
      anim.timer = 0;
      syncPieces(G, { mode: 'none', move: null });   // 自愈：强制对齐 destination canonical
      updateControlsUI();
    }, LOCK_MS + 40);
  }
  syncPieces(G, plan);
  renderGameChrome();
}

/* 唯一游标迁移：先在 source state 结束旧事务，再切换 cursor/G，最后启动新转场。 */
function transitionTo(node){
  if(!node || !tree) return;
  if(node === tree.cursor){
    rtSelectPath(tree, node);
    renderGameChrome();
    return;
  }
  const from = tree.cursor;
  const plan = rtPlanTransition(from, node);
  finishAnim();                              // 此时 G 仍然是 from.state
  rtSelectPath(tree, node);
  tree.cursor = node;
  G = node.state;
  startRender(plan);
}
function gotoNode(node){ transitionTo(node); }

/* ================= 对局流程 ================= */
function newGame(w, h, n){
  clearAIHint();
  finishAnim();                              // 清理旧事务与残影
  tree = rtCreateTree(w, h, n);
  G = tree.root.state;
  if(document.body) document.body.classList.remove('replay');
  hideOverlay();
  buildBoardDOM();
  renderGameChrome();
  maybeAIMove();
}

const isRing = (x, y) => x === 0 || y === 0 || x === G.w - 1 || y === G.h - 1;

/* 在 cursor 落子：
 * - 该点已有 child（无论在不在叶上）：直接进入该分支（直系子 = forward，其余 = jump），
 *   不创建重复分支；
 * - 没有：新建 child（变例追加，不覆盖主线），以普通 forward 进入，绝不先倒带。
 * 人机模式不允许在历史节点走出新变招；外围满终局后不接受新手。 */
function tapCell(x, y, byAI){
  if(!G || !tree) return;
  const locked = Date.now() < navLockedUntil;
  if(locked && !byAI) return;                // 几何动画期间锁定人类输入

  const cur = tree.cursor;
  const existing = rtFindChild(cur, x, y);
  if(existing){ gotoNode(existing); return; }

  if(G.phase === 'settled' && G.reason === 'border'){
    toast('对局已结束（外围占满），可悔棋复盘'); return;
  }
  if(aiSeat >= 0 && !byAI){
    if(aiThinking){ toast('AI 思考中…'); return; }
    if(G.cur === aiSeat){ toast('轮到 AI 落子'); return; }
    if(!rtIsLeaf(cur)){ toast('人机模式下不能在复盘中改走'); return; }
  }
  if(G.phase === 'settled' && G.reason === 'manual'){
    G.phase = 'playing'; G.reason = null;    // 手动结算后点棋盘 = 继续对局
    hideOverlay();
    toast('已继续对局');
  }

  const node = rtAppendMove(tree, cur, x, y);
  if(!node){ toast('此点已有棋子'); return; }
  transitionTo(node);                           // append 后仍先在 parent/source 完成旧事务
  clearAIHint();                                 // 旧建议随新落子失效

  sfxPlace();
  if(node.move.pushes.length) sfxPush();
  if(navigator.vibrate){ try{ navigator.vibrate(10); }catch(e){} }

  if(G.phase === 'settled') settle('border', 480);
  else maybeAIMove();
}

/* 悔棋：只允许在叶子使用，删除当前叶子并回到父节点（子树不受影响）。
 * 人机模式若撤的是 AI 那手，连撤两手回到人类回合。 */
function undo(){
  if(!tree || !G) return;
  if(!rtIsLeaf(tree.cursor)) return;         // 控件已禁用，这里兜底
  clearAIHint();
  finishAnim();                               // 删除前 G 仍然是旧 leaf/source state
  let rec = rtDeleteLeaf(tree);              // 内部已把 cursor 回移到 parent
  if(!rec) return;
  let removedEdges = 1;
  if(aiSeat >= 0 && rec.player === aiSeat){
    const rec2 = rtDeleteLeaf(tree);
    if(rec2){
      rec = rec2;                             // 两层悔棋时只用于保留音效/兼容信息
      removedEdges = 2;
    }
  }
  G = tree.cursor.state;                     // G 与游标同步（唯一 canonical）
  hideOverlay();
  sfxUndo();
  /* 双悔棋跨越两条 edge，诚实使用 jump，不伪装成单步 backward。 */
  startRender(removedEdges === 1
    ? { mode: 'backward', move: rec }
    : { mode: 'jump', move: null });
  maybeAIMove();
}

function settle(reason, delayMs){
  clearAIHint();
  G.phase = 'settled'; G.reason = reason;
  renderGameChrome();
  sfxSettle();
  if(delayMs) setTimeout(showOverlay, delayMs);   // 仅延迟弹窗，状态立即锁定
  else showOverlay();
}

/* ================= 渲染 ================= */
function buildBoardDOM(){
  document.documentElement.style.setProperty('--bw', G.w);
  document.documentElement.style.setProperty('--bh', G.h);
  boardEl.style.aspectRatio = G.w + ' / ' + G.h;
  if(boardEl.setAttribute)
    boardEl.setAttribute('aria-label', G.w + ' 乘 ' + G.h + ' 交点棋盘');
  cellsEl.style.gridTemplateColumns = 'repeat(' + G.w + ', 1fr)';
  cellsEl.innerHTML = '';
  if(fileLabelsEl){
    fileLabelsEl.style.gridTemplateColumns = 'repeat(' + G.w + ', 1fr)';
    fileLabelsEl.innerHTML = '';
    for(let x = 0; x < G.w; x++){
      const el = document.createElement('span');
      el.textContent = coordFile(x);
      fileLabelsEl.appendChild(el);
    }
  }
  if(rankLabelsEl){
    rankLabelsEl.style.gridTemplateRows = 'repeat(' + G.h + ', 1fr)';
    rankLabelsEl.innerHTML = '';
    for(let y = 0; y < G.h; y++){
      const el = document.createElement('span');
      el.textContent = coordRank(y);
      rankLabelsEl.appendChild(el);
    }
  }
  for(let y = 0; y < G.h; y++){
    for(let x = 0; x < G.w; x++){
      const c = document.createElement('div');
      let cls = 'cell';
      if(x === 0) cls += ' first-col';
      if(x === G.w - 1) cls += ' last-col';
      if(y === 0) cls += ' first-row';
      if(y === G.h - 1) cls += ' last-row';
      if(isRing(x, y)) cls += ' ring';
      c.className = cls;
      cellsEl.appendChild(c);
    }
  }
  piecesEl.innerHTML = '';
  pieceEls.clear();
}

function makePieceEl(p){
  const el = document.createElement('div');
  el.className = 'piece p' + p;
  el.innerHTML = '<svg viewBox="0 0 40 40"><text x="20" y="21" text-anchor="middle" '
    + 'dominant-baseline="middle" font-size="19" font-weight="700" '
    + 'fill="rgba(0,0,0,.38)">' + NUM[p] + '</text></svg>';
  return el;
}
function setPos(el, x, y){
  setSlotPosition(el, x, y, G.w, G.h);
}
function renderMarker(){
  const last = G.history[G.history.length - 1];      // marker 永远瞬移，无过渡
  if(!last){
    markerEl.style.display = 'none';
  }else{
    markerEl.style.display = 'block';
    setSlotBox(markerEl, last.x, last.y, G.w, G.h);
  }
  renderAIHintMarker();
}
function renderHUD(){
  const counts = nfBorderCounts(G);
  const settled = G.phase === 'settled';
  let html = '';
  for(let i = 0; i < G.n; i++){
    html += '<div class="chip p' + i + (!settled && i === G.cur ? ' cur' : '') + '">'
          + '<i class="dot"></i><span>' + NAMES[i] + (i === aiSeat ? ' <em class="aitag">AI</em>' : '') + '</span><b>' + counts[i] + '</b></div>';
  }
  scorebarEl.innerHTML = html;

  const bt = G.w * 2 + (G.h - 2) * 2;
  const occ = counts.reduce((a, b) => a + b, 0);
  fillBarEl.style.width = (occ / bt * 100) + '%';
  fillTextEl.textContent = '外围 ' + occ + '/' + bt;

  const depth = rtDepth(tree.cursor), total = rtMainLineLength(tree);
  const atLeaf = rtIsLeaf(tree.cursor);
  if(!atLeaf){
    statusTextEl.innerHTML = '<span style="color:var(--accent)">复盘 · 第 ' + depth + ' / ' + total + ' 手</span>'
      + (G.phase === 'settled' ? ' · 已结算' : '')
      + ' · 点走法栏任意一手直达，或在棋盘走出变招';
  }else if(!settled){
    statusTextEl.innerHTML = '第 <b>' + (depth + 1) + '</b> 手 · 轮到 <span class="tn p' + G.cur + '" style="color:var(--pc)">' + NAMES[G.cur] + (G.cur === aiSeat ? '(AI)' : '') + '</span>'
      + (aiThinking ? ' · <span style="color:var(--accent)">AI 思考中…</span>' : '');
  }else{
    statusTextEl.innerHTML = (G.phase === 'settled' && G.reason === 'border')
      ? '外围已占满 · 对局已结算'
      : '已手动结算 · 点棋盘可继续，或悔棋复盘';
  }
  if(document.body) document.body.classList.toggle('replay', !atLeaf);
  updateControlsUI();
  updateHintUI();
  renderAIProgress();
  updateAIValue();
}
/* 复盘控件可用性（动画锁定期整体禁用，防快速点击混动画） */
function updateControlsUI(){
  if(!tree) return;
  const locked = Date.now() < navLockedUntil;
  const depth = rtDepth(tree.cursor), total = rtMainLineLength(tree);
  const atLeaf = rtIsLeaf(tree.cursor);
  if(btnFirstEl) btnFirstEl.disabled = locked || depth <= 0;
  if(btnPrevEl)  btnPrevEl.disabled  = locked || depth <= 0;
  if(btnNextEl)  btnNextEl.disabled  = locked || !rtNext(tree.cursor);
  if(btnLastEl)  btnLastEl.disabled  = locked || atLeaf;
  if(replayPosEl) replayPosEl.textContent = total ? (depth + ' / ' + total) : '— / —';
  const s = $('#btnSettle'), c = $('#btnClear');
  if(s) s.disabled = locked || !atLeaf || depth === 0;
  if(c) c.disabled = locked || !atLeaf;
}

/* ---------- Lichess 式走法树 ----------
 * 主线内联铺开；同一分叉点的所有 children 都直接可见：
 * children[0] 继续主线，children[1..] 作为浅米色缩进变例块挂在偏离点之后。
 * 点击任意 token 直接跳到该节点（jump 视觉），不用循环按钮。 */
function mtTokenHTML(node){
  const mv = node.move, depth = rtDepth(node);
  const cur = node === tree.cursor;
  const onPath = rtOnPath(tree.cursor, node);
  const numPrefix = (mv.player === 0)
    ? '<b>' + (Math.floor(depth / G.n) + 1) + '.</b>' : '';
  return '<span class="mtok p' + mv.player + (cur ? ' cur' : '') + (onPath ? ' path' : '') + '"'
    + ' data-nid="' + node.id + '" title="第 ' + depth + ' 手 · ' + NAMES[mv.player] + ' '
    + coordName(mv.x, mv.y) + '">'
    + numPrefix + '<i class="dot"></i>' + coordName(mv.x, mv.y) + '</span>';
}
/* 从 start（含）开始的整条变例线：自身 token + 自身变例 + 主线延续 */
function mtChainHTML(start, guard){
  if(guard-- <= 0) return '';
  let out = mtTokenHTML(start);
  for(let i = 1; i < start.children.length; i++)
    out += '<span class="varblock">' + mtChainHTML(start.children[i], guard) + '</span>';
  out += mtLineHTML(start, guard);
  return out;
}
/* node 之后的主线延续（children[0] 链），途中插入各节点的变例块 */
function mtLineHTML(node, guard){
  let out = '';
  let n = node.children[0];
  while(n && guard-- > 0){
    out += mtTokenHTML(n);
    for(let i = 1; i < n.children.length; i++)
      out += '<span class="varblock">' + mtChainHTML(n.children[i], guard) + '</span>';
    n = n.children[0];
  }
  return out;
}
function renderMoveTree(){
  if(!moveTreeEl || !tree) return;
  const guard = 100000;
  let html = '';
  const main = tree.root.children[0];
  if(main){
    /* 首着主线先显示；root siblings 紧跟主线首 token 作为变例，不抢到棋谱最前面。 */
    html += mtTokenHTML(main);
    for(let i = 1; i < tree.root.children.length; i++)
      html += '<span class="varblock">' + mtChainHTML(tree.root.children[i], guard) + '</span>';
    for(let i = 1; i < main.children.length; i++)
      html += '<span class="varblock">' + mtChainHTML(main.children[i], guard) + '</span>';
    html += mtLineHTML(main, guard);
  }
  moveTreeEl.innerHTML = html || '<div class="mtree-empty">尚无着法 · 在棋盘落子开始</div>';
  if(html && moveTreeEl.querySelector){
    const cur = moveTreeEl.querySelector('.mtok.cur');
    if(cur && typeof cur.scrollIntoView === 'function')
      cur.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }
}
/* 当前节点的 contextual fork row：同一分叉点全部后继一目了然 */
function renderForkRow(){
  if(!forkRowEl || !tree) return;
  const kids = tree.cursor.children;
  if(kids.length < 2){
    forkRowEl.classList.remove('show');
    forkRowEl.innerHTML = '';
    return;
  }
  const selId = tree.cursor.selectedChildId != null ? tree.cursor.selectedChildId : kids[0].id;
  let html = '<span class="forklabel">分叉</span>';
  for(const c of kids){
    html += '<button type="button" class="forkbtn p' + c.move.player + (c.id === selId ? ' sel' : '') + '"'
      + ' data-nid="' + c.id + '"><i class="dot"></i>' + coordName(c.move.x, c.move.y) + '</button>';
  }
  forkRowEl.innerHTML = html;
  forkRowEl.classList.add('show');
}
function renderGameChrome(){
  renderMarker();
  renderHUD();
  renderMoveTree();
  renderForkRow();
}

function computeRanks(counts){
  const order = [...Array(counts.length).keys()].sort((a, b) => counts[a] - counts[b] || a - b);
  const rank = Array(counts.length);
  rank[order[0]] = 1;
  for(let i = 1; i < order.length; i++){
    rank[order[i]] = counts[order[i]] === counts[order[i - 1]] ? rank[order[i - 1]] : i + 1;
  }
  return { order, rank };
}
function showOverlay(){
  if(!G || G.phase !== 'settled') return;
  const counts = nfBorderCounts(G);
  const { order, rank } = computeRanks(counts);
  const winners = order.filter(i => rank[i] === 1);
  $('#ovTitle').textContent = G.reason === 'border' ? '外围占满 · 自动结算' : '手动结算';
  $('#ovSub').textContent = '🏆 ' + winners.map(i => NAMES[i]).join('、') + (winners.length > 1 ? ' 并列获胜' : ' 获胜');
  $('#rankList').innerHTML = order.map(i =>
    '<div class="rrow ' + (rank[i] === 1 ? 'win' : '') + '">'
    + '<span class="rk">' + rank[i] + '</span>'
    + '<i class="dot p' + i + '"></i>'
    + '<span class="nm">' + NAMES[i] + '</span>'
    + '<b>' + counts[i] + ' 枚</b></div>'
  ).join('');
  overlayEl.classList.remove('hidden');
}
function hideOverlay(){ overlayEl.classList.add('hidden'); }

/* ================= 导航 ================= */
function navFirst(){ const r = rtFirst(tree); if(r !== tree.cursor) gotoNode(r); }
function navPrev (){ const p = rtPrev(tree.cursor); if(p) gotoNode(p); }
function navNext (){ const n = rtNext(tree.cursor); if(n) gotoNode(n); }
function navLast (){ const l = rtLast(tree.cursor); if(l !== tree.cursor) gotoNode(l); }

/* ================= 交互绑定 ================= */
boardEl.addEventListener('click', e => {
  if(!G) return;
  const r = boardEl.getBoundingClientRect();
  const { x, y } = boardCoordFromClient(e.clientX, e.clientY, r, G.w, G.h);
  tapCell(x, y);
});

/* 走法栏 / fork 行：事件委托，点击 token 直达节点 */
function bindTreeNav(container){
  if(!container || !container.addEventListener) return;
  container.addEventListener('click', e => {
    const t = e.target;
    const tok = t && t.closest ? t.closest('[data-nid]') : null;
    if(!tok || !container.contains(tok)) return;
    const node = rtNode(tree, +tok.dataset.nid);
    if(node && node !== tree.cursor) gotoNode(node);
  });
}
bindTreeNav(moveTreeEl);
bindTreeNav(forkRowEl);

$('#btnFirst').onclick = navFirst;
$('#btnPrev').onclick  = navPrev;
$('#btnNext').onclick  = navNext;
$('#btnLast').onclick  = navLast;
$('#btnHint').onclick = requestAIHint;
$('#btnSettle').onclick = () => {
  if(!G || !rtIsLeaf(tree.cursor) || !G.history.length) return;
  if(G.phase === 'settled'){ showOverlay(); return; }
  settle('manual');
};
$('#btnClear').onclick = function(){
  if(!G.history.length){ newGame(G.w, G.h, G.n); toast('盘面已清空'); return; }
  armConfirm(this, () => { newGame(G.w, G.h, G.n); toast('盘面已清空'); });
};
$('#btnNew').onclick = function(){
  const run = () => { newGame(cfg.w, cfg.h, cfg.players); toast('新对局开始 · ' + cfg.w + '×' + cfg.h + ' · ' + cfg.players + ' 名玩家'); };
  if(G && G.history.length) armConfirm(this, run); else run();
};
$('#ovUndo').onclick = undo;
$('#ovAgain').onclick = () => { newGame(G.w, G.h, G.n); toast('新对局开始'); };
$('#ovClose').onclick = hideOverlay;

/* 步进器 */
function stepper(el, min, max, get, set){
  el.innerHTML = '<button type="button" aria-label="减少">−</button><b></b><button type="button" aria-label="增加">+</button>';
  const btns = el.querySelectorAll('button'), val = el.querySelector('b');
  const upd = () => { val.textContent = get(); };
  btns[0].onclick = () => { set(Math.max(min, get() - 1)); upd(); saveCfg(); };
  btns[1].onclick = () => { set(Math.min(max, get() + 1)); upd(); saveCfg(); };
  upd();
}
stepper($('#stPlayers'), 2, 6, () => cfg.players, v => cfg.players = v);
stepper($('#stW'), 8, 19, () => cfg.w, v => cfg.w = v);
stepper($('#stH'), 8, 19, () => cfg.h, v => cfg.h = v);
const simsInput = $('#simsInput');
simsInput.onchange = () => {
  cfg.aiSims = Math.min(1024, Math.max(8, Math.round(Number(simsInput.value) || 96)));
  simsInput.value = cfg.aiSims;
  saveCfg();
  toast('AI sims 已设为 ' + cfg.aiSims + '；下一次搜索生效');
};
const sndChk = $('#sndChk');
sndChk.checked = cfg.sound;
sndChk.onchange = () => { cfg.sound = sndChk.checked; saveCfg(); if(cfg.sound) beep(500, .06); };

/* 尺寸变化：只有实际高度变化时才终止事务，并把高度写到 root（兄弟节点可继承） */
let renderedBoardHeight = null;
new ResizeObserver(() => {
  const height = Number(boardEl.offsetHeight) || 0;
  if(height === renderedBoardHeight) return;
  renderedBoardHeight = height;
  finishAnim();
  if(document.documentElement && document.documentElement.style && document.documentElement.style.setProperty)
    document.documentElement.style.setProperty('--board-rendered-height', height + 'px');
}).observe(boardEl);

/* ================= 导入 / 导出 =================
   格式：nf1.宽.高.人数.状态(p进行/b外围满/m手动).手数序列(base36坐标).校验和
   （全小写：parseSave 会先对输入做 toLowerCase 再解析）
   推动结果由规则确定性决定，只需记录每手落点，导入时重放即可完整还原历史；
   只导出当前 cursor 路径，分支不影响存档契约 */
const SAVE_TAG = 'nf1';
function checksum(s){
  let h = 5381;
  for(let i = 0; i < s.length; i++) h = (h * 33 + s.charCodeAt(i)) >>> 0;
  return h.toString(36);
}
function serialize(st){
  const moves = st.history.map(m => m.x.toString(36) + m.y.toString(36)).join('');
  const phase = st.phase === 'settled'
    ? (st.reason === 'border' ? 'b' : 'm')
    : 'p';
  const body = [st.w, st.h, st.n, phase, moves].join('.');
  return SAVE_TAG + '.' + body + '.' + checksum(body);
}
function validateReplay(w, h, n, moves){
  const board = Array.from({ length: h }, () => Array(w).fill(null));
  let terminalAt = -1;
  for(let k = 0; k < moves.length; k++){
    if(terminalAt >= 0) return '第 ' + (k + 1) + ' 手起对局已结束，存在多余手数';
    const [x, y] = moves[k];
    if(board[y][x]) return '第 ' + (k + 1) + ' 手落点重复，数据无效';
    computePushes(board, w, h, x, y);
    board[y][x] = { id: 0, p: 0 };                          // 占位，推力只看占用
    if(nfBoardBorderFull(board, w, h)) terminalAt = k;
  }
  return null;
}
function parseSave(text){
  const s = String(text || '').replace(/\s+/g, '').toLowerCase();
  const parts = s.split('.');
  if(parts.length !== 7 || parts[0] !== SAVE_TAG)
    return { error: '代码格式不正确（应为 ' + SAVE_TAG + '.宽.高.人数.状态.手数.校验和）' };
  const [, ws, hs, ns, ph, mv, ck] = parts;
  const w = +ws, h = +hs, n = +ns;
  if(!Number.isInteger(w) || !Number.isInteger(h) || w < 8 || w > 19 || h < 8 || h > 19)
    return { error: '棋盘尺寸无效（应为 8–19）' };
  if(!Number.isInteger(n) || n < 2 || n > 6) return { error: '玩家数量无效（应为 2–6）' };
  if(!'pbm'.includes(ph)) return { error: '对局状态无效' };
  if(checksum(ws + '.' + hs + '.' + ns + '.' + ph + '.' + mv) !== ck)
    return { error: '校验和不匹配，代码可能不完整或被改动' };
  if(mv.length % 2) return { error: '手数数据不完整' };
  const moves = [];
  for(let i = 0; i < mv.length; i += 2){
    const x = parseInt(mv[i], 36), y = parseInt(mv[i + 1], 36);
    if(!(x >= 0 && x < w && y >= 0 && y < h))
      return { error: '第 ' + (i / 2 + 1) + ' 手坐标超出棋盘' };
    moves.push([x, y]);
  }
  const err = validateReplay(w, h, n, moves);
  if(err) return { error: err };
  return { w, h, n, phase: ph, moves };
}
function importGame(save){
  newGame(save.w, save.h, save.n);
  for(const [x, y] of save.moves){
    const nd = rtAppendMove(tree, tree.cursor, x, y);
    if(nd){ tree.cursor = nd; G = nd.state; }
  }
  if(save.phase === 'm' && G.phase !== 'settled'){
    G.phase = 'settled'; G.reason = 'manual';
  }
  /* 外围满（'b'）由重放过程中的终局判定自动置位；
     声明为 p 但最后一手恰好填满外围的存档，以重放结果为准。 */
  startRender({ mode: 'jump', move: null });
  maybeAIMove();
}

let ioMode = 'export';
function openIO(mode){
  ioMode = mode;
  const ta = $('#ioText');
  if(mode === 'export'){
    $('#ioTitle').textContent = '导出对局';
    $('#ioMeta').textContent = G.w + '×' + G.h + ' · ' + G.n + ' 名玩家 · 已下 '
      + G.history.length + ' 手 · '
      + ((G.phase === 'playing') ? '进行中'
        : (G.phase === 'settled' && G.reason === 'border') ? '已结算（外围占满）' : '已结算（手动）');
    ta.value = serialize(G);
    ta.readOnly = true;
    $('#ioCopy').classList.remove('hidden');
    $('#ioDo').classList.add('hidden');
    $('#ioHint').textContent = '复制此代码即可存档或分享；导入后可完整还原盘面与全部历史（支持继续悔棋）。';
    setTimeout(() => { ta.focus(); ta.select(); }, 50);
  }else{
    $('#ioTitle').textContent = '导入对局';
    $('#ioMeta').textContent = '粘贴导出代码，将替换当前对局';
    ta.value = '';
    ta.readOnly = false;
    $('#ioCopy').classList.add('hidden');
    $('#ioDo').classList.remove('hidden');
    $('#ioHint').textContent = '自动忽略空格与换行；导入时逐手校验并重放，可继续悔棋复盘。';
  }
  $('#ioOverlay').classList.remove('hidden');
}
$('#btnExport').onclick = () => openIO('export');
$('#btnImport').onclick = () => openIO('import');
$('#ioClose').onclick = () => $('#ioOverlay').classList.add('hidden');
$('#ioOverlay').addEventListener('click', e => {
  if(e.target === $('#ioOverlay')) $('#ioOverlay').classList.add('hidden');
});
document.addEventListener('keydown', e => {
  if(e.key === 'Escape' && !$('#ioOverlay').classList.contains('hidden'))
    $('#ioOverlay').classList.add('hidden');
});
const ioTextArea = $('#ioText');
ioTextArea.addEventListener('focus', () => { if(ioTextArea.readOnly) ioTextArea.select(); });
$('#ioCopy').onclick = () => {
  const done = () => toast('已复制到剪贴板');
  if(navigator.clipboard && navigator.clipboard.writeText){
    navigator.clipboard.writeText(ioTextArea.value).then(done, () => fallbackCopy(ioTextArea, done));
  }else fallbackCopy(ioTextArea, done);
};
function fallbackCopy(ta, done){
  ta.focus(); ta.select();
  try{ document.execCommand('copy') ? done() : toast('复制失败，请长按手动全选复制'); }
  catch(e){ toast('复制失败，请长按手动全选复制'); }
}
$('#ioDo').onclick = () => {
  const res = parseSave(ioTextArea.value);
  if(res.error){ toast(res.error); return; }
  importGame(res);
  $('#ioOverlay').classList.add('hidden');
  toast('导入成功 · 已还原 ' + res.moves.length + ' 手');
};

/* ================= 人机对战（NFBot） =================
 * AI 走与人类完全相同的规则路径：nfEncodeCanonical 编码 → 内置模型前向 →
 * 掩码后取最优 → rtAppendMove 落子。模型由 tools/inject_web_ai.py 注入本文件。
 * AI 只在叶子（最新一手）行动；用户回看时 AI 不动。 */
function isHumanMode(){
  /* modeSel 是页面启动阶段初始化的；未选 ai0/ai1 即视为双人模式，便于 DOM 桩测试。 */
  return aiSeat < 0 && modeSel.value !== 'ai0' && modeSel.value !== 'ai1';
}
function isAtLeaf(){ return !!tree && rtIsLeaf(tree.cursor); }

function renderAIHintMarker(){
  if(!hintMarkerEl || !G || !aiHint || !isHumanMode() || G.n !== 2 ||
     G.phase !== 'playing' || aiHint.player !== G.cur || G.board[aiHint.y][aiHint.x]){
    if(hintMarkerEl) hintMarkerEl.style.display = 'none';
    return;
  }
  hintMarkerEl.style.display = 'block';
  setSlotBox(hintMarkerEl, aiHint.x, aiHint.y, G.w, G.h);
}

function updateHintUI(){
  if(!hintBtnEl || !hintStatusEl) return;
  const human = !!G && isHumanMode();
  const available = human && G.n === 2 && G.phase === 'playing' && !!aiModel && isAtLeaf();
  hintBtnEl.disabled = !available || aiHintBusy;
  hintBtnEl.textContent = aiHintBusy ? 'AI 推理中…' : (aiHint ? '重新提示' : 'AI 指导');
  if(!human){
    hintStatusEl.textContent = '切换到双人模式后可查看当前回合建议';
  }else if(G.n !== 2){
    hintStatusEl.textContent = 'AI 指导仅支持双人局';
  }else if(!aiModel){
    hintStatusEl.textContent = '未内置模型';
  }else if(!isAtLeaf()){
    hintStatusEl.textContent = '正在复盘浏览 · 回到最新一手后可查看建议';
  }else if(G.phase !== 'playing'){
    hintStatusEl.textContent = '对局已结算';
  }else if(aiHintBusy){
    hintStatusEl.textContent = '正在分析 ' + NAMES[G.cur] + ' 的下一步…';
  }else if(aiHint){
    hintStatusEl.textContent = '建议 ' + NAMES[aiHint.player] + ' 下在 ' + coordName(aiHint.x, aiHint.y) + '，仅供参考';
  }else{
    hintStatusEl.textContent = '当前轮到 ' + NAMES[G.cur] + '，点击按钮查看建议';
  }
}

function formatDuration(ms){
  if(!Number.isFinite(ms) || ms < 1000) return '<1 秒';
  const sec = ms / 1000;
  return sec < 60 ? sec.toFixed(1) + ' 秒' : Math.floor(sec / 60) + ' 分 ' + Math.round(sec % 60) + ' 秒';
}

function renderAIProgress(){
  if(!aiProgressRowEl || !aiProgressTextEl || !aiProgressFillEl) return;
  if(!aiSearchProgress){
    aiProgressRowEl.classList.remove('show');
    aiProgressRowEl.setAttribute && aiProgressRowEl.setAttribute('aria-hidden', 'true');
    return;
  }
  const total = Math.max(1, aiSearchProgress.total || 1);
  const done = Math.min(total, Math.max(0, aiSearchProgress.done || 0));
  const elapsed = Math.max(0, aiSearchProgress.elapsedMs || 0);
  const speed = elapsed > 0 ? done / (elapsed / 1000) : 0;
  const eta = done > 0 ? (elapsed / done) * (total - done) : NaN;
  aiProgressRowEl.classList.add('show');
  aiProgressRowEl.setAttribute && aiProgressRowEl.setAttribute('aria-hidden', 'false');
  aiProgressFillEl.style.width = (done / total * 100) + '%';
  aiProgressTextEl.textContent = 'PUCT ' + done + '/' + total + ' · ' +
    (speed ? speed.toFixed(1) + ' sims/s' : '准备中') +
    (done < total ? ' · 预计还需 ' + formatDuration(eta) : ' · 已完成');
}

function renderAIValue(){
  if(!aiValueRowEl || !aiValueTextEl) return;
  const usable = !!G && G.n === 2 && G.phase === 'playing' && !!aiModel &&
    !!aiValue && Number.isFinite(aiValue.value) && aiValue.player === G.cur;
  if(!usable){
    aiValueRowEl.classList.remove('show');
    aiValueRowEl.setAttribute && aiValueRowEl.setAttribute('aria-hidden', 'true');
    return;
  }
  const value = aiValue.value;
  const bounded = Math.max(-1, Math.min(1, value));
  const sign = value >= 0 ? '+' : '';
  const bias = bounded > 0.2 ? '偏当前行动方' : bounded < -0.2 ? '偏对手' : '接近均势';
  const source = aiValue.source ? ' · ' + aiValue.source : '';
  aiValueRowEl.classList.add('show');
  aiValueRowEl.setAttribute && aiValueRowEl.setAttribute('aria-hidden', 'false');
  aiValueTextEl.textContent = 'AI value · ' + NAMES[G.cur] + ' 视角：' +
    sign + value.toFixed(3) + '（+1 胜 / −1 负 · ' + bias + '）' + source;
}

function setAIValue(value, player, source){
  value = Number(value);
  if(!Number.isFinite(value)){
    aiValue = null;
  }else{
    aiValue = { value, player: player == null ? (G ? G.cur : -1) : player,
                source: source || '网络' };
  }
  renderAIValue();
}

function updateAIValue(){
  if(!isAtLeaf() || !G || G.n !== 2 || G.phase !== 'playing' || !aiModel || !window.NFForward){
    aiValue = null;
    renderAIValue();
    return;
  }
  try{
    const obs = nfEncodeCanonical(G);
    const out = window.NFForward.forward(aiModel, obs.data, G.h, G.w);
    setAIValue(out.value, G.cur, '网络');
  }catch(e){
    aiValue = null;
    renderAIValue();
  }
}

function setAIProgress(progress){
  aiSearchProgress = progress;
  renderAIProgress();
}
function clearAIProgress(){
  aiSearchProgress = null;
  renderAIProgress();
}

function cancelAISearch(){
  if(aiSearchPending){
    aiSearchPending = null;
    aiSearchId++;
  }
  aiThinking = false;
  clearAIProgress();
}

function clearAIHint(){
  aiHintToken++;
  aiHint = null;
  aiHintBusy = false;
  cancelAISearch();
  if(hintMarkerEl) hintMarkerEl.style.display = 'none';
  updateHintUI();
}

function snapshotForSearch(st){
  return {
    w: st.w, h: st.h, n: st.n, cur: st.cur,
    board: st.board.map(row => row.map(pc => pc ? pc.p : -1)),
  };
}

function pickPolicyMove(st){
  const obs = nfEncodeCanonical(st);
  const out = window.NFForward.forward(aiModel, obs.data, st.h, st.w);
  const legal = nfLegalMask(st);
  let best = -1, bestV = -Infinity;
  for(let a = 0; a < legal.length; a++){
    if(!legal[a]) continue;
    if(out.logits[a] > bestV){ bestV = out.logits[a]; best = a; }
  }
  return best >= 0 ? { x: best % st.w, y: Math.floor(best / st.w), value: out.value } : null;
}

function handleAIWorkerMessage(event){
  const msg = event.data || {};
  if(msg.type === 'ready'){
    aiWorkerReady = true;
    updateHintUI();
    return;
  }
  const pending = aiSearchPending;
  if(msg.type === 'progress'){
    if(pending && pending.requestId === msg.requestId)
      setAIProgress({done: msg.done, total: msg.total, elapsedMs: msg.elapsedMs});
    return;
  }
  if(msg.type !== 'result' && msg.type !== 'error') return;
  if(!pending || pending.requestId !== msg.requestId) return;
  aiSearchPending = null;
  if(msg.type === 'error'){
    setAIProgress({done: 0, total: pending.total, elapsedMs: msg.elapsedMs || 0});
    pending.done(null, { error: msg.error });
  }else{
    setAIProgress({done: pending.total, total: pending.total, elapsedMs: msg.elapsedMs});
    pending.done({ x: msg.x, y: msg.y, visits: msg.visits, value: msg.value,
                   elapsedMs: msg.elapsedMs, mcts: true }, {});
  }
  setTimeout(clearAIProgress, 500);
}

function initAIWorker(){
  if(!window.NF_MCTS_WORKER_SOURCE || !window.NF_WEB_MODEL) return;
  if(aiWorker){
    aiWorkerReady = false;
    try{ aiWorker.postMessage({ type: 'init', model: window.NF_WEB_MODEL }); }
    catch(e){ console.error('NF MCTS Worker model init:', e); }
    return;
  }
  const WorkerCtor = window.Worker || (typeof Worker !== 'undefined' ? Worker : null);
  const BlobCtor = window.Blob || (typeof Blob !== 'undefined' ? Blob : null);
  const URLCtor = window.URL || (typeof URL !== 'undefined' ? URL : null);
  if(!WorkerCtor || !BlobCtor || !URLCtor) return;
  try{
    const blob = new BlobCtor([window.NF_MCTS_WORKER_SOURCE], { type: 'application/javascript' });
    const url = URLCtor.createObjectURL(blob);
    const worker = new WorkerCtor(url);
    aiWorker = worker;
    worker.onmessage = handleAIWorkerMessage;
    worker.onerror = e => {
      console.error('NF MCTS Worker:', e);
      aiWorkerReady = false;
      if(aiSearchPending){
        const pending = aiSearchPending;
        aiSearchPending = null;
        clearAIProgress();
        pending.done(null, { error: 'worker-error' });
      }
    };
    worker.postMessage({ type: 'init', model: window.NF_WEB_MODEL });
  }catch(e){
    console.error('无法启动 NF MCTS Worker:', e);
    aiWorker = null; aiWorkerReady = false;
  }
}

function requestAISearch(st, sims, done){
  if(aiSearchPending){
    const old = aiSearchPending;
    aiSearchPending = null;
    old.done(null, { cancelled: true });
  }
  const requestId = ++aiSearchId;
  const total = Math.max(1, sims | 0 || WEB_MCTS_SIMS);
  aiSearchPending = { requestId, done, total };
  setAIProgress({done: 0, total, elapsedMs: 0});
  if(aiWorker && aiWorkerReady){
    try{
      aiWorker.postMessage({ type: 'search', requestId, state: snapshotForSearch(st),
                             sims: total, cPuct: 1.5 });
      return;
    }catch(e){
      aiSearchPending = null;
      console.error('NF MCTS Worker postMessage:', e);
    }
  }
  /* 无 Worker 时只作为测试/兼容回退；正式浏览器会走上面的 PUCT。 */
  setTimeout(() => {
    let move = null, error = null;
    try{ move = pickPolicyMove(st); }catch(e){ error = String(e); }
    if(!aiSearchPending || aiSearchPending.requestId !== requestId) return;
    aiSearchPending = null;
    setAIProgress({done: total, total, elapsedMs: 30});
    done(move, error ? { error, fallback: true } : { fallback: true });
    setTimeout(clearAIProgress, 500);
  }, 30);
}

function requestAIHint(){
  if(!G || !isHumanMode() || G.n !== 2){
    toast('AI 指导只在双人模式的双人局中可用');
    return;
  }
  if(!isAtLeaf()){
    toast('正在复盘浏览 · 回到最新一手后再获取建议');
    return;
  }
  if(G.phase !== 'playing'){
    toast('对局已结算，无法预测下一步');
    return;
  }
  if(!aiModel){
    toast('未内置模型，无法提供 AI 指导');
    return;
  }
  const token = ++aiHintToken;
  const player = G.cur;
  const depth = rtDepth(tree.cursor);
  aiHint = null;
  aiHintBusy = true;
  renderAIHintMarker();
  updateHintUI();

  /* Worker 搜索完成后只显示建议，不自动落子。 */
  setTimeout(() => {
    if(token !== aiHintToken || !G || !tree || G.phase !== 'playing' || G.cur !== player ||
       rtDepth(tree.cursor) !== depth || !isHumanMode()){
      aiHintBusy = false; updateHintUI(); return;
    }
    requestAISearch(G, webMctsSims(G), (mv, meta) => {
      if(token !== aiHintToken || !G || !tree || G.phase !== 'playing' || G.cur !== player ||
         rtDepth(tree.cursor) !== depth || !isHumanMode()) return;
      if(mv){
        aiHint = { x: mv.x, y: mv.y, player, mcts: !!meta.mcts };
        if(Number.isFinite(Number(mv.value)))
          setAIValue(mv.value, player, meta.mcts ? 'PUCT root' : '网络');
        else updateAIValue();
      }else if(meta && meta.error) toast('AI 指导推理失败');
      aiHintBusy = false;
      renderAIHintMarker();
      updateHintUI();
    });
  }, 30);
}

function maybeAIMove(){
  if(!tree || !rtIsLeaf(tree.cursor)) return;    /* 回看时不驱动 AI */
  if(aiSeat < 0 || !G || G.phase !== 'playing' || G.cur !== aiSeat) return;
  if(!aiModel) return;
  if(G.n !== 2){ aiSeat = -1; $('#modeSel').value = 'hh'; return; }
  aiThinking = true;
  renderHUD();
  const player = G.cur;
  const depth = rtDepth(tree.cursor);
  setTimeout(() => {
    if(!G || !tree || G.phase !== 'playing' || G.cur !== player ||
       rtDepth(tree.cursor) !== depth || !rtIsLeaf(tree.cursor)){
      aiThinking = false; renderHUD(); return;
    }
    requestAISearch(G, webMctsSims(G), (mv, meta) => {
      if(!G || !tree || G.phase !== 'playing' || G.cur !== player ||
         rtDepth(tree.cursor) !== depth || !rtIsLeaf(tree.cursor)){
        aiThinking = false; renderHUD(); return;
      }
      aiThinking = false;
      if(mv) tapCell(mv.x, mv.y, true);
      else { if(meta && meta.error) toast('AI 推理失败'); renderHUD(); }
    });
  }, 300);   // 先渲染「AI 思考中」，再把 PUCT 放入 Worker
}

function webModelEntries(){
  const registry = window.NF_WEB_MODEL_REGISTRY;
  if(registry && Array.isArray(registry.models))
    return registry.models.filter(m => m && m.meta && m.arch && m.tensors);
  return window.NF_WEB_MODEL ? [window.NF_WEB_MODEL] : [];
}
function webModelLabel(model){
  const meta = model && model.meta || {};
  return meta.label || meta.version || meta.id || '未命名模型';
}
function webModelById(id){
  const entries = webModelEntries();
  for(const model of entries){
    if(model.meta && model.meta.id === id) return model;
  }
  return entries[0] || null;
}
function populateModelSelect(selectedId){
  if(!modelSelectEl) return;
  const entries = webModelEntries();
  modelSelectEl.innerHTML = '';
  for(const model of entries){
    const option = document.createElement('option');
    option.value = model.meta.id || model.meta.version || '';
    option.textContent = webModelLabel(model);
    modelSelectEl.appendChild(option);
  }
  const fallback = entries.length ? (entries[0].meta.id || entries[0].meta.version || '') : '';
  modelSelectEl.value = entries.some(m => (m.meta.id || m.meta.version || '') === selectedId)
    ? selectedId : fallback;
}
function renderModelCard(){
  if(!modelCardEl) return;
  const model = window.NF_WEB_MODEL;
  const meta = model && model.meta;
  modelCardEl.innerHTML = '';
  if(!meta){ modelCardEl.classList.remove('show'); return; }
  const card = meta.card || {};
  const title = document.createElement('strong');
  title.className = 'model-card-title';
  title.textContent = webModelLabel(model);
  modelCardEl.appendChild(title);
  if(card.summary){
    const summary = document.createElement('div');
    summary.className = 'model-card-summary';
    summary.textContent = card.summary;
    modelCardEl.appendChild(summary);
  }
  if(Array.isArray(card.details) && card.details.length){
    const list = document.createElement('ul');
    list.className = 'model-card-details';
    for(const detail of card.details){
      const item = document.createElement('li');
      item.textContent = String(detail);
      list.appendChild(item);
    }
    modelCardEl.appendChild(list);
  }
  if(card.caution){
    const caution = document.createElement('div');
    caution.className = 'model-card-caution';
    caution.textContent = '注意：' + card.caution;
    modelCardEl.appendChild(caution);
  }
  modelCardEl.classList.add('show');
}
function renderModelVersion(){
  const el = $('#modelVersionDetail');
  const meta = window.NF_WEB_MODEL && window.NF_WEB_MODEL.meta;
  if(el){
    el.textContent = meta ? webModelLabel(window.NF_WEB_MODEL) : '未内置';
    el.title = meta ? [meta.id, meta.source, meta.trained_at, meta.exported_at].filter(Boolean).join(' · ') : '';
  }
  renderModelCard();
}
function cancelAISearchForModel(){
  ++aiSearchId;
  if(aiSearchPending){
    const pending = aiSearchPending;
    aiSearchPending = null;
    pending.done(null, { cancelled: true });
  }
  aiHintToken++;
  aiHintBusy = false;
  aiHint = null;
  aiValue = null;
  clearAIProgress();
  renderAIHintMarker();
}
function selectWebModel(id, quiet){
  const model = webModelById(id);
  if(!model){
    aiModel = null;
    renderModelVersion();
    return false;
  }
  const meta = model.meta || {};
  const oldId = window.NF_WEB_MODEL && window.NF_WEB_MODEL.meta && window.NF_WEB_MODEL.meta.id;
  if(oldId !== meta.id) cancelAISearchForModel();
  try{
    window.NF_WEB_MODEL = model;
    aiModel = window.NFForward && typeof window.NFForward === 'object'
      ? window.NFForward.loadModel(model) : null;
  }catch(e){
    console.error('模型加载失败:', e);
    aiModel = null;
  }
  cfg.modelId = meta.id || meta.version || null;
  saveCfg();
  populateModelSelect(cfg.modelId);
  renderModelVersion();
  updateAIValue();
  if(aiWorker){
    aiWorkerReady = false;
    try{ aiWorker.postMessage({ type: 'init', model }); }
    catch(e){ console.error('Worker 模型切换失败:', e); }
  }else{
    initAIWorker();
  }
  if(!quiet && meta.label) toast('已切换模型：' + meta.label);
  return !!aiModel;
}
function loadAIModel(){
  const entries = webModelEntries();
  if(!entries.length){ renderModelVersion(); return; }
  const registry = window.NF_WEB_MODEL_REGISTRY || {};
  const current = window.NF_WEB_MODEL && window.NF_WEB_MODEL.meta;
  const defaultId = registry.default_id || (current && (current.id || current.version));
  populateModelSelect(cfg.modelId || defaultId);
  selectWebModel(modelSelectEl && modelSelectEl.value || cfg.modelId || defaultId, true);
}

const modeSel = $('#modeSel');
if(modelSelectEl){
  modelSelectEl.addEventListener('change', () => {
    selectWebModel(modelSelectEl.value, false);
  });
}
function applyAIMode(){
  clearAIHint();
  const v = modeSel.value;
  const seat = v === 'ai0' ? 0 : v === 'ai1' ? 1 : -1;
  if(seat >= 0 && !aiModel){ toast('未内置模型，人机模式不可用'); modeSel.value = 'hh'; return; }
  if(seat >= 0 && G && G.n !== 2){ toast('人机模式仅支持双人局（玩家数量设为 2）'); modeSel.value = 'hh'; return; }
  aiSeat = seat;
  cfg.aiSeat = seat;
  saveCfg();
  renderHUD();
  maybeAIMove();
}
modeSel.addEventListener('change', applyAIMode);

/* ================= 启动 ================= */
loadCfg();
loadAIModel();
if(simsInput) simsInput.value = cfg.aiSims;
/* 恢复上次的对战模式（模型缺失时回退双人） */
if(cfg.aiSeat === 0 || cfg.aiSeat === 1){
  if(aiModel){
    aiSeat = cfg.aiSeat;
    modeSel.value = aiSeat === 0 ? 'ai0' : 'ai1';
  }else{
    cfg.aiSeat = -1;
  }
}
newGame(cfg.w, cfg.h, cfg.players);
