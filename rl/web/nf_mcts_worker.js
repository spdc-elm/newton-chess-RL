/* 牛顿棋网页端 PUCT Worker。
 * 构建脚本会把 nf_forward.js + 本文件拼成一个 Blob Worker 字符串，
 * 因此运行时不请求任何外部文件。规则实现与 HTML/Python 纯逻辑层保持同语义，
 * 只在 Worker 内做搜索模拟；真正落子仍由主线程 nfApplyMove 执行。
 *
 * 消息：
 *   {type:'init', model: window.NF_WEB_MODEL}
 *   {type:'search', requestId, state:{w,h,n,cur,board}, sims, cPuct}
 * 返回：
 *   {type:'ready'}
 *   {type:'result', requestId, x, y, visits, value, elapsedMs}
 */
(function(){
'use strict';

let MODEL = null;
const DIRS = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]];

function makeState(src){
  return {
    w: src.w, h: src.h, n: src.n, cur: src.cur,
    board: src.board.map(row => row.slice()),
    phase: 'playing', reason: null,
  };
}

function isRing(x, y, w, h){
  return x === 0 || y === 0 || x === w - 1 || y === h - 1;
}

function computePushes(board, w, h, x, y){
  const pushes = [];
  for(const [dx, dy] of DIRS){
    let cx = x + dx, cy = y + dy;
    if(cx < 0 || cy < 0 || cx >= w || cy >= h) continue;
    if(board[cy][cx] < 0) continue;
    let hitEdge = false;
    for(;;){
      const nx = cx + dx, ny = cy + dy;
      if(nx < 0 || ny < 0 || nx >= w || ny >= h){ hitEdge = true; break; }
      if(board[ny][nx] < 0) break;
      cx = nx; cy = ny;
    }
    if(hitEdge) continue;
    const pc = board[cy][cx];
    board[cy][cx] = -1;
    board[cy + dy][cx + dx] = pc;
    pushes.push({fx:cx, fy:cy, tx:cx + dx, ty:cy + dy, p:pc});
  }
  return pushes;
}

function borderFull(board, w, h){
  for(let x = 0; x < w; x++){
    if(board[0][x] < 0 || board[h - 1][x] < 0) return false;
  }
  for(let y = 1; y < h - 1; y++){
    if(board[y][0] < 0 || board[y][w - 1] < 0) return false;
  }
  return true;
}

function borderCounts(s){
  const out = Array(s.n).fill(0);
  for(let y = 0; y < s.h; y++) for(let x = 0; x < s.w; x++){
    if(!isRing(x, y, s.w, s.h)) continue;
    const p = s.board[y][x];
    if(p >= 0) out[p]++;
  }
  return out;
}

function outcomeFor(s, player){
  const c = borderCounts(s);
  const own = c[player];
  let other = -Infinity;
  for(let i = 0; i < c.length; i++) if(i !== player) other = Math.max(other, c[i]);
  return own < other ? 1 : own > other ? -1 : 0;
}

function legalMask(s){
  const m = new Uint8Array(s.w * s.h);
  for(let y = 0; y < s.h; y++) for(let x = 0; x < s.w; x++)
    m[y * s.w + x] = s.board[y][x] < 0 ? 1 : 0;
  return m;
}

function encodeCanonical(s){
  const planes = s.n + 1, hw = s.w * s.h;
  const data = new Float32Array(planes * hw);
  const planeOf = s.n === 2 ? (p => p === s.cur ? 0 : 1) : (p => p);
  for(let y = 0; y < s.h; y++) for(let x = 0; x < s.w; x++){
    const p = s.board[y][x];
    if(p >= 0) data[planeOf(p) * hw + y * s.w + x] = 1;
  }
  for(let x = 0; x < s.w; x++){
    data[s.n * hw + x] = 1;
    data[s.n * hw + (s.h - 1) * s.w + x] = 1;
  }
  for(let y = 1; y < s.h - 1; y++){
    data[s.n * hw + y * s.w] = 1;
    data[s.n * hw + y * s.w + s.w - 1] = 1;
  }
  return data;
}

function applyMove(s, x, y){
  if(s.board[y][x] >= 0) return null;
  const mover = s.cur;
  const pushes = computePushes(s.board, s.w, s.h, x, y);
  s.board[y][x] = mover;
  const terminal = borderFull(s.board, s.w, s.h);
  if(terminal){ s.phase = 'settled'; s.reason = 'border'; }
  else s.cur = (s.cur + 1) % s.n;
  return {x, y, mover, pushes};
}

function undoMove(s, rec){
  s.board[rec.y][rec.x] = -1;
  for(const m of rec.pushes){
    s.board[m.ty][m.tx] = -1;
    s.board[m.fy][m.fx] = m.p;
  }
  s.cur = rec.mover;
  s.phase = 'playing'; s.reason = null;
}

function evaluate(s){
  const obs = encodeCanonical(s);
  const out = self.NFForward.forward(MODEL, obs, s.h, s.w);
  const mask = legalMask(s);
  const logits = out.logits;
  let max = -Infinity;
  for(let a = 0; a < mask.length; a++) if(mask[a] && logits[a] > max) max = logits[a];
  const priors = new Float64Array(mask.length);
  let sum = 0;
  for(let a = 0; a < mask.length; a++){
    if(!mask[a]) continue;
    const e = Math.exp(logits[a] - max);
    priors[a] = e; sum += e;
  }
  if(sum <= 0){
    let count = 0;
    for(let a = 0; a < mask.length; a++) count += mask[a];
    for(let a = 0; a < mask.length; a++) priors[a] = mask[a] ? 1 / count : 0;
  }else{
    for(let a = 0; a < priors.length; a++) priors[a] /= sum;
  }
  return {priors, value: out.value};
}

function makeEdges(priors){
  const edges = new Map();
  for(let a = 0; a < priors.length; a++) if(priors[a] > 0)
    edges.set(a, {P: priors[a], N: 0, W: 0, child: null});
  return edges;
}

function simulate(s, edges, cPuct){
  if(s.phase === 'settled') return outcomeFor(s, s.cur);
  let total = 1;
  for(const e of edges.values()) total += e.N;

  // PUCT: use sqrt(N_parent), not sqrt(log(N_parent)).  The latter
  // suppresses exploration substantially and changes the training game.
  const parentVisits = total;
  let bestA = -1, bestE = null, bestU = -Infinity;
  for(const [a, e] of edges){
    const q = e.N ? e.W / e.N : 0;
    const u = q + cPuct * e.P * Math.sqrt(parentVisits) / (1 + e.N);
    if(u > bestU){ bestU = u; bestA = a; bestE = e; }
  }
  if(bestA < 0) return 0;
  const rec = applyMove(s, bestA % s.w, Math.floor(bestA / s.w));
  if(!rec) return 0;
  let v;
  try{
    if(s.phase === 'settled'){
      v = outcomeFor(s, rec.mover);
    }else if(bestE.child === null){
      const leaf = evaluate(s);
      bestE.child = makeEdges(leaf.priors);
      v = -leaf.value;
    }else{
      v = -simulate(s, bestE.child, cPuct);
    }
  }finally{
    undoMove(s, rec);
  }
  bestE.N++;
  bestE.W += v;
  return v;
}

function finishSearch(s, edges, root){
  let bestA = -1, bestN = -1, bestP = -Infinity;
  const visits = new Float32Array(s.w * s.h);
  for(const [a, e] of edges){
    visits[a] = e.N;
    if(e.N > bestN || (e.N === bestN && e.P > bestP)){
      bestA = a; bestN = e.N; bestP = e.P;
    }
  }
  if(bestA < 0) return null;
  return {x: bestA % s.w, y: Math.floor(bestA / s.w), visits, value: root.value};
}

let activeRequestId = 0;
function beginSearch(src, sims, cPuct, requestId){
  const s = makeState(src);
  if(s.phase === 'settled') throw new Error('state already settled');
  sims = Math.max(1, sims | 0);
  const root = evaluate(s);
  const edges = makeEdges(root.priors);
  const chunk = Math.max(1, Math.min(8, Math.ceil(sims / 16)));
  const started = Date.now();
  let done = 0;
  activeRequestId = requestId;

  const step = () => {
    if(requestId !== activeRequestId) return; // 被新搜索请求取消
    try{
      const end = Math.min(sims, done + chunk);
      while(done < end){ simulate(s, edges, cPuct); done++; }
      const elapsedMs = Date.now() - started;
      self.postMessage({type:'progress', requestId, done, total:sims, elapsedMs});
      if(done < sims){
        setTimeout(step, 0); // 让 Worker 有机会发进度，也让消息队列保持可响应
        return;
      }
      const result = finishSearch(s, edges, root);
      if(!result) throw new Error('no legal move');
      self.postMessage({type:'result', requestId, x:result.x, y:result.y,
                        visits:result.visits, value:result.value,
                        elapsedMs}, [result.visits.buffer]);
    }catch(err){
      self.postMessage({type:'error', requestId, error:String(err && err.message || err)});
    }
  };
  step();
}

self.onmessage = event => {
  const msg = event.data || {};
  try{
    if(msg.type === 'init'){
      MODEL = self.NFForward.loadModel(msg.model);
      self.postMessage({type:'ready'});
      return;
    }
    if(msg.type === 'search'){
      if(!MODEL) throw new Error('model not initialized');
      beginSearch(msg.state, msg.sims || 16, msg.cPuct || 1.5, msg.requestId);
    }
  }catch(err){
    self.postMessage({type:'error', requestId:msg.requestId, error:String(err && err.message || err)});
  }
};

})();
