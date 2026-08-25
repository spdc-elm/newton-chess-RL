'use strict';
/* ================= 复盘树核心（纯逻辑层，无 DOM / 无 AI / 无动画） =================
 * 本模块只管理：树、唯一游标、稳定节点 ID、稳定棋子 ID、分支、导航、悔棋与转场计划。
 * 与 nf* 纯规则层的契约：
 *   - 每个节点缓存完整 nf state 快照（node.state），创建子节点时 clone 父节点状态；
 *   - clone 的 seq 先置为 tree.nextPieceId - 1，再 nfApplyMove，使新落棋子拿到全树唯一
 *     且永不改变的 piece id；已有棋子永不重新编号。
 *   - 同一颗棋子在 live、replay、任何共享前缀分支中保持同一 id（消灭动画串线根源）。
 *
 * 节点结构：
 *   node = {
 *     id,                 // 全树唯一的稳定节点 ID
 *     parent,             // 父节点或 null（root）
 *     children,           // children[0] 为主线，其余为变例
 *     selectedChildId,    // 当前 fork 选择（仅影响 Next 走向）；null = 用 children[0]
 *     move,               // 该手的 move record { player,x,y,id,pushes:[{id,fx,fy,tx,ty}] }；root 为 null
 *     state,              // 该手之后的完整 nf state 快照
 *   }
 */

function rtCreateTree(w, h, n){
  const root = {
    id: 0, parent: null, children: [], selectedChildId: null,
    move: null, state: nfCreateState(w, h, n),
  };
  return { root, cursor: root, nextNodeId: 1, nextPieceId: 1 };
}

function rtNode(tree, id){
  if(id === tree.root.id) return tree.root;
  /* DFS 查找；树的规模（数百节点）下足够快，且只在点击/渲染时调用 */
  const stack = tree.root.children.slice();
  while(stack.length){
    const nd = stack.pop();
    if(nd.id === id) return nd;
    for(const c of nd.children) stack.push(c);
  }
  return null;
}

/* 查找 parent 下坐标为 (x,y) 的既有子节点（有则复用，不产生重复分支） */
function rtFindChild(node, x, y){
  for(const c of node.children)
    if(c.move.x === x && c.move.y === y) return c;
  return null;
}

/* 在 parent 下创建新手分支（不查重，调用方负责）。返回新子节点。 */
function rtAppendMove(tree, parent, x, y){
  const st = nfCloneState(parent.state);
  st.seq = tree.nextPieceId - 1;                     /* 让 ++seq 恰好等于 nextPieceId */
  st.phase = 'playing';                              /* 变例从可行动状态开始（手动结算只是 UI 概念） */
  st.reason = null;
  const res = nfApplyMove(st, x, y);
  if(!res.ok) return null;
  tree.nextPieceId = st.seq + 1;
  const rec = st.history[st.history.length - 1];      /* { player,x,y,id,pushes } */
  const node = {
    id: tree.nextNodeId++,
    parent, children: [], selectedChildId: null,
    move: { player: rec.player, x: rec.x, y: rec.y, id: rec.id,
            pushes: rec.pushes.map(p => ({ id: p.id, fx: p.fx, fy: p.fy, tx: p.tx, ty: p.ty })) },
    state: st,
  };
  parent.children.push(node);
  parent.selectedChildId = node.id;                  /* 新手即当前选择：Next 可原路返回 */
  return node;
}

/* ---------- 导航 ---------- */
/* 上一手：父节点（root 时返回 null） */
function rtPrev(node){ return node.parent; }
/* 下一手：优先 selectedChildId，否则 children[0]；无后继返回 null */
function rtNext(node){
  if(!node.children.length) return null;
  if(node.selectedChildId != null){
    const sel = node.children.find(c => c.id === node.selectedChildId);
    if(sel) return sel;
  }
  return node.children[0];
}
/* 起点 */
function rtFirst(tree){ return tree.root; }
/* 从 node 沿 selected/children[0] 走到叶子 */
function rtLast(node){
  let n = node;
  for(;;){ const nx = rtNext(n); if(!nx) return n; n = nx; }
}
/* 是否位于叶子（无后继）——悔棋/AI/结算语义都以它为准 */
function rtIsLeaf(node){ return node.children.length === 0; }
/* 节点深度（root=0，第 k 手的节点深度为 k）＝ 当前路径手数 */
function rtDepth(node){
  let d = 0, n = node;
  while(n.parent){ d++; n = n.parent; }
  return d;
}
/* root→node 的路径（含 root 与 node） */
function rtPath(node){
  const out = [];
  let n = node;
  while(n){ out.push(n); n = n.parent; }
  return out.reverse();
}
/* root→node 路径设为当前 selected line：点击深层 token 后，
 * 每一层 parent 都选择通往该 node 的 child，保证 rtMainLineLength 包含 cursor。 */
function rtSelectPath(tree, node){
  const path = rtPath(node);
  for(let i = 0; i + 1 < path.length; i++)
    path[i].selectedChildId = path[i + 1].id;
}
/* 主线长度：从 root 沿 selected 链走到叶子的手数（走法栏分母） */
function rtMainLineLength(tree){ return rtDepth(rtLast(tree.root)); }
/* node 是否在 cursor 的当前路径上（含相等） */
function rtOnPath(cursor, node){
  let n = cursor;
  while(n){ if(n === node) return true; n = n.parent; }
  return false;
}

/* ---------- 分支与悔棋 ---------- */
/* 设置 fork 选择（点击 fork 选项 / 点击走法栏 token 时由调用方配合 goto 使用） */
function rtSelect(parent, childId){
  if(parent.children.some(c => c.id === childId)) parent.selectedChildId = childId;
}
/* 删除当前叶子（悔棋）：仅当 cursor 是叶子时有效。
 * 返回被删节点的 move record；cursor 回到 parent。子树不存在（只删叶子）。 */
function rtDeleteLeaf(tree){
  const cur = tree.cursor;
  if(cur.children.length || !cur.parent) return null;
  const parent = cur.parent;
  const idx = parent.children.indexOf(cur);
  parent.children.splice(idx, 1);
  parent.selectedChildId = null;                     /* 回落到主线 children[0] */
  tree.cursor = parent;
  return cur.move;
}

/* ---------- 转场计划（纯函数，供 renderer 执行） ----------
 * mode 严格四种：
 *   forward  : to 是 from 的直系子 —— 正放 to.move（新子 drop、pushes fx,fy→tx,ty、推子 pulse）
 *   backward : from 是 to 的直系子 —— 倒放 from.move（move.id 反向消失、pushes tx,ty→fx,fy、恢复 pulse）
 *   jump     : 其余一切（跨分支、跳步、开局/末端、导入、重置）—— 快照对齐，不伪装成倒带
 *   none     : 同节点
 * forward/backward 严格互逆：都直接使用 edge 上的 move record，绝不对比两份重新生成的快照。 */
function rtPlanTransition(fromNode, toNode){
  if(fromNode === toNode) return { mode: 'none', move: null };
  if(toNode.parent === fromNode) return { mode: 'forward', move: toNode.move };
  if(fromNode.parent === toNode) return { mode: 'backward', move: fromNode.move };
  return { mode: 'jump', move: null };
}
