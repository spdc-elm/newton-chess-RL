'use strict';
/*
 * 测试加载器：从 newton-force.html 提取内联 <script>，在带 DOM 桩的沙盒中执行，
 * 并导出真实的游戏内部 API。这样测试永远覆盖实际发布的代码，HTML 改动后无需同步测试副本。
 */
const fs = require('fs');
const path = require('path');

/* 有状态元素桩：classList/style/children 都保留状态，便于验证动画事务与 DOM chrome 刷新。 */
function makeEl(){
  const classSet = new Set();
  const styleProps = {};
  const children = [];
  const target = {
    style: {
      _props: styleProps,
      setProperty(name, value){ styleProps[name] = String(value); },
      getPropertyValue(name){ return styleProps[name] || ''; },
      removeProperty(name){ delete styleProps[name]; },
    },
    dataset: {},
    classList: {
      _set: classSet,
      add(...names){ names.forEach(name => String(name).split(/\\s+/).filter(Boolean).forEach(n => classSet.add(n))); },
      remove(...names){ names.forEach(name => String(name).split(/\\s+/).filter(Boolean).forEach(n => classSet.delete(n))); },
      contains(name){ return classSet.has(name); },
      toggle(name, force){
        if(force === true || (force !== false && !classSet.has(name))){ classSet.add(name); return true; }
        classSet.delete(name); return false;
      },
      toString(){ return [...classSet].join(' '); },
    },
    children,
    parentNode: null,
    appendChild(child){
      if(!child) return child;
      if(child.parentNode && child.parentNode.children){
        const old = child.parentNode.children.indexOf(child);
        if(old >= 0) child.parentNode.children.splice(old, 1);
      }
      child.parentNode = target;
      children.push(child);
      return child;
    },
    remove(){
      if(target.parentNode && target.parentNode.children){
        const i = target.parentNode.children.indexOf(proxy);
        if(i >= 0) target.parentNode.children.splice(i, 1);
      }
      target.parentNode = null;
    },
    contains(node){
      if(node === proxy) return true;
      return children.some(child => child === node || (child.contains && child.contains(node)));
    },
    focus(){}, select(){}, click(){},
    addEventListener(){}, removeEventListener(){},
    querySelector(){ return makeEl(); },
    querySelectorAll(){ return [makeEl(), makeEl(), makeEl()]; },
    getBoundingClientRect(){ return { left: 0, top: 0, width: 450, height: 450 }; },
    textContent: '', value: '',
    disabled: false, readOnly: false, checked: false,
  };
  let html = '';
  Object.defineProperty(target, 'innerHTML', {
    configurable: true,
    get(){ return html; },
    set(value){ html = String(value); children.length = 0; },
  });
  const proxy = new Proxy(target, {
    get(t, k){ return k in t ? t[k] : undefined; },
    set(t, k, v){ t[k] = v; return true; },
  });
  return proxy;
}

function loadGame(){
  const htmlPath = path.join(__dirname, '..', '..', 'newton-force.html');
  const html = fs.readFileSync(htmlPath, 'utf8');
  /* 主脚本是最后一个无 src 的 <script>（前面可能是注入的模型/前向代码） */
  const scripts = html.match(/<script>[\s\S]*?<\/script>/g) || [];
  if(!scripts.length) throw new Error('未能在 HTML 中找到内联 <script>');
  const m = scripts[scripts.length - 1].match(/<script>([\s\S]*?)<\/script>/);
  const factory = new Function(
    'document', 'window', 'navigator', 'localStorage', 'requestAnimationFrame', 'ResizeObserver',
    '"use strict";\n' + m[1] +
    '\nreturn {' +
    'computePushes, parseSave, serialize, checksum, validateReplay, importGame,' +
    /* 测试统一绕过几何动画锁（真实浏览器中锁由时钟自然过期） */
    'tapCell: (x,y,byAI) => { navLockedUntil = 0; tapCell(x,y,byAI); },' +
    'newGame, undo, settle, maybeAIMove, requestAIHint, clearAIHint,' +
    'getAIHint: () => aiHint,' +
    'getTree: () => tree, getCursor: () => tree.cursor, isAtLeaf: () => rtIsLeaf(tree.cursor),' +
    'nodeById: id => rtNode(tree, id),' +
    'gotoNodeById: id => { const n = rtNode(tree, id); if(n) gotoNode(n); return !!n; },' +
    'navFirst, navPrev, navNext, navLast,' +
    'depthOf: n => rtDepth(n), mainLineLength: () => rtMainLineLength(tree),' +
    'pathOf: n => rtPath(n), planOf: (a,b) => rtPlanTransition(a,b),' +
    'selectPath: n => rtSelectPath(tree,n),' +
    'getAnimationState: () => ({ fadingCount: fadingEls.size,' +
      'fading: [...fadingEls].map(el => [...el.classList._set]),' +
      'pieces: [...pieceEls].map(([id,el]) => ({ id, classes: [...el.classList._set] })),' +
      'locked: Date.now() < navLockedUntil }),' +
    'getMoveTreeHTML: () => moveTreeEl.innerHTML,' +
    'getForkHTML: () => forkRowEl.innerHTML,' +
    'getBoardRenderedHeight: () => document.documentElement.style._props["--board-rendered-height"],'+
    'coordName,' +
    'nfCreateState, nfCloneState, nfLegalMask, nfXYToAction, nfActionToXY,' +
    'nfApplyMove, nfUndo, nfIsBorderFull, nfBorderCounts, nfEncodeCanonical,' +
    'rtAppendMove: (parent,x,y) => rtAppendMove(tree,parent,x,y),' +
    'rtFindChild, rtPlanTransition, rtCreateTree,' +
    /* 测试专用：跳过几何动画锁（真实浏览器中锁由时钟自然过期） */
    'flushAnimLock: () => { navLockedUntil = 0; },' +
    'getG: () => G, getConfig: () => cfg };'
  );
  const elements = new Map();
  const documentStub = {
    querySelector(sel){
      if(typeof sel === 'string' && sel[0] === '#'){
        if(!elements.has(sel)) elements.set(sel, makeEl());
        return elements.get(sel);
      }
      return makeEl();
    },
    addEventListener(){},
    createElement(){ return makeEl(); },
    createDocumentFragment(){ return makeEl(); },
    documentElement: makeEl(),
    body: makeEl(),
  };
  return factory(
    documentStub,
    {},                                   // window（AudioContext 缺失 → 音效静默失败）
    {},                                   // navigator
    { getItem(){ return null; }, setItem(){}, removeItem(){} },
    cb => cb(),                           // requestAnimationFrame
    class { observe(){} }                 // ResizeObserver
  );
}

module.exports = { loadGame };
