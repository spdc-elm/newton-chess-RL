'use strict';
/* 端到端集成测试：加载注入后的真实 HTML（模型 + 前向 + 主逻辑三个内联脚本），
 * 在 DOM 桩沙盒里走一遍「选人机模式 → 人类落子 → AI 自动应答」。
 * 运行: node test/ai_integration.test.js （需已运行 tools/inject_web_ai.py） */
const fs = require('fs');
const path = require('path');

let pass = 0, fail = 0;
function T(name, cond){
  if(cond){ pass++; console.log('PASS', name); }
  else { fail++; console.log('FAIL', name); }
}

const htmlPath = path.join(__dirname, '..', 'newton-force.html');
const html = fs.readFileSync(htmlPath, 'utf8');
if(!html.includes('NF_AI:BEGIN')){
  console.log('（未注入模型，跳过 AI 集成测试）');
  process.exit(0);
}

/* 万能元素桩（带事件触发） */
function makeEl(){
  const listeners = {};
  const target = {
    style: { setProperty(){}, removeProperty(){} },
    dataset: {},
    classList: { add(){}, remove(){}, contains(){ return false; }, toggle(){} },
    appendChild(){}, remove(){}, focus(){}, select(){}, click(){},
    addEventListener(ev, fn){ (listeners[ev] = listeners[ev] || []).push(fn); },
    removeEventListener(){},
    _fire(ev){ (listeners[ev] || []).forEach(fn => fn()); },
    querySelector(){ return makeEl(); },
    querySelectorAll(){ return [makeEl(), makeEl(), makeEl()]; },
    getBoundingClientRect(){ return { left: 0, top: 0, width: 450, height: 450 }; },
    innerHTML: '', textContent: '', value: '',
    disabled: false, readOnly: false, checked: false,
  };
  return new Proxy(target, {
    get(t, k){ return k in t ? t[k] : undefined; },
    set(t, k, v){ t[k] = v; return true; },
  });
}

const sandboxWindow = {};
globalThis.self = sandboxWindow;          // 浏览器中 self === window，沙盒保持一致
const modeEl = makeEl();
const documentStub = {
  querySelector(sel){ return sel === '#modeSel' ? modeEl : makeEl(); },
  addEventListener(){},
  createElement(){ return makeEl(); },
  createDocumentFragment(){ return { appendChild(){} }; },
  documentElement: makeEl(),
};

/* 依次执行页面里每个内联脚本（同一全局环境，与浏览器一致） */
const scripts = html.match(/<script>[\s\S]*?<\/script>/g);
const factory = new Function(
  'document', 'window', 'navigator', 'localStorage', 'requestAnimationFrame',
  'ResizeObserver', 'performance',
  '"use strict";\n' +
  scripts.map(s => s.replace(/^<script>/, '').replace(/<\/script>\s*$/, '')).join('\n;\n') +
  '\nreturn { getG: () => G, getConfig: () => cfg,' +
  /* 测试统一绕过几何动画锁（真实浏览器中锁由时钟自然过期） */
  'tapCell: (x,y,byAI) => { navLockedUntil = 0; tapCell(x,y,byAI); },' +
  'undo, requestAIHint, getAIHint: () => aiHint, getAIValue: () => aiValue };'
);
const api = factory(
  documentStub, sandboxWindow, {},
  { getItem(){ return null; }, setItem(){}, removeItem(){} },
  cb => cb(),
  class { observe(){} },
  { now: () => Date.now() }
);

(async () => {
  T('页面启动无异常且已开局', !!api.getG() && api.getG().history.length === 0);
  T('window.NF_WEB_MODEL 已注入且含权重',
    typeof sandboxWindow.NF_WEB_MODEL === 'object' && Object.keys(sandboxWindow.NF_WEB_MODEL.tensors || {}).length > 10);
  T('当前局面已生成 value 评估', !!api.getAIValue() && Number.isFinite(api.getAIValue().value) && api.getAIValue().player === api.getG().cur);

  /* 模拟选择「AI 执后手」 */
  modeEl.value = 'ai1';
  modeEl._fire('change');
  await new Promise(r => setTimeout(r, 50));
  T('模式切换后 cfg.aiSeat = 1', api.getConfig().aiSeat === 1);

  /* AI 执后手时先手是人类：人类落子中心 (4,4) */
  api.tapCell(4, 4);
  T('人类落子成功（1 手）', api.getG().history.length === 1);

  /* 等 AI 应答（300ms 延迟 + JS 前向耗时） */
  const deadline = Date.now() + 15000;
  while(api.getG().history.length < 2 && Date.now() < deadline)
    await new Promise(r => setTimeout(r, 50));

  T('AI 在时限内自动应答（2 手）', api.getG().history.length >= 2);
  if(api.getG().history.length >= 2){
    T('AI 落的是蓝方（player=1）', api.getG().history[1].player === 1);
    T('AI 没有落在已占格', !api.getG().history.some((m, i) =>
      i > 0 && m.x === api.getG().history[0].x && m.y === api.getG().history[0].y));
  }

  /* 悔棋：人机模式下应连撤两手回到人类回合（历史清零，轮人类） */
  api.undo();
  T('悔棋回到人类回合（0 手）', api.getG().history.length === 0);
  api.tapCell(3, 3);
  T('人类再落子（1 手）', api.getG().history.length === 1);
  const dl2 = Date.now() + 15000;
  while(api.getG().history.length < 2 && Date.now() < dl2)
    await new Promise(r => setTimeout(r, 50));
  T('AI 再次应答（2 手）', api.getG().history.length >= 2);

  /* 切回双人模式：AI 不再动 */
  modeEl.value = 'hh';
  modeEl._fire('change');
  await new Promise(r => setTimeout(r, 30));
  T('切回双人模式 cfg.aiSeat = -1', api.getConfig().aiSeat === -1);

  /* 双人模式下 AI 只给建议，不自动落子 */
  const beforeHint = api.getG().history.length;
  api.requestAIHint();
  const hintDeadline = Date.now() + 15000;
  while(!api.getAIHint() && Date.now() < hintDeadline)
    await new Promise(r => setTimeout(r, 50));
  const hint = api.getAIHint();
  T('双人模式可得到当前回合 AI 建议', !!hint && hint.player === api.getG().cur);
  if(hint){
    T('AI 建议落在合法空格', !api.getG().board[hint.y][hint.x]);
    api.tapCell(hint.x, hint.y);
    T('点击建议点后由人类正常落子', api.getG().history.length === beforeHint + 1);
    T('落子后旧建议自动清除', api.getAIHint() === null);
  }

  console.log('\n%d passed, %d failed', pass, fail);
  process.exit(fail ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
