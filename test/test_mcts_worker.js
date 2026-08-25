'use strict';
/* Worker bundle 单元测试：实际执行 nf_forward.js + nf_mcts_worker.js，
 * 不依赖浏览器 Worker API，用 vm 模拟 self.postMessage。 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.join(__dirname, '..');
const model = JSON.parse(fs.readFileSync(path.join(ROOT, 'rl', 'web', 'nf_model.json'), 'utf8'));
const forward = fs.readFileSync(path.join(ROOT, 'rl', 'web', 'nf_forward.js'), 'utf8');
const worker = fs.readFileSync(path.join(ROOT, 'rl', 'web', 'nf_mcts_worker.js'), 'utf8');
if(/Math\.sqrt\(logN\)/.test(worker) || /const logN\s*=/.test(worker))
  throw new Error('Worker PUCT 不应使用 sqrt(log N)');
if(!/Math\.sqrt\(parentVisits\)/.test(worker))
  throw new Error('Worker PUCT 未使用 sqrt(N_parent)');
const messages = [];
const selfObj = { postMessage(msg){ messages.push(msg); } };
const context = {
  self: selfObj,
  console,
  Math,
  Date,
  Float32Array,
  Uint8Array,
  Array,
  Map,
  Error,
  atob: global.atob,
  setTimeout,
};
vm.runInNewContext(forward + '\n' + worker, context, { filename: 'nf-mcts-worker.bundle.js' });

(async () => {
  selfObj.onmessage({data: {type: 'init', model}});
  if(!messages.some(m => m.type === 'ready')) throw new Error('Worker 未 ready');
  messages.length = 0;

  const w = 9, h = 9;
  const board = Array.from({length: h}, () => Array(w).fill(-1));
  selfObj.onmessage({data: {
    type: 'search', requestId: 17,
    state: {w, h, n: 2, cur: 0, board},
    sims: 8, cPuct: 1.5,
  }});

  const deadline = Date.now() + 10000;
  let result = null;
  while(Date.now() < deadline){
    result = messages.find(m => m.type === 'result');
    if(result) break;
    await new Promise(r => setTimeout(r, 10));
  }
  if(!result) throw new Error('Worker 未返回 result: ' + JSON.stringify(messages));
  if(!messages.some(m => m.type === 'progress')) throw new Error('Worker 未返回 progress');
  if(!(result.x >= 0 && result.x < w && result.y >= 0 && result.y < h))
    throw new Error('Worker 返回越界动作');
  if(!(result.visits instanceof Float32Array) || result.visits.length !== w * h)
    throw new Error('Worker visits 尺寸错误');
  const total = result.visits.reduce((a, b) => a + b, 0);
  if(total !== 8) throw new Error('Worker visits 总数应为 8，实际 ' + total);
  console.log('Worker PUCT/progress 通过 ✓ action=(%d,%d) visits=%d elapsed=%dms',
              result.x, result.y, total, result.elapsedMs);
})().catch(err => { console.error(err); process.exit(1); });
