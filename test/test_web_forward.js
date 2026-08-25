'use strict';
/* 跨端前向数值验证：JS 实现的 NFNet 前向必须与 PyTorch 输出对齐（容差见向量文件）。
 * 运行: node test/test_web_forward.js  （需先有 rl/web/nf_model.json 与 web_vectors.json） */
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const model = JSON.parse(fs.readFileSync(path.join(ROOT, 'rl', 'web', 'nf_model.json'), 'utf8'));
const vecs = JSON.parse(fs.readFileSync(path.join(ROOT, 'rl', 'fixtures', 'web_vectors.json'), 'utf8'));
const NFForward = require(path.join(ROOT, 'rl', 'web', 'nf_forward.js'));

const m = NFForward.loadModel(model);
let maxErr = 0;
let pass = 0;

for(const c of vecs.cases){
  const obs = new Float32Array(c.obs);
  const { logits, value } = NFForward.forward(m, obs, c.h, c.w);
  let eLogit = 0, eVal = Math.abs(value - c.value);
  for(let i = 0; i < logits.length; i++)
    eLogit = Math.max(eLogit, Math.abs(logits[i] - c.logits[i]));
  const err = Math.max(eLogit, eVal);
  maxErr = Math.max(maxErr, err);
  const ok = err <= vecs.tolerance;
  if(ok) pass++;
  console.log('%s %dx%d n=%d · logit误差 %s · value误差 %s',
              ok ? 'PASS' : 'FAIL', c.w, c.h, c.n, eLogit.toExponential(2), eVal.toExponential(2));
}

console.log('\n最大误差 %s（容差 %s）· %d/%d 通过', maxErr.toExponential(2), vecs.tolerance.toExponential(0), pass, vecs.cases.length);
if(pass !== vecs.cases.length) process.exit(1);
console.log('跨端前向验证通过 ✓');
