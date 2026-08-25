/* 牛顿棋 · 网页端神经网络前向（纯 JS，无依赖）。
 * 与 rl/training/model.py 的 NFNet 逐层对应：
 *   stem(conv3x3+ReLU) → blocks×(conv3x3+ReLU → conv3x3 → +skip → ReLU)
 *   → policy 头 (1x1→ReLU→1x1) 展平为每格 logit
 *   → value 头 (1x1→ReLU→全局平均池化→Linear→tanh)
 * 张量布局均为 NCHW 平铺（Float32Array），卷积为 cross-correlation（与 PyTorch 一致）。
 * 同时支持 Node（测试）与浏览器（HTML 内联）。 */
(function(root){
'use strict';

function decodeB64Float32(b64){
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for(let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function loadModel(json){
  const tensors = {};
  for(const name in json.tensors){
    const t = json.tensors[name];
    const arr = decodeB64Float32(t.b64);
    if(arr.length !== t.shape.reduce((a, b) => a * b, 1))
      throw new Error('张量长度不符: ' + name);
    tensors[name] = { shape: t.shape, data: arr };
  }
  return { arch: json.arch, meta: json.meta || {}, tensors };
}

/* 3x3 卷积，pad=1 stride=1。x: [Cin*H*W]，W/B: torch 权重扁平化，out: [Cout*H*W] */
function conv3x3(x, W, B, Cin, Cout, H, Wd, out){
  out = out || new Float32Array(Cout * H * Wd);
  for(let o = 0; o < Cout; o++){
    const obase = o * H * Wd;
    for(let y = 0; y < H; y++){
      for(let xi = 0; xi < Wd; xi++){
        let acc = B[o];
        for(let c = 0; c < Cin; c++){
          const wbase = (o * Cin + c) * 9;
          const xbase = c * H * Wd;
          for(let ky = 0; ky < 3; ky++){
            const sy = y + ky - 1;
            if(sy < 0 || sy >= H) continue;
            const xrow = xbase + sy * Wd;
            const wrow = wbase + ky * 3;
            for(let kx = 0; kx < 3; kx++){
              const sx = xi + kx - 1;
              if(sx < 0 || sx >= Wd) continue;
              acc += x[xrow + sx] * W[wrow + kx];
            }
          }
        }
        out[obase + y * Wd + xi] = acc;
      }
    }
  }
  return out;
}

/* 1x1 卷积。x: [Cin*HW]，W: [Cout*Cin]，B: [Cout] */
function conv1x1(x, W, B, Cin, Cout, HW, out){
  out = out || new Float32Array(Cout * HW);
  for(let o = 0; o < Cout; o++){
    const wbase = o * Cin, b = B[o], obase = o * HW;
    for(let p = 0; p < HW; p++){
      let acc = b;
      for(let c = 0; c < Cin; c++) acc += W[wbase + c] * x[c * HW + p];
      out[obase + p] = acc;
    }
  }
  return out;
}

function reluInPlace(a){
  for(let i = 0; i < a.length; i++) if(a[i] < 0) a[i] = 0;
  return a;
}

function addInPlace(a, b){
  for(let i = 0; i < a.length; i++) a[i] += b[i];
  return a;
}

function workspaceFor(model, H, W){
  const key = H + 'x' + W;
  if(!model._workspace) model._workspace = {};
  if(!model._workspace[key]){
    const hw = H * W, CH = model.arch.channels;
    model._workspace[key] = {
      h: new Float32Array(CH * hw),
      tmp: new Float32Array(CH * hw),
      pre: new Float32Array(CH * hw),
      p32: new Float32Array(32 * hw),
      logits: new Float32Array(hw),
      v32: new Float32Array(32 * hw),
      pooled: new Float32Array(32),
    };
  }
  return model._workspace[key];
}

function forward(model, obs, H, W){
  const T = model.tensors, CH = model.arch.channels, planes = model.arch.planes_in;
  const hw = H * W;
  if(obs.length !== planes * hw) throw new Error('输入尺寸不符');
  const ws = workspaceFor(model, H, W);
  let h = conv3x3(obs, T['stem.0.weight'].data, T['stem.0.bias'].data, planes, CH, H, W, ws.h);
  reluInPlace(h);
  const tmp = ws.tmp;
  for(let b = 0; b < model.arch.blocks; b++){
    ws.pre.set(h);
    const pre = ws.pre;
    conv3x3(h, T[`trunk.${b}.c1.weight`].data, T[`trunk.${b}.c1.bias`].data, CH, CH, H, W, tmp);
    reluInPlace(tmp);
    conv3x3(tmp, T[`trunk.${b}.c2.weight`].data, T[`trunk.${b}.c2.bias`].data, CH, CH, H, W, h);
    addInPlace(h, pre);
    reluInPlace(h);
  }
  // policy 头
  const p32 = reluInPlace(conv1x1(h, T['policy.0.weight'].data, T['policy.0.bias'].data, CH, 32, hw, ws.p32));
  const logits = conv1x1(p32, T['policy.2.weight'].data, T['policy.2.bias'].data, 32, 1, hw, ws.logits);
  // value 头：全局平均池化 → linear → tanh
  const v32 = reluInPlace(conv1x1(h, T['value.0.weight'].data, T['value.0.bias'].data, CH, 32, hw, ws.v32));
  const pooled = ws.pooled;
  for(let c = 0; c < 32; c++){
    let sum = 0;
    const base = c * hw;
    for(let p = 0; p < hw; p++) sum += v32[base + p];
    pooled[c] = sum / hw;
  }
  const vw = T['value.4.weight'].data, vb = T['value.4.bias'].data;
  let value = vb[0];
  for(let c = 0; c < 32; c++) value += vw[c] * pooled[c];
  value = Math.tanh(value);

  return { logits, value };
}

const NFForward = { loadModel, forward, decodeB64Float32 };
if(typeof module !== 'undefined' && module.exports) module.exports = NFForward;
else root.NFForward = NFForward;

})(typeof self !== 'undefined' ? self : this);
