# 牛顿棋 · 机器接口与 RL 规范（v2）

> 机器友好接口契约。任何实现（JS 纯逻辑层、Python/C++ 环境）满足本文语义即可互换。
> 历史里程碑、旧版胜负奖励讨论、实验发现等已移至 [archive/training-history.md](archive/training-history.md)。

## 分层架构

```
┌─────────────────────────────────────────────┐
│ newton-force.html（单文件）                  │
│  ├─ 人类界面：棋盘渲染 / 动画 / 导入导出      │
│  └─ 纯逻辑层 nf* 函数 ←──── 本文档的契约     │
├─────────────────────────────────────────────┤
│ test/ 直接加载 HTML 内的真实逻辑做一致性验证  │
├─────────────────────────────────────────────┤
│ Python 训练侧 rl/（照本规范实现）             │
│ C++ 核心（规模化且 env 占比 >30% 时再上）     │
└─────────────────────────────────────────────┘
```

## 状态定义

```
st = {
  w, h,          棋盘尺寸，8–19
  n,             玩家数，2–6（RL 当前只用 n=2）
  cur,           当前行动玩家 0..n-1（终局后无意义）
  board,         h×w，格子 = 空 或 { p: 玩家索引 }   （id 字段仅动画用）
  phase,         'playing' | 'settled'
  reason,        'border'（外围满，真正的终局）| 'manual'（人为结算，RL 不使用）
}
```

## 核心函数

| 函数 | 语义 |
|---|---|
| `nfCreateState(w,h,n)` | 初始状态 |
| `nfLegalMask(st)` | `Uint8Array(w*h)`，1 = 可落子 |
| `nfApplyMove(st,x,y)` | 校验 → 牛顿摆推力 → 落子 → 轮转；外围占满则置终局。返回 `{ok,id,pushes,terminal}` |
| `nfUndo(st)` | 撤销一手（恢复 cur/phase），无手可撤返回 null |
| `nfIsBorderFull(st)` / `nfBorderCounts(st)` | 终局判定 / 各玩家边界子数 |
| `nfEncodeCanonical(st)` | 观测张量，见下 |
| `serialize(st)` / `parseSave(text)` | `nf1` 存档格式（跨实现互换 + 一致性测试载体） |

环境完全确定性，天然适合 MCTS。

## 动作空间

- `a = y*W + x`（行主序），动作空间恒为 `W*H`，不随空格数变化。
- 非法动作由掩码处理；对非法动作 step 必须报错而非静默忽略。

## 观测编码（`nfEncodeCanonical`）

- 形状 `(n+1, H, W)` 的 Float32 张量：
  - 平面 `0..n-1`：各玩家棋子占据面；
  - 二人局视角规范化：当前玩家恒在第 0 平面；
  - 平面 `n`：外围掩码（常数面，标出计分区）。
- 行主序展平，`(plane*H + y)*W + x`。
- 多人局不做规范化，RL 支持留待后续。

## 终局与价值目标（v2：分差回归）

- **终局**：仅当外围一圈被占满（`reason === 'border'`）。手动结算不属于环境语义。
- **value 目标（v2 主线）**：从该局面当前行动方视角，

  ```
  B_max = 2*(w+h) − 4            # 外围格总数，9×9 → 32
  m     = (opp_border − own_border) / B_max   ∈ [−1, +1]
  ```

  value head 直接回归期望分差 `E[m]`。胜、负、和分别对应 `m > 0 / < 0 / = 0`，
  即旧版 `z = sign(m)` 的信息被完整保留。
- **动机**：线下实际计分按分差进行，分差本身就是优化目标，并天然推广到多人
  （按最终名次差给分）；同时解决二元标签下「输 1 枚和输 8 枚无区别」的信用分配问题，
  缓解先手大量败局的「摆烂」。
- **policy 目标不变**：MCTS 根节点访问分布 `π`。
- **loss**：`L = CE(π, p) + λ·MSE(v, m)`。当前 pilot 基线 `λ=1`；λ 只影响训练时
  共享主干的梯度，不改变 MCTS 的分差目标。搜索叶子直接使用 `v`（归一化分差），
  二人局换边取反、终局用真实 `m`。
- **多人结算定义**：设外围最少值为 `c*`。每个非第一名玩家 `i` 向第一名支付
  `c_i-c*`；全部收入由并列第一名均分。原始效用总和为 0，统一除以
  `(n-1)*B_max` 映射到 `[-1,1]`。二人局严格退化为上式。
- **多人搜索边界**：三人以上不能逐层简单取负；未来 value 必须输出每位玩家的效用向量，
  MCTS 回传整向量，并由当前行动玩家最大化自己的分量。当前训练仍只覆盖 n=2。
- **数据坑**：自博弈存样本时 `m` 必须换算成**该局面当前行动方**视角；用单元测试锁死。
- 样本契约：`(obs, π, m)`。网络结构不变（仍是 policy logits + 单标量 value），
  只有 value 输出语义变化；JS 前向与导出格式无需改动。

## 训练配置决策（2026-08-24）

- **8× D4 对称增强**：每个基础样本在每个 epoch 中完整展开为 4 次旋转 × 镜像，
  `obs` 棋盘平面与 `π` 动作坐标同步变换，`m` 重复 8 份。Replay buffer 仍以未增强的
  基础样本计数，增强在 batch 构造时完成，不把内存扩大 8 倍。
- **可复用自博弈数据**：每轮未增强的 `(obs, π, m)` 默认保存为
  `rl/runs/<name>/selfplay/iterNNN.npz`；`obs` 使用 uint8，训练读取时转回 float32。
- **评测协议**：新 checkpoint 默认与同尺寸的固定旧 checkpoint 对战，不把 random/greedy
  作为棋力指标。对局先后手各半，使用温度采样，报告最终外围分差及分色结果；
  random/greedy 只用于规则冒烟或历史记录。
- **性能基线**：当前 profiling 显示自博弈约九成时间在 CPU batch=1 网络叶子推理，
  不是规则推进；`NNBot` 已对 CPU 推理启用 `inference_mode + channels_last`，不改网络结构、
  PUCT 公式或训练目标。更大的 GPU 批量叶子评估属于后续架构优化。
- **当前状态**：分差回传、value 迁移、D4×8、样本持久化、λ 参数化、CPU channels_last 与开局覆盖已经实现；
  开局覆盖版已完成 global iter60→210，先手 arena 有改善但仍明显弱，网页仍部署旧 value 语义模型。

## Web 模型版本

网页内置模型是 `rl/web/nf_model.json`。`meta` 必须带公开版本号，换模型时先改这里再构建 HTML。

| 字段 | 含义 |
|---|---|
| `version` | 对外版本，默认 `YYYY.MM.DD-iterN`，也可用 `--version` 指定 |
| `source` | 导出时 checkpoint 所在 run 目录名 |
| `iter` / `trained_size` / `params` | 训练迭代、主训练尺寸、参数量 |
| `value_target` / `augmentation` | 价值语义与数据增强 |
| `exported_at` | 导出时间 |

换模型流程：

```sh
python3 rl/training/export_web_model.py --ckpt <run>/latest.pt
python3 tools/build_html.py
sh test/run.sh
```

构建脚本会把 `meta` 写进 `newton-force.html` 的 `<!-- NF_MODEL ... -->` 注释；页面「对局设置 → 内置模型」读取同一个 `meta`。不要手改 HTML 里的版本字串。生成的 `newton-force.html` 不进 Git，由 CI 构建并发布到 Pages。

## 模型接入契约

1. **全卷积（FCN）**：头禁用固定尺寸全连接（全局池化除外），
   同一权重可在任意 8–19 棋盘推理。
2. **I/O 签名固定**：输入 `float32[(n+1), H, W]`，
   输出 `[policy(W*H 行主序 logits), value(标量)]`。
3. **传输格式**：权重 JSON + HTML 内置纯 JS 前向（保持单文件离线）；
   ONNX/onnxruntime-web 保留给未来更大模型。
4. **HTML 侧 NFBot**：加载模型 → canonical 编码 + 合法掩码 → 推理 → PUCT →
   `nfApplyMove`。AI 与人类走同一条规则路径。
5. **跨端数值验证**：导出时生成测试向量，JS/PyTorch 双端前向最大绝对误差 < 1e-3 方可上线。

## 跨实现一致性测试

1. JS 参考实现在各尺寸随机生成 N≥100 局，导出 `nf1` 存档；
2. 被测实现逐手重放，比对每步盘面哈希与边界计分；
3. 反向：被测实现对局的存档由 JS 重放校验。

## 已知边界

- `seq`/`id` 是动画专用，机器侧忽略；
- `manual` 结算只存在于人类界面；
- 多人局观测未规范化，RL 留待后续版本。

## 当前部署状态

- 内置模型版本：`2026.08.25-iter210`
- 来源：`marginv2_lambda4_opening15_d4_9x9_to210@iter210`（λ=4、D4×8、15 类首着覆盖，4×64、301,442 参数）。
- value 语义：当前行动方视角的归一化外围分差 `m`，不是旧版胜负倾向。
- 更新方式：见上文「Web 模型版本」。
