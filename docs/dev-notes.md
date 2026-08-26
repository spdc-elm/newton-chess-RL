# 牛顿棋 · 开发笔记

对外说明见仓库根目录英文 `README.md`。本文保留较细的中文工程说明。

# 牛顿棋 NEWTON CHESS

八向推力 · 牛顿摆传递 · 外围少者胜。离线单文件棋类游戏，支持 2–6 人与 8–19 自定义棋盘。

## 玩法

直接用浏览器打开 **`newton-force.html`** 即可，零依赖、可离线、移动端适配优先。

## 规则摘要

1. 玩家轮流把一枚己方棋子落到任意空格，落子后永久留在盘上。
2. **八向推力**：落子瞬间同时向周围八个方向发出推力。
3. **牛顿摆传递**：每个方向上，与落点相邻的连续棋子链中只有**链末端的一枚被推动一格**，
   中间的棋子保持不动（隔山打牛）。链末端之外是空格则移入；链末端已贴边则该方向不动——
   棋子永远不会被推出棋盘。
4. 刚落下的棋子是力的源头，自身不会被推动。
5. 棋盘最外围一圈被占满时自动结算，也可随时手动结算（结算后仍可悔棋复盘）。
6. **外围少者胜**：统计每位玩家在外围一圈上的棋子数，最少者获胜，可并列。

## 功能

- 双人至六人对战、宽高 8–19 自定义
- **人机对战**：可让内置 AI 执先手或后手（设置 → 对战模式；AI 走与人类完全相同的规则路径，
  支持任意 8–19 棋盘尺寸）。人机模式下悔棋会连撤两手回到你的回合；
  有趣的性质：本游戏在浅层战术水平上后手优势明显，AI 执后手时不可轻敌
- **双人模式 AI 指导**：在双人对战中点击「AI 指导」，自动按当前轮到的玩家给出下一落点建议；只高亮提示，不会替玩家落子
- 无限悔棋（结算后也可复盘）、清空盘面、手动/自动结算
- **复盘与变招分支（lichess 式走法树）**：右侧棋谱面板内联展示整棵走法树，
  主线顺排、变例缩进块展开，同一分叉点的全部后继一目了然；点击任意一手直达该局面（跨分支为
  jump 瞬切），分叉点下方另有 fork 按钮行。在任意历史节点落不同子即走出新变招，分支数不限。
  跳步控制固定在面板底部：⏮ ◀ ▶ ⏭。人机模式回看时只能切换分支，不能在历史节点改走新变招。
  存档只导出当前光标路径，分支不影响存档契约。悔棋只在最新一手可用，删除叶子并保留兄弟分支
- **布局**：主操作区为「AI 指导 / 结算 / 清空盘面」，AI 状态、PUCT 进度与 value 信息位于棋谱上方，复盘跳步、棋谱树与 fork 选择随后显示；手机端自上而下为标题/计分 → 棋盘 → 主操作 → AI 分析 → 复盘；
  桌面端（≥900px）棋盘居左、棋谱面板居右且高度跟随棋盘实际高度，设置与规则收在底部
- 落子与推子动画（动画事务化：几何动画期间锁定输入防止串线，跳转可随时打断并强制对齐）、
  实时边界计分板、WebAudio 合成音效（可关）
- **导入 / 导出存档**：导出一段可复制的代码，包含完整历史；
  导入时逐手校验并重放，可继续悔棋复盘

## 存档格式

```
nf1.<宽>.<高>.<人数>.<状态>.<手数序列>.<校验和>
     8-19 8-19 2-6  p/b/m   base36坐标对    djb2-base36
```

- 状态：`p` 进行中 / `b` 外围占满自动结算 / `m` 手动结算
- 手数序列：每手两个 base36 字符（x、y 各一位），按落子顺序排列
- 推动结果由规则确定性决定，因此只记录落点即可完整还原历史；
  导入时逐手重放校验（重复落点、越界等都会被拒绝）
- 解析前会去除空白并转小写，校验和不匹配则拒绝导入

## 工程结构

```
newton-force/
├── newton-force.template.html  静态 HTML/CSS 源模板（可编辑）
├── newton-force.html          编译产物（不要手改，构建脚本会覆盖）
├── package.json               npm test 入口（零依赖）
├── docs/
│   ├── rl-interface.md        机器接口 / RL 契约（现行规范）
│   └── archive/               历史档案：训练里程碑、旧版奖励方案、实验发现
├── tools/
│   ├── inject_web_ai.py       兼容入口：把模型/前向/Worker 注入 HTML
│   ├── build_html.py          单文件构建入口（模块 → newton-force.html）
│   └── monitor_stageA.sh      阶段 A 长训监控（每 5 分钟记录状态）
├── test/                      全部测试：sh test/run.sh 一键跑完
│   ├── run.sh                 语法检查 + JS 测试 + 双端一致性 + 跨端前向验证
│   ├── helpers/load-game.js   从 HTML 提取内联脚本、DOM 桩沙盒加载真实游戏 API
│   ├── rules.test.js          规则测试（25）：牛顿摆/边界/八向/悔棋/结算
│   ├── env.test.js            机器接口测试（24）：掩码/applyMove/undo/观测编码/确定性
│   ├── serialization.test.js  存档测试（17）：往返/重放一致/篡改检测/容错
│   ├── replay.test.js         复盘树/动画/chrome 场景测试（46）：分支、深层路径、转场事务与 UI 刷新
│   ├── conformance_*.js/.py   JS↔Python 规则双向一致性（14 局 × 逐步盘面/存档串/观测编码）
│   └── test_web_forward.js    纯 JS 神经网络前向 vs PyTorch 数值对齐（误差 ~1e-5）
└── rl/                        Python 训练侧（契约见 docs/rl-interface.md）
    ├── nf_env.py              规则核心（与 JS 端语义逐位一致）
    ├── evaluate.py            checkpoint 对战与分色统计
    ├── eval_policy_only.py    网页同款 AI 强度评估（纯策略 / 采样形态）
    ├── arena.py               同尺寸 checkpoint 分色循环赛 + Elo
    ├── bots/                  random / greedy（规则冒烟）/ mcts(纯UCT) / nn(PUCT+Dirichlet)
    ├── training/              model(NFNet全卷积) / selfplay / train / export_web_model / build_web_model_registry
    ├── web/                   nf_replay.js(复盘树纯逻辑) + nf_app.js(主应用/模型选择/model card) + nf_model.json(默认权重) + nf_models.json(registry) + nf_forward.js(JS前向) + nf_mcts_worker.js(PUCT Worker)
    ├── fixtures/              双端一致性夹具 + 前向验证向量
    └── runs/                  训练产物（metrics.jsonl + checkpoint + arena_report.md + loss_curves.png）
```

### AI 接入架构（单文件不破坏）

训练在 Python 侧完成后，`tools/build_html.py`（`inject_web_ai.py` 仍可作为兼容入口）把模型权重、
纯 JS 前向和 PUCT Worker bundle 拼接进 `newton-force.html` 的标记区域。游戏运行时：

`nfEncodeCanonical 编码 → 纯 JS 前向（全卷积网络，尺寸无关）→ 合法性掩码 → PUCT 搜索`

人机模式下，Worker 返回搜索后的动作，AI 自动执行；双人模式下，「AI 指导」使用同一个
Worker，只显示搜索建议，不自动改变状态。搜索使用设置中的 sims 参数；Worker 实时回传已完成 sims、当前速度和预计剩余时间。
因此同一个模型会自动适配先手和后手。PUCT 探索项统一为 `Q + c_puct · P · √N_parent / (1 + N_child)`；训练侧与网页 Worker 共用这一定义。无论是人机自动落子，还是玩家点击推荐点，最终都调用同一条
`nfApplyMove` 规则路径。重新训练后只需重跑导出+构建两个脚本即可更新棋力。

### 工程纪律：源代码和编译产物必须分开

`newton-force.html` 是给浏览器、手机和分享使用的**编译产物**，不是日常编辑的源码。
不要直接修改它，包括 `NF_AI`、`NF_APP` 注入区和其中的内联 JavaScript；下一次构建会覆盖手工修改，
也会让测试、源码和发布文件产生漂移。

正确的修改位置：

- 静态 HTML/CSS：编辑 `newton-force.template.html`；
- 复盘树纯逻辑：编辑 `rl/web/nf_replay.js`；
- 主线程游戏/UI/AI 适配：编辑 `rl/web/nf_app.js`；
- 网络前向：编辑 `rl/web/nf_forward.js`；
- 后台 PUCT、规则模拟和进度消息：编辑 `rl/web/nf_mcts_worker.js`；
- 模型参数：由 checkpoint 通过 `export_web_model.py` 或四模型 registry 通过 `build_web_model_registry.py` 生成，不手改 base64；
- 模型选择和 model card：编辑 `rl/web/nf_app.js` 与 `newton-force.template.html`，模型数据放在 `rl/web/nf_models.json`；
- 构建发布文件：运行 `python3 tools/build_html.py`；CI 在 main push 后自动构建、测试并发布；
- 构建后：运行 `sh test/run.sh`。

如果发布版行为不对，先回到对应源模块修复，再重新构建，不能在 HTML 产物上打补丁。

### 设计约束

- **本体必须保持单文件**：最终发布的 `newton-force.html` 所有 CSS/JS/参数内联，不引用外部运行时资源。
- **纯逻辑层**：HTML 内的 `nf*` 函数与 DOM 完全解耦，UI 与测试都走这一条路径；
  它同时也是未来 RL 环境（Python/C++）的语义基准，契约见 `docs/rl-interface.md`。
- 测试不复制算法副本，而是通过 `test/helpers/load-game.js` 提取编译后 HTML 内的
  `<script>` 并在 DOM 桩沙盒中执行，直接驱动真实发布代码；修改源模块或模板后先运行
  `python3 tools/build_html.py`，再运行测试，避免测试和发布产物分叉。

## 运行测试

```sh
sh test/run.sh        # 或：npm test
```

需要 Node.js（仅用于测试，游戏本身不需要）。
