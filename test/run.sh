#!/bin/sh
# 一键测试：语法检查（提取内联脚本）+ 全部 JS 测试 + 双端一致性 + 跨端前向验证
set -e
cd "$(dirname "$0")/.."

echo "== 语法检查（逐个内联脚本） =="
node -e '
const fs = require("fs");
const html = fs.readFileSync("newton-force.html", "utf8");
const scripts = html.match(/<script>[\s\S]*?<\/script>/g) || [];
if(!scripts.length){ console.error("未找到内联 <script>"); process.exit(1); }
scripts.forEach((s, i) => fs.writeFileSync(".inline" + i + ".js", s.replace(/^<script>/, "").replace(/<\/script>$/, "")));
console.log("发现 " + scripts.length + " 个内联脚本");
'
for f in .inline*.js; do node --check "$f"; done && echo "✓ 语法 OK"
rm -f .inline*.js

echo "== 规则测试 =="
node test/rules.test.js

echo "== 机器接口测试 =="
node test/env.test.js

echo "== 序列化测试 =="
node test/serialization.test.js

echo "== 复盘 / 分支测试 =="
node test/replay.test.js

echo "== Python RL 分差 / PUCT / D4 测试 =="
python3 rl/tests/test_bots.py
python3 rl/tests/test_training.py

echo "== JS↔Python 规则一致性（JS→PY→JS） =="
node test/conformance_export_js.js > /dev/null
python3 test/conformance_py.py
node test/conformance_verify_js.js

echo "== 跨端神经网络前向验证（JS vs PyTorch） =="
if [ -f rl/web/nf_model.json ]; then
  node test/test_web_forward.js
else
  echo "（未注入模型，跳过；运行 rl/training/export_web_model.py 后重试）"
fi

echo "== Worker PUCT 搜索协议测试 =="
node test/test_mcts_worker.js

echo "== AI 人机对战端到端集成测试 =="
node test/ai_integration.test.js

echo ""
echo "全部通过 ✓"
