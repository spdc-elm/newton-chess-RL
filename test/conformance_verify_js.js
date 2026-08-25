'use strict';
/* 一致性验证（Python→JS 方向）：
 * 读取 rl/fixtures/py_games.json，用真实 JS 代码逐手重放，
 * 比对每步盘面/行动方、终局计分与 nf1 存档串。 */
const fs = require('fs');
const path = require('path');
const { loadGame } = require('./helpers/load-game');

function main(){
  const api = loadGame();
  const data = JSON.parse(fs.readFileSync(
    path.join(__dirname, '..', 'rl', 'fixtures', 'py_games.json'), 'utf8'));
  if(data.generator !== 'py') throw new Error('夹具生成器标记不对');
  const fails = [];

  for(let gi = 0; gi < data.games.length; gi++){
    const g = data.games[gi];
    api.newGame(g.w, g.h, g.n);
    let broke = false;
    for(let k = 0; k < g.moves.length; k++){
      const [x, y] = g.moves[k];
      const beforeLen = api.getG().history.length;
      api.tapCell(x, y);
      const s = api.getG();
      if(s.history.length !== beforeLen + 1){
        fails.push(`PY game${gi} 第 ${k + 1} 手被 JS 拒绝（应为合法）`);
        broke = true; break;
      }
      const flat = [];
      for(let yy = 0; yy < g.h; yy++) for(let xx = 0; xx < g.w; xx++){
        const pc = s.board[yy][xx];
        flat.push(pc ? pc.p : -1);
      }
      if(JSON.stringify(flat) !== JSON.stringify(g.steps[k].board) || s.cur !== g.steps[k].cur){
        fails.push(`PY game${gi} 第 ${k + 1} 手后盘面/行动方不一致`);
        broke = true; break;
      }
    }
    if(broke) continue;
    const s = api.getG();
    const counts = Array(g.n).fill(0);
    for(let y = 0; y < g.h; y++) for(let x = 0; x < g.w; x++){
      if(x === 0 || y === 0 || x === g.w - 1 || y === g.h - 1){
        const pc = s.board[y][x];
        if(pc) counts[pc.p]++;
      }
    }
    if(JSON.stringify(counts) !== JSON.stringify(g.border_counts))
      fails.push(`PY game${gi} 边界计分不一致 ${counts} != ${g.border_counts}`);
    if(s.cur !== g.cur_final) fails.push(`PY game${gi} 终局行动方不一致`);
    const actual = s.phase === 'settled' ? s.reason : 'not-terminal';
    if(actual !== g.terminal_reason) fails.push(`PY game${gi} 终局原因不一致 ${actual} != ${g.terminal_reason}`);
    if(api.serialize(s) !== g.code)
      fails.push(`PY game${gi} nf1 存档串不一致\n  js: ${api.serialize(s)}\n  py: ${g.code}`);
  }

  if(fails.length){
    console.error('✗ JS↔PY 一致性失败 %d 项:', fails.length);
    for(const f of fails) console.error(' -', f);
    process.exit(1);
  }
  console.log('✓ PY→JS 方向: %d 局已重放比对，全部一致', data.games.length);
}

main();
