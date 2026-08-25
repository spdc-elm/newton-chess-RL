'use strict';
/* 一致性夹具（JS→Python 方向）：
 * 用真实游戏代码打随机对局，逐步导出盘面快照与 nf1 存档，
 * 供 Python 端逐手比对。种子固定，可复现。 */
const fs = require('fs');
const path = require('path');
const { loadGame } = require('./helpers/load-game');

function mulberry32(seed){
  let a = seed >>> 0;
  return function(){
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SIZES = [
  [8, 8, 2], [8, 8, 2], [9, 9, 2], [9, 9, 3], [10, 12, 2], [12, 12, 4], [19, 19, 5],
];
const GAMES_PER_SIZE = 2;

function playRandom(api, w, h, n, rng){
  api.newGame(w, h, n);
  const moves = [], steps = [];
  for(let guard = 0; guard < 1000; guard++){
    const G = api.getG();
    if(G.phase !== 'playing') break;
    const empties = [];
    for(let y = 0; y < h; y++) for(let x = 0; x < w; x++)
      if(!G.board[y][x]) empties.push([x, y]);
    if(!empties.length) break;
    const [x, y] = empties[Math.floor(rng() * empties.length)];
    api.tapCell(x, y);
    moves.push([x, y]);
    /* 落子后必须重新取 G：tapCell 会把 G 切换到新节点的状态 */
    const after = api.getG();
    const flat = [];
    for(let yy = 0; yy < h; yy++) for(let xx = 0; xx < w; xx++){
      const pc = after.board[yy][xx];
      flat.push(pc ? pc.p : -1);
    }
    steps.push({ board: flat, cur: after.cur });
  }
  const G = api.getG();
  return {
    w, h, n,
    moves,
    steps,
    terminal_reason: G.phase === 'settled' ? G.reason : 'not-terminal',
    border_counts: (() => {
      // 从盘面直接数外围
      const c = Array(n).fill(0);
      for(let y = 0; y < h; y++) for(let x = 0; x < w; x++){
        if(x === 0 || y === 0 || x === w - 1 || y === h - 1){
          const pc = G.board[y][x];
          if(pc) c[pc.p]++;
        }
      }
      return c;
    })(),
    cur_final: G.cur,
    code: api.serialize(G),
    encode_final: Array.from(api.nfEncodeCanonical(G).data),
  };
}

function main(){
  const api = loadGame();
  const games = [];
  let seed = 42;
  for(const [w, h, n] of SIZES){
    for(let i = 0; i < GAMES_PER_SIZE; i++){
      const rng = mulberry32(seed++);
      games.push(playRandom(api, w, h, n, rng));
    }
  }
  const out = { generator: 'js', version: 1, games };
  const dest = path.join(__dirname, '..', 'rl', 'fixtures', 'js_games.json');
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.writeFileSync(dest, JSON.stringify(out));
  const totalMoves = games.reduce((a, g) => a + g.moves.length, 0);
  console.log('✓ JS 夹具已生成:', dest);
  console.log('  对局数', games.length, '· 总手数', totalMoves,
    '· 终局原因分布', JSON.stringify(games.reduce((m, g) => (m[g.terminal_reason] = (m[g.terminal_reason] || 0) + 1, m), {})));
}

main();
