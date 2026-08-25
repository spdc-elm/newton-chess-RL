# Newton Chess

Eight-way pushes, Newton-cradle chains, fewest border stones wins.

Play: https://spdc-elm.github.io/newton-chess-RL/

Or open `newton-force.html` locally. No install.

## Build

```sh
python3 tools/build_html.py
npm test
```

Edit sources, not `newton-force.html`.

## Model version

The shipped net is labeled in `rl/web/nf_model.json` → `meta.version`.  
Current: **2026.08.25-iter210**.  
It also appears under Settings → 内置模型, and as `<!-- NF_MODEL ... -->` in the built HTML.

To ship a new net:

```sh
python3 rl/training/export_web_model.py --ckpt <run>/latest.pt
python3 tools/build_html.py
```

See `docs/rl-interface.md`. Longer Chinese notes: `docs/dev-notes.md`.
