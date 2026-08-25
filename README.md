# Newton Chess

Eight-way pushes, Newton-cradle chains, fewest border stones wins.

Play: https://spdc-elm.github.io/newton-chess-RL/

## Build

```sh
python3 tools/build_html.py
npm test
```

`newton-force.html` is generated. Do not commit it. CI builds and publishes it to Pages.

## Model version

Shown under Settings → 内置模型. Source of truth: `rl/web/nf_model.json` → `meta.version`.  
Current: **2026.08.25-iter210**.

```sh
python3 rl/training/export_web_model.py --ckpt <run>/latest.pt
python3 tools/build_html.py
```

See `docs/rl-interface.md`. Longer Chinese notes: `docs/dev-notes.md`.
