# Newton Chess

Eight-way pushes, Newton-cradle chains, fewest border stones wins.

Play: https://spdc-elm.github.io/newton-chess-RL/

## Build

```sh
python3 tools/build_html.py
npm test
```

`newton-force.html` is generated. Do not commit it. CI builds and publishes it to Pages.

## Built-in models

The default model is shown under Settings → 内置模型. The single-model compatibility file is
`rl/web/nf_model.json`; the selectable model registry is `rl/web/nf_models.json`.

Current default: **2026-08-26 · Gumbel Full**. The UI also includes dated cards for the
opening15 improvement, corrected- PUCT imbalance, and historical `sqrt(log N)` Stage A models.
All browser versions currently use the standard PUCT Worker; native browser Gumbel search is a later task.

```sh
python3 rl/training/build_web_model_registry.py
python3 tools/build_html.py
npm test
```

`newton-force.html` is generated and should not be committed. `.github/workflows/pages.yml` builds,
tests, and publishes it through GitHub Pages on `main`.
