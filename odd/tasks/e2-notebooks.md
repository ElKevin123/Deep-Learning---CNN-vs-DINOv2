# E2 — Data notebook, teardown/backbone-swap notebook, decision log

## Objective
Deliver E2 (teardown + preliminary intervention + decision log) as Kaggle-ready notebooks
in `Colabs/` plus `DECISIONS.md` and an updated `README.md`. Notebooks are NOT executed
locally; the team runs them on Kaggle (T4 GPU, Internet ON) and sends results back.

## Problem / why
Overture "Transportation Networks" is vector-only (road segments, GeoParquet). There is no
imagery, so RGB patches must come from another source (Esri World Imagery) and Overture
segments must be rasterized as masks. Misregistration between vector and imagery is the main
data risk, so it must be measured, not assumed.

## Decisions taken with the user (2026-10-21)
- Dataset 1 (own): Esri World Imagery tiles (zoom 17, ~1.17 m/px at Lima) + Overture road
  segments rasterized -> 512x512 patches. Default AOI: Lima (configurable bbox).
- Dataset 2 (reference benchmark): Massachusetts Roads (Mnih 2013), Kaggle copy.
- Same architectures on both datasets: ResNet-50 vs DINOv2 (ViT-S/14) + common U-Net decoder,
  loss BCE + Dice.
- Alignment of Overture masks must be quantified with explicit metrics.

## Scope
1. `Colabs/01_data_exploration_alignment.ipynb`
   - Part A: Overture download (overturemaps CLI, bbox) -> vector EDA (subtype/class counts,
     length per class in km, map) -> filter drivable classes -> Esri tile download (cached,
     retry, polite concurrency) -> 2x2 tiles = 512 px patch -> rasterize buffered segments
     (width by class, meters -> EPSG:3857 units / cos(lat)) -> patch EDA (road-pixel ratio,
     empty patches, overlays) -> spatial-block split (no leakage) -> export PNG + CSV.
   - Part B: Alignment audit (Overture): overlay QA, edge-alignment shift search
     (Sobel on image vs shifted mask boundary, +-R px), road-likeness cross-correlation,
     per-patch offset (dx, dy) in px and m, peak confidence, IoU-vs-shift sensitivity curve,
     manual QA CSV template. Optional offset-corrected masks.
   - Part C: Massachusetts Roads EDA + same alignment audit as reference.
2. `Colabs/02_teardown_backbone_swap.ipynb`
   - Teardown: params (total/trainable/frozen, per block), FLOPs (torch.utils.flop_counter),
     latency (torch.cuda.Event, fp32/fp16, ms/patch, throughput), VRAM (idle, peak inference,
     peak train step), effective receptive field, ResNet hooks layer1-4, DINOv2 attention
     per head + PCA of patch tokens (+ AUC of PCA vs road mask).
   - Backbone swap: shared U-Net decoder, ResNet-50 adapter, DINOv2 simple-pyramid neck,
     BCE+Dice, AMP, conditions resnet50 / dinov2_frozen / dinov2_partial, datasets
     massachusetts / overture, metrics IoU road, mIoU (road+bg), F1, P/R, relaxed F1 (rho=3 px),
     JSON results export. Optional model-based label audit on Overture.
3. `DECISIONS.md` decision log (data, architecture, compute, AI usage, pre-registered
   hypotheses, licenses) + `README.md` with structure, how to run on Kaggle, results table
   placeholders.

## Constraints
- Kaggle T4 16 GB, ~12 h sessions, Internet must be ON (Overture, Esri, torch.hub).
- Seeds fixed; paths configurable; artifacts in English.
- Do not claim Esri ToS allows ML training: flag it for the team to verify and cite.

## Checklist
- [x] T1 Notebook 01 (data + alignment) — route: delegated writer (2+ non-trivial files)
- [x] T2 Notebook 02 (teardown + backbone swap) — route: delegated writer
- [x] T3 DECISIONS.md + README.md — route: delegated writer
- [x] T4 Static validation (nbformat schema, ast-parse every code cell) + parent review

## Acceptance criteria
- Both notebooks are valid nbformat v4 JSON, all code cells parse, no outputs stored.
- Each notebook starts with a markdown summary and a single CONFIG cell.
- Results are written to `/kaggle/working/results/*.json` so the team can send them back.

## Checks
- `python -c "import json,ast; ..."` parse of every code cell (local, no execution).
- TDD: off (no project test config; notebooks are run on Kaggle).

## Progress
- 2026-10-21: branch `feat/e2-notebooks` created; plan agreed.
- 2026-10-21: written `Colabs/01_data_exploration_alignment.ipynb` (56 cells; Parts A/B/C; static parse OK, not executed).
- 2026-10-21: written `Colabs/02_teardown_backbone_swap.ipynb` (41 cells; Block A teardown + Block B training; static parse OK, not executed).
- 2026-10-21: written `DECISIONS.md` (D01-D21) and `README.md` (Spanish).
- Builder scripts live in the session scratchpad (nbhelp.py, nb1_part*.py, nb2_part*.py, build_nb*.py, validate.py); notebooks regenerate from them.

## Next step
T1–T3 via delegated writer, then T4.
- 2026-10-21: parent review done (validate.py ALL OK, 0 parse errors; ERF/rasterize/metrics spot-checked). Runtime on Kaggle still pending.
