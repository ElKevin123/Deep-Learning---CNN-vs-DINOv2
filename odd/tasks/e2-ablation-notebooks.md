# E2 — Ablation notebooks (one per architecture) + member guides

## Objective
Three Kaggle/Colab notebooks (ResNet-50, DINOv2 frozen, DINOv2 partial) that share byte-identical
data, augmentation, decoder, loss, metrics and evaluation code, plus one Markdown guide per
team member explaining the architecture and what they may modify.

## Problem / why
Notebooks 03–05 crop Massachusetts on the fly (random train crops, slow epoch 1). The agreed plan
(`plan_de_arquitecturas_y_ablaciones.md`, 2026-10-29) needs: a fixed 512x512 patch dataset,
identical augmentation for all three models (Level 1 base benchmark), and per-model ablations
(Level 2) that only touch backbone-specific knobs.

## Decisions
- Member assignment follows the plan file: 1 = resnet50, 2 = dinov2_frozen, 3 = dinov2_partial.
- Fixed dataset: tile-level resplit (SPLIT_SEED 42, 70/15/15 -> 820/176/175), 3x3 grid (0, 494, 988),
  reject >25 % blank, for ALL splits (train included). Built inside each notebook, deterministic,
  verified by a dataset fingerprint saved in every result JSON.
- Locked config (COMMON, MODEL_BASE, ALLOWED) is guarded by a config fingerprint; overrides outside
  ALLOWED[condition] raise. "base" run must have no overrides.
- Shared augmentation adds a soft synthetic-shadow occlusion (SHADOW_P = 0.3) for all models.
- Official metric unchanged: test @ thr 0.5, no TTA. Threshold sweep on val and flip TTA are reported
  as separate extra columns for all three models.
- Notebooks keep the project's Spanish prose convention (03–05 are Spanish).

## Checklist
- [x] T1 Builder script generating 3 notebooks from shared + per-model cells — route: inline (single builder script; shared code identical by construction)
- [x] T2 Three member guides (Markdown) — route: inline
- [x] T3 Static validation (JSON, ast) + CPU smoke test of shared code with a synthetic dataset — route: inline

## Acceptance criteria
- Shared cells are byte-identical across the three notebooks.
- Every code cell parses; CPU smoke test of dataset prep, augmentation, resnet50 model, loss, metrics,
  one train step and evaluation passes.
- Decoder parameter count asserts 3,098,433.

## Checks
- TDD: off (no project test config; notebooks run on Kaggle). Functional checks: validate script + CPU smoke test.

## Progress
- 2026-10-29: plan read (plan md + Gemini PDF); 03–05 confirmed identical except CONDITION.
- 2026-10-29: built `Colabs/E2-ablaciones/06_resnet50_ablacion.ipynb`, `07_dinov2_frozen_ablacion.ipynb`,
  `08_dinov2_partial_ablacion.ipynb` (29 cells each; 25 shared cells byte-identical; locked fingerprint 234ea433fba9)
  and `GUIA_Integrante{1,2,3}_*.md`. Builder: session scratchpad (`shared_cells.py`, `per_model.py`, `build_ablation_nbs.py`).
- Verification: CPU smoke test (synthetic 12-tile Massachusetts, 2 steps, batch 2) ran every code cell OK for the 3
  notebooks, including all knob variants and custom zones. Trainable params match 03–05 (28.00 / 3.47 / 7.02 M).
  Real GPU runs on Kaggle pending (team).

## Next step
Team runs SMOKE_TEST on Kaggle, compares fingerprints, then the base + ablation runs.
