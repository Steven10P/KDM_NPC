# Experiments Index — Tesis_KDM_NPC

Layout: each `exp_<n>/` holds `DESIGN.md` (research design) → `IMPLEMENTATION.md` (build plan)
→ `results/<condición>/` (runs) → `reports/summary.md` (write-up). See any `DESIGN.md` for detail.

Nomenclatura: `exp_<NN>_<dataset>_<bloque>`, bloque ∈ {`npc_repro`, `kdm_base`, `kdm_sweep`}.
Condición (bajo `results/`): `<modelo>_<estructura>_<hparams>_seed<NN>`.

Tracker: MLflow local (`mlflow.db`, sqlite, gitignored — respaldo en Drive).
Cómputo GPU: Kaggle Kernels (primario) / Colab (respaldo manual).

| Exp | Título | Estado | Hipótesis (1 línea) | Veredicto | Fecha |
|-----|-------|--------|---------------------|---------|------|
| [exp_01](exp_01_mnist_npc_repro/DESIGN.md) | Replicación NPC en MNIST-Addition (Gate #0) | In Progress | NPC(K) y NPC(D) reproducen la Tabla 2 dentro de ±1 std en nuestro ambiente | — | 2026-07-12 |
