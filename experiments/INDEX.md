# Experiments Index — Tesis_KDM_NPC

Layout: each `exp_<n>/` holds `DESIGN.md` (research design) → `IMPLEMENTATION.md` (build plan)
→ `results/<condición>/` (runs) → `reports/summary.md` (write-up). See any `DESIGN.md` for detail.

Nomenclatura: `exp_<NN>_<dataset>_<bloque>`, bloque ∈ {`npc_repro`, `kdm_base`, `kdm_sweep`}.
Condición (bajo `results/`): `<modelo>_<estructura>_<hparams>_seed<NN>`.

Tracker: MLflow local (`mlflow.db`, sqlite, gitignored — respaldo en Drive).
Cómputo GPU: Kaggle Kernels (primario) / Colab (respaldo manual).

| Exp | Título | Estado | Hipótesis (1 línea) | Veredicto | Fecha |
|-----|-------|--------|---------------------|---------|------|
| [exp_01](exp_01_mnist_npc_repro/DESIGN.md) | Replicación NPC en MNIST-Addition (Gate #0) | ✅ Complete | NPC(K) y NPC(D) reproducen la Tabla 2 dentro de ±1 std en nuestro ambiente | **Superado** — NPC(K) 99.20% vs 99.189±0.08%; NPC(D) 99.00% vs 99.171±0.11%; verosim. circuito K casi exacta (0.01007) | 2026-07-14 |
| [exp_02](exp_02_mnist_kdm_base/DESIGN.md) | Cascada KDM en MNIST-Addition: Cartesian vs. Distributional | In Progress | Ambas variantes del KDM final dan efectividad similar; si es cercano, se prefiere Distributional por escalabilidad | — | 2026-07-14 |
