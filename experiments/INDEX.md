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
| [exp_02](exp_02_mnist_kdm_base/DESIGN.md) | Cascada KDM en MNIST-Addition: Cartesian vs. Distributional | ✅ Complete | Ambas variantes del KDM final dan efectividad similar; si es cercano, se prefiere Distributional por escalabilidad | **Cartesian gana claro** (99.40% vs 97.17% accuracy; 99.37% vs 82.57% accuracy de atributos) — no son matemáticamente equivalentes (normalizar-antes-de-promediar ≠ promediar-antes-de-normalizar) | 2026-07-14 |
| [exp_03](exp_03_mnist_kdm_sweep/DESIGN.md) | Mejor KDM en MNIST-Addition: barrido de hiperparámetros + confirmación 5 semillas | ✅ Complete | Con los hiperparámetros correctos, KDM iguala o supera a NPC en MNIST-Addition | **Confirmado** — KDM 99.314%±0.099% (5 semillas) supera a NPC(K) 99.189±0.08% y NPC(D) 99.171±0.11%, diferencia significativa (t-test una muestra, p<0.05); solo `lr_kdm` 1e-3→3e-3 fue necesario, otros ejes no fueron aditivos | 2026-07-15 |
| [exp_04](exp_04_mnist_evaluation/DESIGN.md) | Evaluación extendida + interpretabilidad nativa (KDM vs. NPC) en MNIST-Addition | ✅ Complete | Confusión/ROC/PR por muestra y mecanismos de interpretabilidad nativos (no genéricos) diferencian cualitativamente a KDM y NPC más allá de la accuracy agregada | **Confirmado** — ROC-AUC casi idéntico (≥0.9996) entre los 3 modelos; KDM: entropía de atribución más alta en errores (1.263 vs 0.986 nats, incertidumbre nativa); NPC: MPE-alignment=1.0 en ambas variantes (aciertos por razonamiento correcto), CE-correction 0.75(K)/0.857(D); circuito Data 15,410 nodos vs. 140 de Knowledge para accuracy menor | 2026-07-15 |
| [exp_05](exp_05_gtsrb_kdm_npc/DESIGN.md) | KDM vs. NPC en GTSRB (imágenes reales, 43 clases, sensibilidad de `sigma_mult`/escalamiento de `n_comp_final`) | 🔄 Fase A + A2 + confirmación completas, Fase B pendiente de aprobación | KDM Cartesian sostiene su ventaja bajo ruido real de imagen y un espacio de salida ~2.3× mayor | `lr_kdm=3e-3` (heredado de MNIST) diverge en GTSRB; `lr_kdm=3e-4` converge. Con eso fijo, `sigma_mult=0.5` da la misma accuracy (99.95%) con TV **19.4× mejor** (0.0025 vs. 0.0488) — `search-lr3e4-sig05`, ganador final. Combinarlo con `n_comp_final=645` no ayuda (ejes no aditivos, igual que en `exp_03`) | 2026-07-16 |
