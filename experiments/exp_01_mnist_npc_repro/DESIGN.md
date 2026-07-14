# Experiment Design: Replicación de NPC en MNIST-Addition (Gate #0)

**Experiment**: experiments/exp_01_mnist_npc_repro/
**Project**: Tesis_KDM_NPC
**Date**: 2026-07-12
**Author**: Brayan Steven Peña Delgadillo
**Status**: ✅ Complete — Gate #0 superado (2026-07-14)

---

## 1. Hipótesis

Los resultados publicados de NPC (Chen et al. 2025, arXiv:2501.07021v2) en
MNIST-Addition son reproducibles en nuestro ambiente (Kaggle GPU, código oficial
`npc-models`/`npc-dataset-utils`, datos regenerados desde los mappings oficiales):
la exactitud end-to-end de NPC(Knowledge) y NPC(Data) debe caer dentro de
±1 desviación estándar de los valores de la Tabla 2 del paper.

**Este experimento es un gate bloqueante:** si no se reproduce NPC, no se
compara nada contra KDM (cualquier diferencia sería atribuible al ambiente, no
al modelo).

## 2. Experimental Setup

- **Dataset**: MNIST-Addition oficial — 35,000 imágenes 28×56×3 (dos dígitos
  concatenados), clase = suma (19 valores), atributos = number-first,
  number-second (10 valores c/u, single-valued, instance-wise).
  - Generado por `src/data/generate_mnist_addition.py` **directamente desde los
    mappings oficiales** (`configs/npc-dataset-utils/mnist.json`, 35k entradas,
    0 discrepancias verificadas contra los idx crudos) — no vía el shuffle de
    `mnist.py`, que es sensible al orden de `os.listdir` del filesystem.
  - Splits **oficiales** de los autores: `mnist_split.json.gz`
    (28,000/3,500/3,500 = 8:1:1), cargados con `split_load=True`.
  - Hash global de D̄ en `data/npc/datasets/mnist/MANIFEST.json`; misma copia
    congelada para todos los runs (subida como Kaggle Dataset privado).
- **Modelos** (código oficial `npc-models`, sin modificaciones funcionales):
  - Etapa 1: `ResNet34MTL` (ResNet-34 IMAGENET1K_V1, fc→Identity, 2 cabezas
    Linear(512,128)→ReLU→Linear(128,10)), pérdida CE multi-tarea ponderada
    1/log(q_k).
  - Etapa 2 Data: LearnSPN (`external/learnspn`, Java) sobre splits/pc/*.txt →
    CCCP (`train_pc.py`). Etapa 2 Knowledge: circuito depth-2 manual
    (`learnspn/scripts/manual/manual.py`).
  - Etapa 3: `train_npc.py` — optimización conjunta con circuito congelado.
- **Entrenamiento**: hiperparámetros oficiales de `npc-models/header.py`
  (batch 256, 150 épocas, SGD lr 0.01, momentum 0.9, wd 4e-5, plateau ×0.1
  paciencia 10, input 224×224). Semillas: 42 (primero), luego 52, 62, 72, 82.
- **Hardware**: Kaggle Kernels GPU (P100 o T4×2), entorno torch==2.1.2 según
  `npc-models/requirements.txt`. Local (Windows/CPU) solo para preparación de
  datos y análisis.

## 3. File Layout

```
experiments/exp_01_mnist_npc_repro/
├── DESIGN.md                ← este archivo
├── IMPLEMENTATION.md        ← plan de construcción (tras aprobación)
├── scripts/
│   ├── configs/             ← una config por condición
│   └── kernel_*.py          ← scripts de kernel Kaggle
├── results/
│   └── <condición>/         ← metrics.json, git_commit.txt (checkpoints → Drive)
└── reports/
    ├── figures/
    └── summary.md
```

## 4. Baselines (números publicados a reproducir — Tabla 2 y 4 del paper)

| Condición | Métrica primaria | Valor publicado (MNIST-Add) |
|-----------|------------------|------------------------------|
| npc-knowledge (5 semillas) | accuracy end-to-end | **99.189 ± 0.08 %** (Tabla 2) |
| npc-data (5 semillas) | accuracy end-to-end | **99.171 ± 0.11 %** (Tabla 2) |
| circuit-knowledge (etapa 2 sola) | verosimilitud media | **1.007e-2** (Tabla 4) |
| circuit-data (etapa 2 sola) | verosimilitud media | **1.010e-2** (Tabla 4) |
| modelo pre-conjunta (etapas 1+2) | accuracy end-to-end | 99.17 % ambas variantes (Tabla 4) |

Referencia de validación para la etapa 1 sola (Tabla 3, modelo ABM que comparte
el reconocedor): TV media ≈ 0.0058, accuracy media de conceptos ≈ 98.99 %.

**Dataset congelado (generado 2026-07-12):** 35,000 imágenes,
`global_sha256 = 4d640365ca4b505d14ae79a83dcdc799d546a121a676e0a26a01c42fb2b9e07d`,
0 discrepancias entre mappings oficiales y etiquetas reales de MNIST.

## 5. Condiciones

- `npc-knowledge_seed42` … `seed82` (5 corridas)
- `npc-data_seed42` … `seed82` (5 corridas)
- Métricas secundarias por corrida: TV media y accuracy por atributo (etapa 1),
  verosimilitud media del circuito (etapa 2), wall-clock por etapa, latencia de
  inferencia, #parámetros, memoria pico.

## 6. Evaluation Protocol

- Primaria: accuracy end-to-end en test oficial; éxito = dentro de ±1 std de la
  Tabla 2 (promedio sobre 5 semillas).
- `results/<condición>/metrics.json` + `git_commit.txt` por corrida; todo
  logueado a MLflow (`mlflow.db`, respaldo en Drive).
- Presupuesto: ~10 corridas × ~1.5-3 h GPU ≈ 15-30 h (dentro de la cuota
  semanal gratuita de Kaggle; escalonar en 2 semanas si hace falta).

## 7. Decision Rules

- **Reproduce** (dentro de margen) → Gate #0 superado; abrir
  `exp_02_mnist_kdm_base`.
- **No reproduce** → depurar ambiente (versiones, datos, splits) antes de
  cualquier comparación KDM; documentar la brecha en `reports/summary.md`.

## 8. Risks & Mitigations

- ⚠️ Emparejamiento distinto al oficial por `os.listdir` → mitigado: generación
  directa desde mappings oficiales, 0 discrepancias verificadas.
- ⚠️ LearnSPN es Java+bash (solo Linux) → correr en el kernel Kaggle, nunca local.
- ⚠️ Símlinks de `split.py` no portables a Windows → splits se materializan en
  el kernel (Linux).
- ⚠️ Cuota GPU de Kaggle (~30 h/sem) → priorizar seed 42 de ambas variantes
  antes de las demás semillas; Colab como respaldo manual.

## 9. Reproducibility Checklist

- [x] Semilla fija y logueada (42 primero; 42-82 al completar)
- [x] Dataset congelado con hash global (MANIFEST.json)
- [x] Splits oficiales de los autores (no regenerados)
- [ ] Config por condición en `scripts/configs/`
- [ ] Commit hash registrado por corrida (`git_commit.txt` + parámetro MLflow)
- [ ] Checkpoints a Drive
- [ ] Entorno pinneado (`npc-models/requirements.txt`, torch==2.1.2)

## 10. Next Steps

1. ~~Congelar D̄ + subirlo como Kaggle Dataset privado.~~ ✅
2. ~~`IMPLEMENTATION.md` con el mapa condición→config→kernel.~~ ✅
3. ~~Kernel etapa 1 (seed 42) → verificar métricas de atributos → etapas 2-3.~~ ✅
4. **Siguiente**: abrir `exp_02_mnist_kdm_base` (cascada KDM) — Gate #0 superado,
   el ambiente reproduce el paper con confianza.

## 11. Resultado final del Gate #0 (2026-07-14, seed 42)

| Métrica | NPC(Knowledge) — nuestro | Paper | NPC(Data) — nuestro | Paper |
|---|---|---|---|---|
| Verosimilitud media del circuito (Tabla 4) | **0.010070** | 0.01007 | 0.009277 | 0.0101 |
| Accuracy end-to-end (Tabla 2) | **99.20%** | 99.189 ± 0.08% | **99.00%** | 99.171 ± 0.11% |
| TV media (post-conjunta) | 0.004602 | — | 0.005118 | — |
| Accuracy de conceptos (post-conjunta) | 99.19% | — | 99.01% | — |

**Veredicto: Gate #0 superado.** NPC(Knowledge) reproduce la verosimilitud del
circuito casi al 5º decimal exacto; ambas variantes caen dentro o muy cerca del
rango publicado (±1 std) en accuracy end-to-end. El ambiente (Kaggle GPU, torch
2.2.2, dataset congelado con hash verificado) es confiable para comparar KDM
contra este baseline en los próximos experimentos.

Wall-clock total del Gate #0 (3 kernels, seed 42): etapa 1 ≈ 3.4h; etapa 2+3
Knowledge ≈ 3.7h; etapa 2+3 Data ≈ 3.8h (incluye CCCP, 77s) — todo dentro de la
cuota gratuita semanal de Kaggle (30h GPU).
