# Experiment Design: Cascada KDM en MNIST-Addition — Cartesian vs. Distributional

**Experiment**: experiments/exp_02_mnist_kdm_base/
**Project**: Tesis_KDM_NPC
**Date**: 2026-07-14
**Author**: Brayan Steven Peña Delgadillo
**Status**: In Progress

---

## 1. Hipótesis

La cascada KDM (tronco ResNet-34 compartido + 2 cabezas KDM por dígito + KDM
final para la suma) entrena en MNIST-Addition con ambas formas de construir el
KDM final — Opción A (`cartesian_product` + punto único) y Opción B (KDM
distribucional nativa vía `CrossProductKernelLayer`) — dando resultados
similares en efectividad; si la diferencia es marginal, se prefiere la Opción
B por su mejor escalabilidad a datasets con más atributos (GTSRB, CelebA,
AwA2).

## 2. Contexto y las dos opciones

Ver discusión completa en el plan de implementación aprobado
(`~/.claude/plans/proud-dazzling-ritchie.md`, íntegro también más abajo en
`IMPLEMENTATION.md`). Resumen:

Pipeline común: imagen `(bs,3,28,56)` → tronco ResNet-34 (`fc→Identity`) →
`neck (bs,512)` → 2 cabezas `KDMClassModel(512,10,encoder=Identity(),n_comp)`
→ `p1,p2` (bs,10) cada una.

- **A (cartesian)**: `cartesian_product([p1,p2])` → `(bs,100)` → punto único
  (`pure2dm`) → `KDMLayer(dim_x=100,dim_y=19,kernel=Cosine())` → `dm2discrete`.
- **B (distributional)**: KDM explícita de 100 componentes (`comp2dm`, pesos
  `p1[i]*p2[j]`, vectores `concat(onehot_i,onehot_j)`) →
  `KDMLayer(dim_x=20,dim_y=19,kernel=CrossProductKernelLayer(10,Cosine(),Cosine()))`.

## 3. Experimental Setup

- **Dataset**: mismo `D̄` congelado de `exp_01`
  (`bspenad10/mnist-addition-npc`, hash
  `4d640365ca4b505d14ae79a83dcdc799d546a121a676e0a26a01c42fb2b9e07d`), cargado
  con `NPCDataset` (`npc-models/dataset.py`).
- **Modelo**: `src/models/kdm_cascade.py::KDMCascade(final_mode=...)`.
- **Entrenamiento** (ronda de comparación, no la corrida oficial): end-to-end,
  30 épocas, batch 256, SGD lr 0.01 momentum 0.9 (tronco) + Adam lr 1e-3
  (parámetros KDM), semilla 42 única. `n_comp_head=100`, `n_comp_final=190`.
- **Hardware**: Kaggle GPU (mismo flujo que `exp_01`).

## 4. Condiciones

| Condición | `final_mode` | Config |
|---|---|---|
| `kdm-final-cartesian_seed42` | `"cartesian"` | `scripts/cartesian/` |
| `kdm-final-distributional_seed42` | `"distributional"` | `scripts/distributional/` |

## 5. Evaluation Protocol

- **Tiempo**: wall-clock total y por época.
- **Efectividad**: accuracy end-to-end (suma), accuracy y TV media por dígito
  (misma fórmula que `test_neural.py` de NPC).
- **Costo**: #parámetros totales y de la etapa final.
- Resultados en `results/<condición>/metrics.json`.

## 6. Decision Rule

- Si una opción domina en tiempo Y efectividad → esa opción.
- Si es un empate/trade-off cercano (diferencia de accuracy <1-2 puntos,
  tiempo similar) → **Opción B (distributional)** por escalabilidad, según lo
  acordado con el usuario.
- Documentar la decisión acá antes de escalar a `exp_03` (siguiente dataset).

## 7. Reproducibility Checklist

- [x] Semilla fija (42)
- [x] Dataset congelado reusado de exp_01, hash verificado en el kernel
- [ ] Config por condición en `scripts/<condición>/`
- [ ] `git_commit.txt` por corrida
- [ ] Checkpoints → Drive (no a git)

## 8. Next Steps

1. `src/models/kdm_cascade.py`.
2. Smoke test local (CPU, ambas variantes).
3. `IMPLEMENTATION.md` + kernels de Kaggle.
4. Lanzar cartesian primero, verificar, luego distributional.
5. Comparar y documentar decisión (sección 6).
