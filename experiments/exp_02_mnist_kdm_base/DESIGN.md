# Experiment Design: Cascada KDM en MNIST-Addition — Cartesian vs. Distributional

**Experiment**: experiments/exp_02_mnist_kdm_base/
**Project**: Tesis_KDM_NPC
**Date**: 2026-07-14
**Author**: Brayan Steven Peña Delgadillo
**Status**: ✅ Complete — decisión tomada (2026-07-14): **Cartesian**

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

## 6. Resultado y Decisión (2026-07-14)

| Métrica | Cartesian | Distributional |
|---|---|---|
| Accuracy end-to-end (suma) | **99.40%** | 97.17% |
| Accuracy conjunta de atributos | **99.37%** | 82.57% |
| TV media | **0.0123** | 0.1176 |
| Tiempo/época | 65.9s | 66.6s (empatado) |
| Parámetros del KDM final | 22,800 | 7,600 |

**Decisión: Cartesian.** No fue el escenario "empate cercano" anticipado en la
regla de decisión original — Cartesian domina en efectividad (+2.2 puntos en
accuracy end-to-end, +16.8 puntos en accuracy conjunta de atributos) y empata
en tiempo. Por la regla "si una opción domina en tiempo y efectividad, esa
opción gana", se descarta Distributional pese a su ventaja teórica de
escalabilidad.

**Por qué divergen (no son equivalentes, corrección de un error de análisis
previo):** antes de correr el experimento se asumió que ambas construcciones
del KDM final eran matemáticamente equivalentes (mismo peso $p_1[i]p_2[j]$
para cada combinación). Al revisar la Ec. 12 con más cuidado tras ver la
brecha empírica: **no lo son**, porque difieren en el orden de dos
operaciones que no conmutan:
- *Cartesian* normaliza **una sola vez**, sobre el vector ya combinado
  (promedio-antes-de-normalizar).
- *Distributional* calcula un posterior normalizado **por separado para cada
  una de las 100 combinaciones** de dígitos, y **luego** promedia esos 100
  posteriores ya normalizados (normaliza-antes-de-promediar).

Como la normalización es no lineal, promediar-y-normalizar ≠
normalizar-y-promediar. Distributional le da voz completa (posterior afilado)
incluso a combinaciones de dígitos poco probables antes de diluirlas en el
promedio, lo que empeora la señal. Cartesian evita ese paso intermedio.

**Implicación para los próximos datasets (GTSRB, CelebA, AwA2):** se usa
Cartesian como construcción por defecto del KDM final. La ventaja de
escalabilidad de Distributional (evitar materializar el vector denso
completo) puede volver a evaluarse si el producto de cardinalidades de
atributos crece mucho (p. ej. AwA2), pero con la penalización de efectividad
observada acá, el trade-off ya no es obvio a favor de Distributional — habría
que revalidar caso por caso, no asumir.

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
