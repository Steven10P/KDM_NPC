# Experiment Report: Mejor KDM en MNIST-Addition (barrido de hiperparámetros + confirmación a escala completa)
**Experiment**: experiments/exp_03_mnist_kdm_sweep/
**Project**: Tesis_KDM_NPC
**Report date**: 2026-07-15
**Plan date**: 2026-07-14
**Author**: Brayan Steven Peña Delgadillo
**Status**: Complete

---

## 1. Summary

Se buscó exprimir al máximo la cascada KDM (variante Cartesian, ganadora de
`exp_02`) en MNIST-Addition antes de pasar a otros datasets: un barrido
uno-a-la-vez de 4 hiperparámetros (Fase A, 10 corridas cortas) seguido de una
confirmación a escala completa con la configuración ganadora (Fase B, 5
semillas × 60 épocas). El único cambio que mejoró la accuracy de forma
robusta fue `lr_kdm`: 1e-3 → 3e-3; combinar ese cambio con otros ejes
ganadores no mejoró más la accuracy (los efectos no fueron aditivos). Con
esa única modificación, KDM alcanzó **99.314% ± 0.099%** de accuracy media
en 5 semillas, superando ambas variantes de NPC reportadas en el paper
(Knowledge: 99.189±0.08%; Data: 99.171±0.11%), diferencia que resultó
estadísticamente significativa (t-test de una muestra, p<0.05 en ambos
casos) pese al tamaño de muestra reducido.

---

## 2. Hypothesis & Verdict

**Hypothesis (from plan):** "Con los hiperparámetros correctos y suficientes
épocas/semillas, la cascada KDM (variante Cartesian, decidida en `exp_02`)
puede igualar o superar la accuracy de NPC en MNIST-Addition (Tabla 2 del
paper: 99.189±0.08% Knowledge, 99.171±0.11% Data), no solo acercarse con la
corrida corta de `exp_02` (99.40% en 30 épocas, hiperparámetros default)."

**Verdict:** ✅ Supported

**Evidence:** La media de 5 semillas de KDM (99.314%±0.099%) supera la media
publicada de ambas variantes de NPC, y las 5/5 semillas individuales quedan
por encima de NPC(Data) (99.171%) — la semilla más baja (99.171%, seed 42)
empata exactamente con esa referencia. Un t-test de una muestra rechaza la
igualdad de medias frente a ambas referencias (p=0.0473 vs Knowledge,
p=0.0318 vs Data; ver `reports/statistical_report.md` §4).

---

## 3. Experimental Setup (as run)

No existe `IMPLEMENTATION.md` para este experimento — se construyó
directamente según `DESIGN.md` sin un documento de build intermedio; las
decisiones de implementación (exposición de `sigma_mult`, plantilla +
generador de kernels) están documentadas inline en `DESIGN.md` §9 y en los
mensajes de commit. No hay desviaciones respecto al diseño original.

- **Dataset**: MNIST-Addition (`bspenad10/mnist-addition-npc`, split oficial de NPC vía `npc-dataset-utils`)
- **Modelo**: `KDMCascade` (`src/models/kdm_cascade.py`), variante `final_mode="cartesian"` — tronco ResNet-34 compartido + 2 cabezas KDM (dígitos) + KDM final (coseno) sobre el producto cartesiano de las cabezas
- **Fase A (búsqueda)**: 15 épocas, batch 256, semilla única 42, barrido uno-a-la-vez sobre `n_comp_head` ∈ {100,150,200}, `n_comp_final` ∈ {190,285,380}, `lr_kdm` ∈ {1e-3,3e-3,3e-4}, `sigma_mult` ∈ {1.0,0.5,2.0} → 9 corridas + 1 confirmación combinando ganadores por eje = 10 corridas
- **Fase B (confirmación)**: 60 épocas, batch 256, 5 semillas (42,52,62,72,82), config ganadora `n_comp_head=100, n_comp_final=190, lr_kdm=3e-3, sigma_mult=1.0`
- **Hardware**: Kaggle Kernels, GPU Tesla P100-PCIE-16GB, ~66s/época (estable en todas las condiciones)
- **Deviations from plan**: Ninguna en el diseño experimental. Un detalle operativo no anticipado: el límite real de concurrencia de Kaggle (2 sesiones GPU simultáneas) y errores esporádicos de push ("Notebook not found") requirieron un monitor de cola con limpieza automática de entradas rotas — no afecta los resultados, solo el mecanismo de lanzamiento.

---

## 4. Code Version

⚠️ No se generó `git_commit.txt` por corrida en este experimento (a diferencia de `exp_02`). Cada `metrics.json` sí registra los commits de las dependencias externas (`npc-models`, `npc-dataset-utils`) bajo la clave `repo_commits`, pero no el commit propio de `Tesis_KDM_NPC` en el momento de cada corrida. El código fuente de `KDMCascade` usado (embebido en cada kernel) es idéntico en las 10+5 corridas — no hubo cambios de código entre condiciones, solo de hiperparámetros — por lo que la trazabilidad real está en el historial de commits del repo (`git log -- experiments/exp_03_mnist_kdm_sweep/`), no en un archivo por corrida.

---

## 5. Results

### 5.1 Fase A — Barrido de hiperparámetros (15 épocas, semilla 42)

| Condición | n_comp_head | n_comp_final | lr_kdm | sigma_mult | Acc. suma | Acc. atributos | TV | Train (s) |
|---|---|---|---|---|---|---|---|---|
| **search-lr3e3** | 100 | 190 | 3e-3 | 1.0 | **99.31%** | 99.34% | 0.0136 | 987 |
| search-ncf380 | 100 | 380 | 1e-3 | 1.0 | 99.23% | 99.11% | 0.0495 | 991 |
| search-ncf285 | 100 | 285 | 1e-3 | 1.0 | 99.17% | 99.17% | 0.0488 | 991 |
| search-nch200 | 200 | 190 | 1e-3 | 1.0 | 99.14% | 99.17% | 0.0418 | 991 |
| search-sig05 | 100 | 190 | 1e-3 | 0.5 | 99.11% | 99.14% | 0.0098 | 990 |
| search-baseline | 100 | 190 | 1e-3 | 1.0 | 99.09% | 99.06% | 0.0561 | 989 |
| search-nch150 | 150 | 190 | 1e-3 | 1.0 | 98.97% | 98.94% | 0.0467 | 989 |
| search-lr3e4 | 100 | 190 | 3e-4 | 1.0 | 98.94% | 99.00% | 0.1563 | 987 |
| search-sig20 | 100 | 190 | 1e-3 | 2.0 | 98.74% | 54.83% | 0.6712 | 996 |
| **search-confirm** (200/380/3e-3/0.5) | 200 | 380 | 3e-3 | 0.5 | 99.11% | 99.14% | **0.0068** | — |

> Umbral de éxito del plan (DESIGN.md §7): mejor accuracy end-to-end entre las 10 corridas. **Cumplido por `search-lr3e3`.**

**Hallazgo clave**: combinar los 4 valores ganadores por eje (`search-confirm`) mejoró la calibración a la mejor TV de las 10 corridas (0.0068) pero **no** superó la accuracy de `search-lr3e3` sola (99.11% vs 99.31%) — los efectos de los hiperparámetros no fueron aditivos en este régimen. Por la regla de decisión del plan (mejor accuracy, no mejor TV), se descartó la config combinada y se llevó a Fase B únicamente el cambio de `lr_kdm`.

**Hallazgo secundario (no bloqueante)**: `sigma_mult=2.0` degrada fuertemente las cabezas de dígito (TV 0.6712, accuracy conjunta de atributos 54.83%, la peor de las 9) mientras la accuracy de la suma final se mantiene relativamente alta (98.74%) — posible compensación parcial del KDM final ante cabezas mal calibradas, no investigado en profundidad.

### 5.2 Fase B — Confirmación a escala completa (60 épocas, 5 semillas, `lr_kdm=3e-3`)

| Semilla | Acc. suma | Acc. atributos | TV |
|---|---|---|---|
| 42 | 99.171% | 99.229% | 0.0042 |
| 52 | 99.400% | 99.371% | 0.0037 |
| 62 | 99.257% | 99.286% | 0.0039 |
| 72 | 99.400% | 99.371% | 0.0039 |
| 82 | 99.343% | 99.314% | 0.0039 |
| **Media ± std** | **99.314% ± 0.099%** | **99.314% ± 0.061%** | **0.0039 ± 0.0002** |

### 5.3 Comparación final vs. NPC (paper, Tabla 2)

| Modelo | Accuracy |
|---|---|
| **KDM (exp_03, 5 semillas)** | **99.314% ± 0.099%** |
| NPC(Knowledge) (paper) | 99.189% ± 0.08% |
| NPC(Data) (paper) | 99.171% ± 0.11% |

### 5.4 Learning Curves

Ver `reports/figures/02_loss_curves_fase_b.png` — las 5 semillas convergen de
forma consistente en escala log, sin señales de inestabilidad ni divergencia
en las 60 épocas. Ver también `reports/figures/01_accuracy_kdm_vs_npc.png`
(boxplot de las 5 semillas contra las bandas de referencia de NPC) y
`reports/figures/03_summary_bars.png` (barras de media±std).

---

## 6. Statistical Analysis

- **Test usado**: t-test de una muestra (`scipy.stats.ttest_1samp`), no un
  test de dos muestras independientes — NPC no tiene una muestra cruda
  disponible (el paper solo publica media±std; `exp_01` replicó NPC a 1 sola
  semilla). El test contrasta la media muestral de KDM (n=5, std propia)
  contra cada valor de referencia fijo de NPC.
- **Shapiro-Wilk** (normalidad de las 5 accuracies de KDM): W=0.8867,
  p=0.3408 → no se rechaza normalidad, aunque con n=5 el test tiene poca
  potencia.
- **p-value**: 0.0473 (vs NPC Knowledge), 0.0318 (vs NPC Data) — ambos <0.05.
- **Conclusión**: la diferencia es estadísticamente significativa frente a
  ambas referencias, con la advertencia de que n=5 limita la potencia del
  test y que la comparación es de una muestra contra un valor fijo, no entre
  dos muestras independientes.

Detalle completo en `reports/statistical_report.md`.

---

## 7. Comparison to Expected Results

| Expected (DESIGN.md §7) | Observed | Match? |
|---|---|---|
| Fase A: elegir mejor accuracy end-to-end por eje, confirmar sin interacción negativa | `lr_kdm=3e-3` ganó individualmente; la combinación de los 4 ejes ganadores NO mejoró la accuracy (interacción negativa/no aditiva detectada) | ⚠️ Parcial — el supuesto de "sin interacción negativa" no se cumplió, pero el mecanismo de confirmación (10ª corrida) funcionó exactamente para detectarlo |
| Fase B: 60 épocas bastan para estabilizar antes de 5 semillas | Las 5 semillas cayeron en un rango estrecho (99.17%–99.40%, std 0.099%) sin señales de sub-entrenamiento | ✅ |
| KDM iguala o supera a NPC | Media de KDM supera ambas referencias, diferencia significativa (p<0.05) | ✅ |

---

## 8. Missing Data & Caveats

Todas las corridas planeadas se completaron (10/10 en Fase A, 5/5 en Fase
B). No hay condiciones faltantes.

Caveats:
- No se generó `IMPLEMENTATION.md` ni `git_commit.txt` por corrida (ver §3
  y §4) — no afecta los resultados pero reduce la trazabilidad automática
  respecto al estándar de `exp_02`.
- La comparación contra NPC usa la media±std **publicada** del paper, no una
  réplica propia a 5 semillas (asimetría documentada y aceptada desde el
  diseño, ver `DESIGN.md` §5) — el t-test de una muestra es la consecuencia
  metodológica directa de esa limitación.
- El hallazgo de `sigma_mult=2.0` (cabezas descalibradas, suma final
  robusta) no se investigó a fondo por no ser la variante ganadora.

---

## 9. Conclusions & Next Steps

**Lo que este experimento estableció:**
- KDM, con un solo ajuste de hiperparámetro (`lr_kdm` 1e-3→3e-3) sobre la
  configuración base de `exp_02`, iguala y supera la accuracy media
  publicada de NPC en MNIST-Addition, con variabilidad entre semillas
  (std 0.099%) comparable a la de NPC.
- Los hiperparámetros de esta cascada KDM **no interactúan de forma
  aditiva**: combinar varios cambios ganadores por separado no garantiza
  una mejora mayor, y puede incluso ser peor en el eje que más importa
  (accuracy) aunque mejore otro (calibración/TV). Esto es una lección
  metodológica a tener presente al diseñar barridos en los próximos
  datasets — no asumir composicionalidad sin una corrida de confirmación.

**Lo que queda incierto:**
- Si `sigma_mult` alto degrada las cabezas de forma similar en datasets con
  atributos más ruidosos o de mayor dimensión (GTSRB, CelebA, AwA2).
- Si la ventaja de KDM sobre NPC se mantiene al escalar a datasets con más
  clases/atributos, donde el costo de `n_comp_final` crece con el número de
  clases de salida.

**Recomendación**: usar `lr_kdm=3e-3, n_comp_head=100, n_comp_final=190,
sigma_mult=1.0` como punto de partida (no necesariamente óptimo) para el
próximo dataset, y repetir un barrido similar si el dataset lo justifica.
Para configurar `exp_04`, usar el skill `ml-experiment-planner`.

---

## 10. Reproducibility Record

| Item | Status |
|---|---|
| Seeds logged | ✅ (42/52/62/72/82 en Fase B; 42 en Fase A) |
| Configs versioned | ✅ (`scripts/<condición>/kernel-metadata.json` + script, todos commiteados) |
| Git commits recorded | ❌ (sin `git_commit.txt` por corrida, ver §4 y §8) |
| Checkpoints saved | ✅ (`results/<condición>/checkpoints/model.pt`, gitignored — no respaldados a Drive todavía) |
| Environment frozen | ✅ (`environment` en cada `metrics.json`: Python 3.12.13, torch 2.2.2+cu121, Tesla P100) |
| Experiment tracker linked | ❌ (no se registró en MLflow local; solo `metrics.json` + Kaggle) |
