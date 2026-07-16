# Experiment Design: KDM vs. NPC en GTSRB (German Traffic Sign Recognition Benchmark)

**Experiment**: experiments/exp_05_gtsrb_kdm_npc/
**Project**: Tesis_KDM_NPC
**Date**: 2026-07-15
**Author**: Brayan Steven Peña Delgadillo
**Status**: ✅ Fase A completa (2026-07-16) — ganador: `search-lr3e4` (`lr_kdm=3e-4`)

## Fase A — resultado completo (9/9 corridas)

**Hallazgo central**: de los 4 ejes barridos (`n_comp_per_value`,
`n_comp_final`, `lr_kdm`, `sigma_mult`), **`lr_kdm` es, por lejos, el único
que importa** — y de forma mucho más dramática que en `exp_03`/MNIST (ahí
era "el eje más importante"; acá es la diferencia entre un modelo inútil y
uno casi perfecto). El baseline heredado de MNIST (`lr_kdm=3e-3`) **diverge
catastróficamente** en las 7 corridas que lo mantienen fijo — la pérdida
converge bien las primeras épocas y luego explota sin recuperarse — porque
el KDM final de GTSRB usa `dim_x=3120` (31× más grande que el `dim_x=100`
de MNIST), y esa tasa de aprendizaje simplemente no es estable a esa
escala. Bajarla a `lr_kdm=3e-4` (10× más chica que el ganador de MNIST)
resuelve el problema por completo.

| Condición | n_comp_per_value | n_comp_final | lr_kdm | sigma_mult | Acc. suma | Acc. atributos | TV | Estado |
|---|---|---|---|---|---|---|---|---|
| search-baseline | 10 | 430 | 3e-3 | 1.0 | 8.29% | 0.00% | 0.5726 | diverge (época 4) |
| search-npv15 | 15 | 430 | 3e-3 | 1.0 | 9.05% | 0.00% | 0.5706 | diverge (época 6, tras 98.5% en época 5) |
| search-npv20 | 20 | 430 | 3e-3 | 1.0 | 13.26% | 0.31% | 0.5505 | diverge (época 5) |
| search-ncf172 | 10 | 172 | 3e-3 | 1.0 | 59.98% | 55.45% | 0.2056 | diverge parcial, se recupera algo tras época 10 |
| search-ncf645 | 10 | 645 | 3e-3 | 1.0 | 12.09% | 4.34% | 0.5060 | diverge |
| search-lr1e3 | 10 | 430 | 1e-3 | 1.0 | 8.62% | 0.00% | 0.5664 | diverge (época 3) |
| **search-lr3e4** | **10** | **430** | **3e-4** | **1.0** | **99.95%** | **99.97%** | **0.0488** | **converge perfectamente** |
| search-sig05 | 10 | 430 | 3e-3 | 0.5 | 18.72% | 4.31% | 0.5002 | diverge |
| search-sig20 | 10 | 430 | 3e-3 | 2.0 | 11.04% | 0.00% | 0.5647 | diverge |

**Nota sobre `search-ncf172`**: es la única corrida (además de la
ganadora) que muestra algo de recuperación — con menos componentes finales
(172 vs. 430) el modelo tiene menos parámetros en la capa más grande, lo
que aparentemente amortigua parcialmente la inestabilidad de
`lr_kdm=3e-3`, aunque sin llegar a converger limpiamente. Es consistente
con la hipótesis: más componentes finales = más parámetros sensibles a
una tasa de aprendizaje demasiado alta.

**Decisión (regla del §5)**: ganador de Fase A = mejor accuracy end-to-end
entre las 9 corridas → **`search-lr3e4`**
(`n_comp_per_value=10, n_comp_final=430, lr_kdm=3e-4, sigma_mult=1.0`).
Ninguna corrida de confirmación combinada es necesaria — la diferencia es
tan grande (99.95% vs. la segunda mejor, 60.0%) que no hay ambigüedad de
interacción no-aditiva que resolver, a diferencia de `exp_03`.

## Próximos pasos: Fase B (pendiente de aprobación)

Con Fase A cerrada, el siguiente paso del protocolo (`DESIGN.md §5`,
`IMPLEMENTATION.md §4`) es **Fase B**: confirmación a escala completa con
la configuración ganadora — 60 épocas × 5 semillas (42, 52, 62, 72, 82),
igual que `exp_03`. **No se ha lanzado todavía** — queda pendiente de
aprobación explícita antes de comprometer ~5× el cómputo de GPU ya usado
en Fase A (9 corridas cortas ≈ 3h; Fase B completa ≈ 5h más).

Antes de lanzar, vale la pena decidir explícitamente:
- Si vale la pena una corrida corta adicional probando `n_comp_final`
  más bajo (ej. 172-290) **combinado con** `lr_kdm=3e-4` — dado que
  `search-ncf172` fue la única otra corrida con señales de recuperación,
  y ese eje nunca se probó junto con la tasa de aprendizaje correcta.
- Si 60 épocas siguen siendo razonables dado que `search-lr3e4` ya llega a
  99.95% en solo 15 épocas (posible sobre-entrenamiento a 60 épocas, o
  margen para converger aún más si TV sigue bajando).

---

## 0. Selección de dataset (evaluación de viabilidad)

Los tres candidatos considerados para el siguiente bloque, con evidencia
verificada directamente sobre el repo (no supuestos):

| | **GTSRB** | CelebA | AwA2 |
|---|---|---|---|
| Imágenes reales disponibles | ✅ (con una corrección: la copia inicial en `data/gtsrb/` resultó ser una distribución incompatible — nombres de archivo distintos a los que espera `gtsrb_split.json.gz`; se descargó el mirror oficial correcto, `meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`, 39,209 imágenes, ver `IMPLEMENTATION.md §6` paso 2) | ❌ solo el `.txt` de atributos (26MB); faltan las imágenes (~1.3GB+) | ❌ carpeta casi vacía (116KB) |
| Split congelado | ✅ `gtsrb_split.json.gz` ya existe en `npc-dataset-utils/configs` | existe el `.json.gz`, pero sin imágenes es irrelevante | idem |
| Clases | 43, con desbalance de clase real (151 a 1501 imágenes/clase observado) | ~200K instancias (`instance_wise`) | 50 clases de animal |
| Atributos | 4, **single-hot**: color(3), shape(4), symbol(26), text(10) | 5, **multi_hot** (varios valores simultáneos) | 4 mostrados, **multi_hot** |
| Circuito Data (LearnSPN) | 359 nodos | 4,445 nodos | 1,611 nodos |
| Circuito Knowledge (manual) | 390 nodos | 6,855 nodos | **106,897 nodos** |
| clase ↔ atributos | tupla **fija/determinista** por clase (sin colisión combinatoria) | requiere lógica multi-etiqueta nueva | requiere lógica multi-etiqueta nueva |

**Decisión: GTSRB.** Es el único sin trabajo de adquisición de datos
pendiente; tiene ruido real de imagen (responde directamente a la pregunta
de sensibilidad de `sigma_mult`); sube de 19→43 clases (prueba real de
escalamiento de `n_comp_final` sin comprometerse a un circuito de 106,897
nodos); mantiene atributos single-hot (adaptación de código más directa
desde `KDMCascade`/NPC ya probados en MNIST-Addition); y sus dos circuitos
precomputados son diminutos, de bajo riesgo computacional en CPU (recordar
`exp_04`: un circuito de "solo" 15,410 nodos ya tardó ~15-20 min en un único
pase de evaluación local — un circuito de 106,897 nodos, como el Knowledge de
AwA2, sería impracticable con el mismo patrón). CelebA y AwA2 quedan para
`exp_06`/`exp_07` una vez resuelta la adquisición de datos, y en el caso de
AwA2, una decisión explícita sobre si correr solo la variante Data.

**Advertencia metodológica (para no sobre-interpretar resultados):** en
MNIST-Addition, `suma = d1+d2` es una función **no inyectiva** — varios pares
de dígitos comparten la misma suma, y `n_comp_final` debe resolver esa
colisión genuina. En GTSRB, `clase = f(color, shape, symbol, text)` es una
tupla **fija por clase** (p. ej. "límite de velocidad 20" siempre es
color-rojo + shape-círculo + symbol-vacío + text-20) — no hay colisión
combinatoria equivalente. Este experimento prueba escalamiento de
`n_comp_final` con **más clases de salida**, no con **más colisión
combinatoria** — son preguntas distintas, y este dataset solo responde la
primera.

**Segunda advertencia, más fuerte, verificada en `gtsrb.json`**: las claves
de `mappings` están indexadas **por nombre de clase** (`"regulatory--
maximum-speed-limit-20": {...}`), no por instancia/imagen individual (a
diferencia de `mnist.json`, que es `instance_wise: true`). Esto significa
que la etiqueta de atributo (color/shape/symbol/text) es **idéntica para
todas las imágenes de una misma clase** — no hay variación de atributo
intra-clase que capturar. En consecuencia, **la descomposición en cabezas de
atributo NO ofrece aquí la misma ventaja de generalización composicional
que en MNIST-Addition** (donde cada cabeza de dígito generaliza a
combinaciones de suma no vistas juntas durante entrenamiento): en GTSRB,
"reconocer los 4 atributos" es informacionalmente equivalente a "reconocer
la clase" directamente — no hay combinaciones atributo-clase no vistas que
el KDM final/circuito deba inferir. Este experimento sigue siendo válido
para las dos preguntas que motivan `exp_05` (sensibilidad de `sigma_mult` a
ruido real, escalamiento de `n_comp_final` con más clases), pero **no** debe
presentarse como una prueba de generalización composicional a combinaciones
no vistas — esa pregunta requeriría un dataset `instance_wise` con
variación de atributo real dentro de cada clase (candidato natural:
CelebA, `exp_06`).

---

## 1. Objetivo

Extender la comparación KDM-vs-NPC (`exp_01`-`exp_04`, MNIST-Addition) a un
dataset con imágenes reales fotografiadas (no dígitos escaneados
sintéticamente combinados) y con más clases de salida (43 vs. 19), para
determinar si la ventaja de KDM Cartesian confirmada en `exp_03` (99.314%
±0.099% vs. NPC 99.189%/99.171%, p<0.05) se sostiene bajo:
1. Ruido real de imagen (iluminación, ángulo, desenfoque de movimiento,
   suciedad en la señal) — no presente en MNIST (dígitos limpios).
2. Un espacio de salida ~2.3× más grande (43 vs. 19 clases).
3. Desbalance de clase real (GTSRB no está balanceado; MNIST-Addition sí lo
   estaba por construcción).

## 2. Hipótesis de escalado

**H1 (sigma_mult ante ruido real)**: el valor óptimo de `sigma_mult` puede
diferir de 1.0 (el ganador en MNIST-Addition, `exp_03`) porque el ruido de
imagen real cambia la geometría del espacio de embeddings del tronco
ResNet-34 — un kernel RBF calibrado para dígitos limpios puede quedar mal
escalado para fotos ruidosas. Se repite el mismo barrido de `exp_03`
(`sigma_mult` ∈ {1.0, 0.5, 2.0}) para ver si la degradación fuerte observada
con `sigma_mult=2.0` en MNIST (TV 0.6712, accuracy de atributos 54.83%) se
repite, se atenúa o se agrava con ruido real.

**H2 (n_comp_final ante más clases)**: en `exp_03`, el `n_comp_final=190`
ganador equivale exactamente a **10 componentes por clase** (190/19=10,
usados por `init_kdm_layer` para estratificar el muestreo inicial por
clase-suma). La hipótesis es que `n_comp_final` en GTSRB necesita **menos**
de 10 componentes/clase — no la misma densidad — porque GTSRB no tiene la
colisión combinatoria que esa densidad resolvía en MNIST-Addition (ver
advertencia metodológica arriba). Se prueba con tres valores, todos
múltiplos exactos de 43 (`stratified_idx` en `kdm_cascade.py:138-148`
exige `n_comp % n_values == 0`, ver `IMPLEMENTATION.md §2`): **172** (4
componentes/clase, bien por debajo de la densidad de MNIST), **430** (10
componentes/clase, misma densidad que el ganador de MNIST), **645** (15
componentes/clase, por encima) — para localizar el punto de rendimientos
decrecientes.

**Veredicto esperado**: si H2 se confirma (190 ≈ 430 ≈ 650 en accuracy),
es evidencia de que el tamaño de `n_comp_final` en KDM está gobernado por la
complejidad de la *relación* clase-atributos (colisión combinatoria), no por
el conteo bruto de clases — un hallazgo metodológico relevante para diseñar
`exp_06`/`exp_07` (CelebA/AwA2, con relaciones clase-atributos más
complejas).

## 3. Protocolo de datos

- **Split**: el ya congelado `gtsrb_split.json.gz` (`npc-dataset-utils`) — no
  se regenera. Train/val/test siguen la proporción estándar de NPC (igual
  metodología que `exp_01`/`exp_03`).
- **Desbalance de clase**: NO se rebalancea artificialmente — es una
  propiedad real de GTSRB (151 a 1501 imágenes/clase observadas en
  `Training/`) y parte del reto que este experimento evalúa. Se documenta
  explícitamente en el checklist de métricas (§6) que el desbalance hace que
  **Average Precision macro** y el **diagrama de fiabilidad por clase** sean
  más informativos que accuracy/ROC-AUC agregados (que ya saturaban en
  MNIST y saturarían aún más aquí si se domina por las clases mayoritarias).
- **Tronco compartido**: **ResNet-34 preentrenado (IMAGENET1K_V1), idéntico
  al usado en MNIST-Addition** — decisión deliberada de **no** cambiar a
  ResNet-50/MobileNetV3. Cambiar dataset Y arquitectura del tronco a la vez
  confundiría el efecto que se quiere medir (¿la ventaja de KDM es del
  dataset, o de la arquitectura?). Mantener el tronco constante aísla la
  variable de interés. Un tronco más grande queda como pregunta abierta para
  un experimento de ablación posterior, no para este.

## 4. Configuración de modelos

### KDM (única variante: Cartesian)

`exp_02` ya descartó Distributional (99.40% vs. 97.17% en MNIST-Addition,
no son matemáticamente equivalentes) — no se repite esa comparación aquí.

Cascada generalizada de **2 → 4 cabezas** (una por atributo: color, shape,
symbol, text), cada una un `KDMClassModel` sobre el neck de 512-dim del
tronco compartido, igual que en MNIST-Addition. KDM final: `KDMLayer` con
`CosineKernelLayer` sobre el producto cartesiano de las 4 cabezas —
`dim_x = 3 × 4 × 26 × 10 = 3,120` (vs. 100 en MNIST-Addition, salto de
31×). Detalles de generalización de código en `IMPLEMENTATION.md §2`.

### NPC (ambas variantes: Knowledge y Data)

- **Knowledge**: circuito manual precomputado, `outputs/manual/gtsrb.spn.txt`
  (390 nodos) — ya existe, no requiere LearnSPN ni Java.
- **Data**: circuito LearnSPN precomputado, `outputs/learnspn/gtsrb.spn.txt`
  (359 nodos) — ya existe.
- Mismo protocolo de 3 etapas que `exp_01` (atributos → circuito → conjunta),
  sin cambios de código (ver `IMPLEMENTATION.md §3`).

## 5. Pipeline de grid-search no aditivo (Fase A)

Mismo protocolo exacto de `exp_03` (el que ya reveló que los hiperparámetros
**no** interactúan aditivamente en MNIST-Addition — no se asume que sí lo
harán aquí, se vuelve a verificar):

**Barrido uno-a-la-vez** (15 épocas, semilla única 42, batch 256):

| Eje | Valores (baseline primero) |
|---|---|
| `n_comp_per_value` (ver `IMPLEMENTATION.md §2` — reemplaza `n_comp_head`) | 10 (misma densidad que MNIST), 15, 20 |
| `n_comp_final` | 430 (10/clase, densidad MNIST), 172 (4/clase), 645 (15/clase) |
| `lr_kdm` (Adam) | 3e-3 (ganador MNIST), 1e-3, 3e-4 |
| `sigma_mult` | 1.0 (ganador MNIST), 0.5, 2.0 |

9 corridas (baseline + 8 variaciones un-eje-a-la-vez) + **1 corrida de
confirmación** combinando los mejores valores por eje = 10 corridas cortas,
igual que `exp_03`.

**Regla de decisión** (idéntica a `exp_03 §7`): ganador de Fase A = mejor
accuracy end-to-end entre las 10 corridas. Si la corrida de confirmación
(combinación de ganadores por eje) NO supera a la mejor corrida individual,
se documenta como evidencia adicional de no-aditividad (como pasó en
`exp_03`) y se usa la mejor corrida individual, no la combinada.

**Solo si Fase A confirma un ganador claro** se pasa a Fase B (confirmación
a escala completa, 60 épocas × 5 semillas) — no se comprometen 5 semillas
sin este paso intermedio, mismo criterio de `exp_03`.

## 6. Checklist de métricas (evaluación holística desde el día 1)

Se reutilizan literalmente los scripts de `exp_04` (no se reescriben):

- [ ] `predictions.npz` por condición (KDM + NPC-Knowledge + NPC-Data):
  probabilidades por muestra, no solo accuracy agregada — desde la primera
  corrida de Fase A, no solo en Fase B (a diferencia de `exp_03`, donde las
  predicciones por muestra no se guardaron y hubo que reconstruirlas en
  `exp_04` desde cero).
- [ ] Matriz de confusión 43×43 normalizada por fila
  (`build_eval_plots.py`, adaptar `N_CLASSES=19→43`).
- [ ] ROC one-vs-rest macro-promediada + AUC, y **Average Precision macro**
  (`build_eval_plots.py`) — con 43 clases desbalanceadas, AP macro es más
  discriminante que ROC-AUC (que ya saturaba en MNIST con 19 clases
  balanceadas; con más clases y desbalance real, ROC-AUC arriesga saturar
  aún más).
- [ ] **Diagrama de fiabilidad / calibración (reliability diagram)** — nuevo,
  no existía en `exp_04`. `sklearn.calibration.calibration_curve` sobre la
  probabilidad top-1 vs. tasa de acierto observada, por bins, para cada
  modelo. Motivado directamente por el desbalance de clase real de GTSRB:
  un modelo puede tener buena accuracy agregada pero probabilidades mal
  calibradas en las clases minoritarias.
- [ ] Curvas de entrenamiento **desde la primera corrida**: pérdida de train
  por época (igual que `exp_03`) + **accuracy/pérdida de validación cada N
  épocas** (nuevo — en `exp_03`/`exp_01` esto faltó y `exp_04` tuvo que
  documentarlo como limitación; aquí se agrega desde el diseño, no como
  parche posterior).
- [ ] Interpretabilidad nativa KDM: atribución por componente + entropía
  (correctas vs. incorrectas) — `build_interpretability_kdm.py`, sin cambios
  de lógica (solo `N_CLASSES`/rutas).
- [ ] Interpretabilidad nativa NPC: MPE alignment rate + CE correction rate
  + estructura del circuito — `build_interpretability_npc.py`, sin cambios
  de lógica.

## 7. Reproducibilidad (checklist previo, antes de arrancar)

- [ ] Presupuesto de GPU de Kaggle verificado antes de la Fase A.
- [ ] Mapeo carpeta-numérica ↔ nombre-semántico de clase verificado (ver
  `IMPLEMENTATION.md §1`) antes de construir el dataset de Kaggle.
- [ ] `git_commit.txt` por corrida (a diferencia de `exp_03`, que no lo
  generó — se corrige aquí).
- [ ] `loss_history` de train Y de validación por época desde la Fase A.

## 8. Próximos pasos

1. Escribir `IMPLEMENTATION.md` (este mismo turno).
2. Construir el mapeo de clases + empaquetar el dataset de Kaggle.
3. Generalizar `KDMCascade` a K=4 cabezas (`src/models/`).
4. Fase A (10 corridas cortas) → elegir configuración ganadora.
5. Fase B (5 semillas) si Fase A confirma un ganador claro.
6. Evaluación holística (checklist §6) + informe final.
