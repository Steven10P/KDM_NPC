# Implementation Plan: KDM vs. NPC en GTSRB

**Experiment**: experiments/exp_05_gtsrb_kdm_npc/
**Status**: Hoja de ruta técnica — sin código escrito todavía
**Prerrequisito**: `DESIGN.md` (este documento asume su §0-8 ya aprobados)

---

## 1. Arquitectura de datos

### 1.1 Mapeo carpeta-numérica ↔ nombre-semántico de clase (verificado, no supuesto)

GTSRB ya está descargado localmente (`data/gtsrb/gtsrb/GTSRB/{Training,
Final_Test}`, 703MB, 43 carpetas `00000`-`00042` con imágenes reales +
`GT-XXXXX.csv` de anotación por carpeta). `npc-dataset-utils` espera que las
imágenes vivan en `instances/<nombre-semántico>/<archivo>`, donde
`<nombre-semántico>` son las 43 claves de
`gtsrb.json["mappings"]` (ej. `"regulatory--maximum-speed-limit-20"`).

**Verificado leyendo el JSON directamente**: el orden de las 43 claves de
`mappings` coincide EXACTAMENTE con el índice oficial de clase de GTSRB
(0=speed-limit-20, 1=speed-limit-30, ..., 42=end-of-no-overtaking-by-heavy-
goods-vehicles) — es decir, la carpeta `0000N` corresponde a la N-ésima
clave del diccionario `mappings` en el orden en que aparece en el JSON. No
hace falta ninguna tabla de mapeo externa: se construye programáticamente
leyendo `gtsrb.json` una vez.

```python
# scripts/_build_class_mapping.py (nuevo, corre una sola vez, no en Kaggle)
import json
with open("external/npc-dataset-utils/configs/npc-dataset-utils/gtsrb.json") as f:
    cfg = json.load(f)
class_names = list(cfg["mappings"].keys())   # orden = ClassId oficial 0..42
# class_names[0] == "regulatory--maximum-speed-limit-20"  (carpeta "00000")
# class_names[42] == última clave                          (carpeta "00042")
```

**Verificación obligatoria antes de empaquetar**: confirmar con al menos 3
carpetas al azar que el atributo esperado (leído de
`GT-000NN.csv`, columna `ClassId`, y de `gtsrb.json["mappings"][class_names[N]]
["labels"]`) es consistente — ej. carpeta `00014` (índice 14) debe mapear a
`"regulatory--stop"` con `color=red`... si por algún motivo alguna versión
de GTSRB reordena las clases, esta verificación lo detecta antes de subir
un dataset con etiquetas cruzadas.

### 1.2 Empaquetado como dataset de Kaggle (mismo patrón que exp_01)

Reproducir el proceso ya ejecutado y documentado en
`exp_01_mnist_npc_repro/IMPLEMENTATION.md` (ver `data/kaggle_dataset_stage/`
como plantilla real ya usada):

1. Reorganizar `data/gtsrb/gtsrb/GTSRB/Training/0000N/*.ppm` →
   `gtsrb_processed/<class_names[N]>/*.ppm` (symlink o copia, usando el
   mapeo de §1.1). Usar el split ya materializado en el `Training/` oficial
   de GTSRB como pool para train+validate+test, o — más simple y
   consistente con `exp_01`/`exp_03` — dejar que `split.py` reconstruya
   train/validate/test desde `gtsrb_split.json.gz` (ya existe en
   `npc-dataset-utils/configs`) sobre el pool combinado de imágenes.
2. `MANIFEST.json` con `global_sha256` del zip (mismo patrón que
   `data/kaggle_dataset_stage/MANIFEST.json`).
3. `kaggle datasets create` (o `version` si el slug ya existe) — nombre
   sugerido: `bspenad10/gtsrb-npc`.
4. **Diferencia con MNIST-Addition**: NO hace falta generar el dataset desde
   cero con un script propio (`generate_mnist_addition.py` no tiene
   análogo aquí) — GTSRB es uno de los datasets *originales* soportados por
   `npc-dataset-utils` (su `gtsrb_split.json.gz` ya viene del repo oficial),
   así que el trabajo real es solo reorganizar carpetas + re-mapear nombres,
   no construir un dataset sintético nuevo.

### 1.3 Tronco compartido

**ResNet-34 preentrenado (`torchvision.models.resnet34(weights=
"IMAGENET1K_V1")`, `fc=Identity()`), idéntico a `build_shared_trunk()` en
`src/models/kdm_cascade.py:28-32`** — sin cambios. Ver `DESIGN.md §3` para
la justificación de por qué NO se cambia a ResNet-50/MobileNetV3 (aislar la
variable dataset de la variable arquitectura).

---

## 2. Adaptación de KDM (K=4 cabezas, no 2)

### 2.1 Generalizar `KDMCascade`

`src/models/kdm_cascade.py` está hardcodeado a 2 cabezas fijas
(`head1`/`head2`, `N_DIGIT_VALUES=10` compartido por ambas). Para GTSRB, las
4 cabezas tienen **cardinalidades distintas** (color=3, shape=4, symbol=26,
text=10) — el código debe generalizarse a una lista:

```python
ATTRIBUTE_CARDINALITIES = {"color": 3, "shape": 4, "symbol": 26, "text": 10}
N_CLASSES = 43

class KDMCascadeGTSRB(nn.Module):
    def __init__(self, n_comp_per_value: int, n_comp_final: int, sigma_head=1.0):
        super().__init__()
        self.trunk = build_shared_trunk()
        self.heads = nn.ModuleDict({
            name: KDMClassModel(RESNET_NECK_SIZE, card, nn.Identity(),
                                n_comp=n_comp_per_value * card, sigma=sigma_head)
            for name, card in ATTRIBUTE_CARDINALITIES.items()
        })
        self.kdm_final = KDMLayer(
            kernel=CosineKernelLayer(),
            dim_x=math.prod(ATTRIBUTE_CARDINALITIES.values()),  # 3*4*26*10=3120
            dim_y=N_CLASSES, n_comp=n_comp_final,
        )

    def forward(self, image):
        neck = self.trunk(image)
        p = {name: head(neck) for name, head in self.heads.items()}
        joint = cartesian_product(list(p.values()))   # ya soporta N>=2 factores
        rho_x = pure2dm(joint)
        p_class = dm2discrete(self.kdm_final(rho_x))
        return p, p_class
```

`cartesian_product` (`external/kdm/kdm/utils.py:159-179`) **ya generaliza a
N≥2 factores nativamente** (fold/reduce sobre la lista) — verificado leyendo
el código, no requiere ningún cambio.

### 2.2 Fix necesario: `n_comp_per_value` reemplaza `n_comp_head`

**Gotcha real, encontrado leyendo `stratified_idx`
(`kdm_cascade.py:138-148`)**: exige `n_comp % n_values == 0` exactamente
(divide en partes iguales por valor de atributo). Con 4 cabezas de
cardinalidad distinta, un único `n_comp_head` compartido (como en MNIST, que
funcionaba porque digit1 y digit2 tenían ambos cardinalidad 10) **no
generaliza** — el mínimo común múltiplo de {3,4,26,10} es 780, un número de
componentes por cabeza poco práctico.

**Fix**: expresar el hiperparámetro de la Fase A como `n_comp_per_value`
(componentes **por valor de atributo**, no total) — cada cabeza usa
`n_comp = n_comp_per_value * cardinalidad_de_esa_cabeza`. Esto es
exactamente lo que ya pasaba implícitamente en MNIST-Addition
(`n_comp_head=100 = 10 valores × 10 comps/valor`) — solo se hace explícito
para que generalice a cardinalidades heterogéneas. La aserción
`n_comp % n_values == 0` se sigue cumpliendo por construcción (`n_comp =
n_comp_per_value * n_values`), sin relajar la estratificación exacta.

Mismo razonamiento para `n_comp_final` sobre las 43 clases — debe ser
múltiplo exacto de 43 (ver `DESIGN.md §2/§5`: 172/430/645 = 4/10/15
componentes por clase).

### 2.3 Generalizar `init_components`

El bucle de `stratified_idx` + `init_kdm_layer` por cabeza
(`kdm_cascade.py:150-171`) se generaliza a un `for name, card in
ATTRIBUTE_CARDINALITIES.items(): ...` sobre el diccionario de cabezas, en
vez de las 2 líneas repetidas manualmente para `head1`/`head2`. La
inicialización del `kdm_final` usa `cartesian_product` sobre la lista de
one-hot verdaderos de las 4 cabezas (generalización directa del patrón de
2 factores ya existente en `kdm_cascade.py:163-166`).

---

## 3. Adaptación de NPC

**Sin cambios de código** — a diferencia de KDM, `npc-models` ya está escrito
de forma genérica sobre `config_dataset["attributes"]` (ver
`model.py::ResNet34MTL.__init__`, ya itera sobre attributes de cardinalidad
arbitraria) y sobre `header.config_pc["file_path_pc"]` (string, no
hardcodeado a MNIST). Los únicos cambios son de **configuración**, no de
código:

- `header.dataset_prefix = "gtsrb"` (en vez de patchear el archivo, como se
  hizo en los kernels de `exp_01`/`exp_03` — mismo patrón).
- Knowledge: `header.config_pc["file_path_pc"]` → `outputs/manual/gtsrb.spn.txt`
  (390 nodos, ya existe, no requiere Java/LearnSPN).
- Data: `outputs/learnspn/gtsrb.spn.txt` (359 nodos, ya existe).
- **No se necesita ninguna aproximación nueva para `multi_hot`** — a
  diferencia de CelebA/AwA2 (que si se abordaran, requerirían adaptar
  `computeConceptAccuracy`/la función de pérdida de atributos a
  multi-etiqueta), GTSRB es single-hot como MNIST-Addition.
- Riesgo de cómputo: ambos circuitos (359/390 nodos) son ~40-100× más chicos
  que el circuito Data de MNIST (15,410 nodos, que ya tardó ~15-20 min en un
  pase de eval en `exp_04`) — bajo riesgo de que el entrenamiento en Kaggle
  se vuelva impracticable por el lado del circuito.

---

## 4. Pipeline de ejecución de dos pasos

### Paso 1 — Fase A (barrido rápido)

Mismo patrón de plantilla + generador que `exp_03`
(`exp_03_mnist_kdm_sweep/scripts/_template_kernel.py` +
`_generate_kernel.py`) — **reusar el generador tal cual**, solo cambiar:
- Import de `KDMCascadeGTSRB` en vez de `KDMCascadeCartesian`.
- Placeholders de hiperparámetros: `n_comp_per_value` (10/15/20),
  `n_comp_final` (430/172/645), `lr_kdm` (3e-3/1e-3/3e-4), `sigma_mult`
  (1.0/0.5/2.0).
- 15 épocas, semilla única 42, batch 256 — igual que `exp_03`.
- 9 corridas un-eje-a-la-vez + 1 de confirmación = 10 corridas cortas.
- **Nuevo respecto a `exp_03`**: guardar `val_loss_history`/`val_accuracy_
  history` cada 5 épocas (no solo train) — requiere separar un mini-split de
  validación del train set (ya materializado por `split.py`/
  `gtsrb_split.json.gz`, que sí tiene partición `validate` propia, a
  diferencia de cómo se usó en `exp_03` donde no se evaluó validación
  durante el entrenamiento).

### Paso 2 — Fase B (confirmación + 5 semillas)

**Condicionada explícitamente** a que la Fase A confirme un ganador claro
sin interacción aditiva fuerte (mismo criterio de decisión que
`DESIGN.md §5`) — si la corrida de confirmación combinada no supera a la
mejor corrida individual, se usa esta última, igual que en `exp_03`.

- 60 épocas (mismo criterio de `exp_03`: convergencia rápida ya observada
  en el dataset hermano MNIST-Addition, a confirmar aquí con las curvas de
  validación de la Fase A antes de fijar el número de épocas).
- 5 semillas (42, 52, 62, 72, 82) — mismo protocolo.
- `git_commit.txt` por corrida (fix respecto a `exp_03`, que no lo generó —
  ver `DESIGN.md §7`).

---

## 5. Módulo de métricas unificado (extiende exp_04, no reescribe)

Los 4 scripts de `exp_04_mnist_evaluation/scripts/` se **copian y adaptan**
(no se reescriben desde cero):

| Script | Cambio necesario para GTSRB |
|---|---|
| `run_inference_kdm.py` | `KDMCascade`→`KDMCascadeGTSRB`, `N_CLASSES=19→43`, checkpoint path a `exp_05/results/.../model.pt` |
| `run_inference_npc.py` | `dataset_prefix="gtsrb"`, rutas de circuito §3, checkpoints de `exp_05` |
| `build_eval_plots.py` | `N_CLASSES=19→43`; **agregar `average_precision_score` como métrica primaria** (no solo secundaria) dado el desbalance real — ver `DESIGN.md §6` |

**Actualización (2026-07-15): `build_interpretability_kdm.py`/
`build_interpretability_npc.py` quedaron reemplazados** por
`src/metrics/interpretability_suite.py` — módulo compartido (no vive dentro
de ningún `exp_NN`, para poder importarse igual desde `exp_04` y `exp_05`),
escrito explícitamente agnóstico al dataset:

- `KDMExplainer`: recibe `attribute_names`/`attribute_cardinalities` como
  parámetros (no hardcodeados a 2 dígitos) — para GTSRB, pasar
  `["color","shape","symbol","text"]` / `[3,4,26,10]` y un `head_accessor`
  que resuelva `model.heads[name]` según cómo se nombre `KDMCascadeGTSRB`
  (§2). `decode_final_component` ya generaliza a K≥2 atributos (reshape a
  grid K-dimensional + argmax marginal por eje, no el truco de 2 factores de
  `exp_04`).
- `NPCExplainer`: recibe `pc_joint`/`pc_marginal`/`pc_settings_joint` ya
  construidos con la config de GTSRB (§3) — `mpe_query`/`counterfactual` no
  cambian, ya que envuelven `test_npc.py::computeNPCOutput`/`findMPE`/
  `findCE`, que ya son genéricos sobre cardinalidad de atributos.
- `select_comparison_instances`/`build_comparison_panel`: sin cambios — solo
  reciben arrays de predicción/verdad y diccionarios de metadatos.
- **Trabajo específico de GTSRB, no cubierto por el módulo**: el lookup de
  "imagen de entrenamiento más cercana" (`nearest_training_images_for_head`)
  necesita su propio `prepare_train_sample.py` para GTSRB (mismo patrón que
  `exp_04`, adaptado a `gtsrb_split.json.gz`); construir un
  `build_comparison_report.py` propio de `exp_05` que instancie
  `KDMExplainer`/`NPCExplainer` con la configuración de GTSRB (igual que
  `exp_04_mnist_evaluation/scripts/build_comparison_report.py`, que sirve de
  plantilla directa).

**Nuevo, no existía en `exp_04`**:

- `build_calibration_plots.py` — diagrama de fiabilidad (reliability
  diagram) por modelo: `sklearn.calibration.calibration_curve(y_true_top1,
  p_max, n_bins=10, strategy="quantile")` (quantile en vez de uniform,
  porque con 43 clases desbalanceadas los bins uniformes quedan vacíos en
  los extremos) + Expected Calibration Error (ECE) como escalar resumen.
- Extensión de `build_training_curves.py`: ahora sí existen
  `val_loss_history`/`val_accuracy_history` reales por época (guardados
  desde la Fase A, ver §4) — graficar train vs. validación superpuestos,
  cerrando la limitación que `exp_04` tuvo que documentar como vacío para
  KDM en MNIST-Addition.

**Checklist de guardado por corrida (desde la primera corrida de Fase A, no
solo en Fase B)**:
- [ ] `metrics.json`: hiperparámetros, `train_loss_history`,
  `val_loss_history`, `val_accuracy_history` (cada 5 épocas), métricas
  finales de test.
- [ ] `predictions.npz`: probabilidades por muestra (no solo agregados) —
  para poder correr `build_eval_plots.py`/`build_calibration_plots.py` sin
  tener que re-inferir después, a diferencia de `exp_03`→`exp_04`.
- [ ] `git_commit.txt`.
- [ ] `checkpoints/model.pt` (gitignored, ver patrón ya establecido en
  `.gitignore`).

---

## 6. Orden de ejecución recomendado

1. `scripts/_build_class_mapping.py` (§1.1) + verificación manual de 3 clases.
2. Empaquetar y subir el dataset de Kaggle (§1.2).
3. Escribir `src/models/kdm_cascade_gtsrb.py` (§2) — verificar con un
   forward pass de prueba local (batch pequeño, CPU) antes de subir a
   Kaggle, igual que se hizo para MNIST-Addition en `exp_02`.
4. Adaptar plantilla+generador de kernels de Fase A (§4, Paso 1).
5. Correr las 10 corridas cortas de Fase A → elegir ganador (§ `DESIGN.md §5`).
6. Si Fase A confirma un ganador: Fase B (5 semillas).
7. Copiar y adaptar los scripts de métricas de `exp_04` (§5) → informe final.
