# Experiment Report: Evaluación extendida + interpretabilidad nativa (KDM vs. NPC) en MNIST-Addition

**Experiment**: experiments/exp_04_mnist_evaluation/
**Project**: Tesis_KDM_NPC
**Report date**: 2026-07-15
**Author**: Brayan Steven Peña Delgadillo
**Status**: Complete

---

## 1. Summary

`exp_01` y `exp_03` ya establecieron accuracy/TV/verosimilitud como escalares
agregados, pero nunca guardaron predicciones por muestra — así que no existía
matriz de confusión, ROC/AUC ni precision-recall para ningún modelo. Este
experimento corre inferencia local (CPU, sin Kaggle) reusando los checkpoints
ya entrenados de KDM (`exp_03/final-seed42`) y NPC (`exp_01`, ambas variantes,
seed 42) sobre el split de test congelado, y agrega dos cosas más pedidas
explícitamente: gráficas de diagnóstico de entrenamiento (para ver
estabilidad/overfitting) y una comparación de los mecanismos de
interpretabilidad **nativos** de cada arquitectura — no herramientas
genéricas de ML tabular (PDP, feature-importance) que no calzan sobre una CNN
compartida, sino los mecanismos que la propia estructura de cada modelo
provee: atribución por componente en KDM, e inferencia MAP tratable (MPE) +
explicación contrastiva (CE) en NPC, esta última ya implementada por los
autores originales en `npc-models/test_npc.py`.

---

## 2. Setup

- **Modelos**: KDM (`exp_03/results/final-seed42`, variante Cartesian),
  NPC Knowledge y NPC Data (`exp_01/results/npc-{knowledge,data}_seed42`) —
  una semilla representativa por modelo (42), la misma usada como referencia
  principal en los experimentos originales.
- **Split**: test congelado de MNIST-Addition (3500 imágenes), materializado
  localmente desde `mnist_split.json.gz` + imágenes ya procesadas — mismo
  split exacto usado en `exp_01`/`exp_03` (no se regenera nada).
- **Cómputo**: 100% local, CPU. No se reentrena ningún modelo.
- **Fidelidad verificada**: la inferencia local reproduce EXACTAMENTE las
  métricas ya publicadas en `exp_01`/`exp_03` (ver tabla abajo) — confirma que
  el pipeline de carga de checkpoints + dataset es correcto antes de confiar
  en las métricas nuevas (confusión/ROC/PR/interpretabilidad).

| Modelo | Accuracy (exp_01/exp_03) | Accuracy (recomputada aquí) |
|---|---|---|
| KDM | 99.171% (`exp_03` final-seed42) | 99.171% ✅ |
| NPC (Knowledge) | 99.20% (`exp_01`) | 99.20% ✅ |
| NPC (Data) | 99.00% (`exp_01`) | 99.00% ✅ |

---

## 3. Evaluación estándar (matriz de confusión, ROC/AUC, precision-recall)

`reports/figures/eval_confusion_matrices.png`,
`reports/figures/eval_roc_macro.png`, `reports/figures/eval_precision_recall_macro.png`

| Modelo | Accuracy | ROC-AUC (macro, one-vs-rest, 19 clases) | Average Precision (macro) |
|---|---|---|---|
| KDM | 0.9917 | 0.99966 | 0.9958 |
| NPC (Knowledge) | 0.9920 | 0.99973 | 0.9983 |
| NPC (Data) | 0.9900 | 0.99966 | 0.9970 |

Los tres modelos son prácticamente indistinguibles en ROC-AUC (todos
≥0.9996) — con ~99% de accuracy sobre 19 clases balanceadas, el AUC
one-vs-rest satura y deja de discriminar entre modelos; el **Average
Precision** (más sensible en este régimen de alta accuracy) sí ordena
igual que la accuracy: NPC(Knowledge) > NPC(Data) > KDM, aunque las
diferencias son pequeñas (0.9958–0.9983). Las matrices de confusión de los
tres modelos son visualmente casi diagonales puras — los pocos errores no se
concentran en una franja de "sumas adyacentes" obvia para ninguno de los tres.

---

## 4. Curvas de entrenamiento (diagnóstico de estabilidad)

`reports/figures/training_kdm_loss_vs_epoch.png`,
`reports/figures/training_npc_stage1_val_accuracy_vs_epoch.png`

- **KDM** (exp_03, Fase B, 60 épocas × 5 semillas): pérdida de train
  desciende establemente en escala log; la semilla 42 muestra un salto
  transitorio de 0.0025→0.0102 alrededor de la época 44 antes de
  re-converger — inestabilidad puntual, no divergencia.
- **NPC stage 1** (ResNet34MTL, seed 42, 150 épocas, compartido por ambas
  variantes): accuracy de validación sube de 0.949 a >0.99 en ~10 épocas y
  luego oscila en una banda estrecha (0.989–0.991) durante las 140 épocas
  restantes, sin degradación — **no hay señal de overfitting** en esta etapa
  (el mejor punto, época 49, está a solo 0.002 del promedio de las últimas 50
  épocas).
- **Limitación documentada, no resuelta**: no existe curva de *validación*
  por época para KDM (solo se registró train), ni ninguna curva por época
  para las etapas 2/3 de NPC (circuito CCCP + optimización conjunta PGD) —
  esos runs originales no guardaron logs con esa granularidad. No se
  fabricó ningún dato para rellenar este vacío.

---

## 5. Interpretabilidad nativa

### 5.1 KDM — atribución exacta por componente

`reports/figures/interp_kdm_component_attribution_examples.png`,
`reports/figures/interp_kdm_attribution_entropy.png`

La predicción final de KDM es `Σ c_w_i · kernel(x, c_x_i)²` sobre 190
prototipos aprendidos (Ec. 12) — una descomposición **exacta**, no una
aproximación post-hoc. Para 5 muestras de ejemplo (3 correctas, 2
incorrectas), la atribución es marcadamente dispersa (sparse): las
predicciones correctas concentran >90% del peso en 1-2 componentes cuyo par
de dígitos más cercano (proyección de `c_x` sobre la base one-hot del
producto cartesiano) coincide con la verdad. La muestra 74 (real=1,
predicho=4) muestra el modelo asignando >30% de peso, con alta confianza, al
componente "c4X: (4,0)→4" — es decir, el error es un caso claro de
confusión visual dígito-a-dígito, visible directamente en la atribución, sin
necesitar ningún método externo.

**Hallazgo cuantitativo**: la entropía de la distribución de atribución es
sistemáticamente más alta en predicciones incorrectas (media 1.263 nats)
que en correctas (media 0.986 nats) — KDM "sabe cuándo no sabe" de forma
nativa: cuando el modelo reparte su decisión entre más prototipos en
conflicto, es más probable que se equivoque.

### 5.2 NPC — inferencia MAP tratable (MPE) y explicación contrastiva (CE)

`reports/figures/interp_npc_mpe_ce_comparison.png`

Usando los mecanismos ya implementados por los autores de NPC
(`npc-models/test_npc.py`, activados con `header.npc_interpret=True`):

| Métrica | Knowledge | Data |
|---|---|---|
| MPE alignment rate | **1.000** | **1.000** |
| CE correction rate | 0.750 | 0.857 |

- **MPE alignment rate = 1.0 en ambas variantes**: para el 100% de las
  predicciones correctas, la explicación MAP exacta del circuito (la
  asignación de dígitos que maximiza `matrix_pc × matrix_neural`) coincide
  exactamente con los dígitos reales — cuando NPC acierta la suma, lo hace
  por el camino correcto (reconociendo bien ambos dígitos), no por
  coincidencia aritmética entre dígitos mal reconocidos.
- **CE correction rate**: de las predicciones incorrectas, 75% (Knowledge) y
  85.7% (Data) se pueden corregir a la clase correcta con un ajuste mínimo
  (por gradiente) de las probabilidades de atributo — sugiere que la mayoría
  de los errores de NPC son "casi aciertos" del reconocedor de atributos, no
  fallas del circuito en sí. Data tiene una tasa de corrección más alta,
  consistente con su ligera desventaja en concept accuracy (0.9898 vs.
  0.9919) — sus errores de atributo tienden a ser más marginales/corregibles.
- **Estructura del circuito** (interpretabilidad estructural, inferencia
  tratable por construcción): Knowledge tiene 140 nodos (1 suma, 100
  producto, 39 hoja, profundidad 2 — el circuito manual, deliberadamente
  compacto); Data tiene 15,410 nodos (11,512 suma, 3,859 producto, 39 hoja,
  profundidad 5 — la estructura aprendida por LearnSPN, mucho más compleja
  para una ganancia de accuracy menor).

### 5.3 Comparación conceptual

KDM y NPC son interpretables **por construcción**, cada uno a su manera: KDM
vía combinación explícita y dispersa de prototipos (analogía con k-NN /
memoria); NPC vía inferencia probabilística exacta sobre una estructura de
circuito (analogía con razonamiento simbólico composicional). Ninguno
necesita SHAP/LIME/PDP para explicar su propia capa de decisión — esas
herramientas solo aplicarían, en el mejor de los casos, al tronco
convolucional compartido (ResNet-34), que es idéntico en ambas familias de
modelos y por tanto no es lo que las distingue.

---

## 6. Reproducibilidad

| Item | Status |
|---|---|
| Reentrenamiento | ❌ No aplica — reusa checkpoints ya versionados de `exp_01`/`exp_03` |
| Fidelidad verificada | ✅ Accuracy recomputada coincide exactamente con los runs originales |
| Split de test | ✅ Mismo split congelado, materializado con `scripts/prepare_test_split.py` |
| Scripts versionados | ✅ `scripts/{prepare_test_split,run_inference_kdm,run_inference_npc,build_eval_plots,build_interpretability_kdm,build_interpretability_npc,build_training_curves}.py` |
| Ajustes de entorno (Windows) | Documentados inline en el código, no aplicados a `external/` vendorizado (separador de rutas, `map_location` de `torch.load`) |
| Predicciones por muestra | ✅ `results/{kdm_final-seed42,npc_knowledge_seed42,npc_data_seed42}/predictions.npz` |

---

## 7. Conclusiones y próximos pasos

**Lo que este experimento estableció:**
- Los tres modelos son casi indistinguibles en ROC-AUC/AP a este nivel de
  accuracy (>99%) — la comparación fina ya la había resuelto `exp_03` vía
  accuracy + t-test.
- Ninguno de los dos modelos muestra señales de overfitting en las curvas
  disponibles (KDM: pérdida de train estable; NPC stage 1: validación estable
  desde la época ~10).
- **Ambos modelos tienen mecanismos de interpretabilidad nativos,
  cualitativamente distintos y genuinamente informativos**: la entropía de
  atribución de KDM predice el error: (1.263 vs 0.986 nats); el 100% de
  MPE-alignment de NPC confirma que sus aciertos son por razonamiento
  correcto, no casualidad aritmética.

**Lo que queda incierto:**
- Si la ventaja de CE-correction-rate de Data sobre Knowledge (85.7% vs 75%)
  se sostiene en datasets con atributos más ruidosos.
- Si la brecha estructural del circuito Data (15,410 nodos vs 140 de
  Knowledge, para una accuracy *menor*) es un patrón que se repite en otros
  datasets NPC — de ser así, es un argumento fuerte a favor de circuitos
  Knowledge-driven cuando el conocimiento de dominio está disponible.

**Recomendación**: usar `lr_kdm=3e-3, n_comp_head=100, n_comp_final=190,
sigma_mult=1.0` (config de `exp_03`) como punto de partida para `exp_05` en
el siguiente dataset, y replicar este mismo protocolo de evaluación extendida
(confusión/ROC/PR + interpretabilidad nativa) una vez haya checkpoints
entrenados allí.
