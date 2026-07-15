# Experiment Design: Evaluación extendida + interpretabilidad nativa (KDM vs. NPC) en MNIST-Addition

**Experiment**: experiments/exp_04_mnist_evaluation/
**Project**: Tesis_KDM_NPC
**Date**: 2026-07-15
**Author**: Brayan Steven Peña Delgadillo
**Status**: ✅ Complete (2026-07-15)

---

## 1. Motivación

`exp_01` (NPC) y `exp_03` (KDM) ya establecieron accuracy end-to-end, TV
distance y verosimilitud del circuito como métricas de comparación, pero solo
como escalares agregados en `metrics.json` — nunca se guardaron las
predicciones/probabilidades por muestra del test set, así que no existía
matriz de confusión, curva ROC/AUC, ni precision-recall para ninguno de los
dos modelos (`exp_01/reports/figures/` está vacío; `exp_02` no tiene
`reports/` en absoluto; `exp_03` solo tiene accuracy-comparison + loss-curves
+ summary-bars).

Además, el usuario pidió explícitamente explorar si KDM y NPC tienen
mecanismos de interpretabilidad **propios** (más allá de forzar
feature-importance/PDP/SHAP genéricos de modelos tabulares, que no calzan
bien sobre una CNN compartida). La respuesta, verificada leyendo el código
fuente exacto de ambos, es que **sí los tienen, y son de naturaleza muy
distinta**:

- **KDM**: la predicción es una combinación lineal EXACTA de un número finito
  de prototipos aprendidos (Ec. 12, `kdm/layers/kdm_layer.py::forward`) — se
  puede leer directamente qué componentes dominaron una predicción dada, sin
  aproximar nada.
- **NPC**: el circuito probabilístico ya trae, en el propio repo de los
  autores (`npc-models/test_npc.py`), dos mecanismos activables con
  `header.npc_interpret = True`: **MPE** (Most Probable Explanation — la
  asignación exacta de atributos que maximiza la probabilidad conjunta,
  inferencia MAP tratable) y **CE** (Contrastive/Counterfactual Explanation
  — ajuste por gradiente de las probabilidades de atributo hasta cambiar la
  predicción, para las muestras mal clasificadas).

Este experimento no reentrena nada — reusa los checkpoints ya entrenados de
`exp_01` (NPC, ambas variantes, seed 42) y `exp_03` (KDM, `final-seed42`) y
corre inferencia local (CPU) sobre el split de test congelado.

## 2. Alcance

- **Modelos**: KDM (`exp_03/results/final-seed42`), NPC Knowledge y NPC Data
  (`exp_01/results/npc-{knowledge,data}_seed42`) — una semilla representativa
  por modelo, la misma ya usada como referencia principal en `exp_01`/`exp_03`.
- **Split**: test congelado de MNIST-Addition (3500 imágenes), materializado
  localmente desde `mnist_split.json.gz` + las imágenes ya procesadas en
  `data/npc/datasets/mnist/instances/processed/` — no se re-genera el split,
  solo se extrae el subconjunto "test" (ver `scripts/prepare_test_split.py`).
- **Cómputo**: 100% local, CPU (`conda env tesis_kdm_npc`) — no requiere
  Kaggle. Verificado viable: torch+kdm-torch instalados, npc-models clonado,
  imágenes y los 5 checkpoints necesarios ya en disco.

## 3. Qué se produce

1. **Evaluación estándar** (`build_eval_plots.py`): matriz de confusión
   (19×19, normalizada por fila), curva ROC one-vs-rest macro-promediada +
   AUC, curva precision-recall macro-promediada + AP — los 3 modelos
   superpuestos.
2. **Curvas de entrenamiento** (`build_training_curves.py`): pérdida de train
   por época de KDM (ya existía en `metrics.json`, 60 épocas × 5 semillas) +
   accuracy de validación por época de NPC stage 1 (real, parseada del log
   crudo de Kaggle que sí quedó guardado, 150 épocas). **Limitación
   documentada**: no existe curva train-vs-validación para KDM (nunca se
   registró validación durante el entrenamiento), ni ninguna curva por época
   para las etapas 2/3 de NPC (circuito CCCP + optimización conjunta PGD) —
   no se inventa lo que no se registró en los runs originales.
3. **Interpretabilidad nativa de KDM** (`build_interpretability_kdm.py`):
   atribución por componente (Ec. 12) para muestras de ejemplo, etiquetada
   con el par de dígitos más cercano de cada uno de los 190 prototipos
   aprendidos; diagnóstico de entropía de la atribución (correctas vs.
   incorrectas) como señal nativa de incertidumbre.
4. **Interpretabilidad nativa de NPC** (`build_interpretability_npc.py`):
   MPE alignment rate y CE correction rate (Knowledge vs. Data), tabla
   cualitativa de ejemplos (verdad / predicción / explicación MPE /
   ¿coincide?), estadísticas estructurales del circuito (nodos suma/producto/
   hoja, profundidad).

## 4. Reproducibilidad

- No se reentrena ningún modelo — se reusan checkpoints ya versionados
  (`exp_01`, `exp_03`).
- Todos los scripts son deterministas dado el checkpoint + split (no hay
  aleatoriedad en inferencia).
- Ajustes necesarios para correr localmente en Windows (documentados inline
  en el código, no aplicados a los repos vendorizados en `external/`):
  separador de rutas (`os.path.join` produce `\` en Windows, las claves de
  `mnist.json["mappings"]` usan `/`), y `torch.load` sin `map_location`
  (los checkpoints se guardaron desde CUDA en Kaggle).

## 5. Resultados

Ver `reports/summary.md` para el informe completo. Resumen:

- Fidelidad verificada: la inferencia local reproduce EXACTAMENTE las
  métricas ya publicadas (KDM 99.171%, NPC-Knowledge 99.20%, NPC-Data 99.00%).
- ROC-AUC macro (19 clases) casi idéntico entre los tres modelos (≥0.9996);
  Average Precision sí ordena igual que accuracy (Knowledge > Data > KDM).
- KDM: entropía de atribución por componente más alta en errores (1.263 nats)
  que en aciertos (0.986 nats) — señal nativa de incertidumbre.
- NPC: MPE alignment rate = 1.0 en ambas variantes (los aciertos son por
  razonamiento correcto); CE correction rate 0.75 (Knowledge) / 0.857 (Data).
- Circuito Data: 15,410 nodos (profundidad 5) vs. 140 de Knowledge
  (profundidad 2) para una accuracy ligeramente menor — el conocimiento de
  dominio produce un circuito mucho más compacto.
