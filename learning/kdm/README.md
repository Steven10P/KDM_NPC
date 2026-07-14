# Ruta de aprendizaje: Kernel Density Matrices (KDM)

Curso propio, aparte del trabajo de implementación (`experiments/`), para
entender KDM en profundidad: la matemática del paper, el código exacto de
`external/kdm/kdm/`, y cómo se aplica a las distintas tareas que resuelve.

Cada módulo es un notebook que combina tres cosas para el mismo tema:
1. **Matemática** — citando la sección exacta de
   [`docs/s42484-025-00299-9.pdf`](../../docs/s42484-025-00299-9.pdf)
   (González, Ramos-Pollán & Gallego, *Quantum Machine Intelligence*, 2025).
2. **Código** — citando archivo:línea exactos de `external/kdm/kdm/`, no
   reconstruido de memoria.
3. **Ejercicio práctico** — sobre un dataset guía consistente entre módulos
   (**MNIST dígitos**, salvo el módulo de regresión).

No reinventa lo que ya existe: para "cómo usar" cada modelo ya hay templates
completos y ejecutables en la skill `.claude/skills/kdm/references/*.md` y en
`external/kdm/examples/*.ipynb`. Este curso agrega lo que faltaba: conectar la
matemática con el código línea a línea, y un hilo narrativo único a través de
las tareas.

## Dataset guía

- **MNIST (dígitos 0-9)** — clasificación, estimación de densidad, densidades
  conjuntas, muestreo/generación, incertidumbre/OOD. Ya está descargado
  localmente en `data/mnist/` (no hace falta re-descargar). Conecta
  naturalmente con MNIST-Addition, el dataset real de la tesis.
- **Dataset tabular pequeño de sklearn** (diabetes o California housing) —
  solo para el módulo de regresión, porque MNIST no tiene un target continuo
  natural.

## Módulos

| # | Notebook | Tema | Estado |
|---|---|---|---|
| 00 | [`00_fundamentos_matrices_densidad.ipynb`](notebooks/00_fundamentos_matrices_densidad.ipynb) | Matrices de densidad, la tripleta (C,p,k), representación tensorial `(bs,n,d+1)`, `pure2dm`/`comp2dm`/`dm2comp`/`samples2dm`/`dm2discrete` | ✅ listo |
| 01 | [`01_kdm_layer_y_clasificacion.ipynb`](notebooks/01_kdm_layer_y_clasificacion.ipynb) | `KDMLayer` (Ec. 12 línea por línea), kernels RBF/coseno, `init_kdm_layer`, clasificación con MNIST | ✅ listo |
| 02 | [`02_estimacion_de_densidad.ipynb`](notebooks/02_estimacion_de_densidad.ipynb) | `KDMProjLayer`, `KDMDenEstModel` (= GMM con kernel RBF), densidad por clase de dígito (PCA 2D, visualizada) | ✅ listo |
| 03 | `03_densidades_conjuntas.ipynb` | `CrossProductKernelLayer`, `KDMJointDenEstModel`, `cartesian_product` — imagen+etiqueta conjunta (prefigura el mecanismo central de la comparación KDM-vs-NPC) | ⏳ pendiente |
| 04 | `04_regresion.ipynb` | `KDMRegressModel`, `dm_rbf_expectation`/`dm_rbf_variance` — dataset tabular aparte | ⏳ pendiente |
| 05 | `05_muestreo_y_generacion.ipynb` | Muestreo desde una KDM, patrón generador (c_x↔c_y invertido), generación condicional de dígitos | ⏳ pendiente |
| 06 | `06_incertidumbre_y_ood.ipynb` | Entropía, calibración, `log_marginal` para detección OOD (MNIST rotado) | ⏳ pendiente |
| 07 | `07_kdm_memoria_y_explicabilidad.ipynb` *(opcional)* | KDM basado en memoria, `predict_explain` — conecta con la Opción D de la comparación con NPC | ⏳ pendiente |

## Cómo usar este curso

Cada notebook es autocontenido — se puede correr de punta a punta en el
entorno `tesis_kdm_npc` (kdm-torch ya instalado ahí). Los módulos se
construyen de a uno por sesión, no todos de una vez.
