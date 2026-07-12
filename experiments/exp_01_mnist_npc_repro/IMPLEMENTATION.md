# Implementation Plan — exp_01_mnist_npc_repro

**Estado**: etapa 1 lista para lanzar; etapas 2-3 pendientes de la etapa 1.

## Mapa condición → script/config

| Condición | Script | Dónde corre | Estado |
|---|---|---|---|
| npc-neural_seed42 (etapa 1) | `scripts/kernel_stage1_neural.py` + `scripts/kernel-metadata.json` | Kaggle GPU | listo para push |
| circuit-knowledge / circuit-data (etapa 2) | pendiente (usa `learnspn` manual.py / learnspn.bash + `train_pc.py`) | Kaggle GPU | pendiente |
| npc-knowledge_seed42 / npc-data_seed42 (etapa 3) | pendiente (`train_npc.py`) | Kaggle GPU | pendiente |

## Datos

- Dataset congelado: Kaggle Dataset privado `bspenad10/mnist-addition-npc`
  (mnist_addition_processed.zip 43.6 MB + mnist.json + mnist_split.json.gz +
  MANIFEST.json). Staging local: `data/kaggle_dataset_stage/`.
- Hash global verificado en el kernel contra
  `4d640365ca4b505d14ae79a83dcdc799d546a121a676e0a26a01c42fb2b9e07d` (aborta si
  no coincide).
- Generador: `src/data/generate_mnist_addition.py` (lee mappings oficiales;
  0 discrepancias contra etiquetas reales de MNIST — verificado 2026-07-12).

## Decisiones de construcción (y por qué)

1. **Generación de D̄ desde mappings oficiales, no desde el shuffle de
   `mnist.py`**: el shuffle depende del orden de `os.listdir` (NTFS ≠ ext4);
   los mappings publicados (`mnist.json`) codifican el emparejamiento exacto
   de los autores. Equivalencia de píxeles documentada en el generador.
2. **Único patch a los repos oficiales**: `dataset_prefix = "awa2"` → `"mnist"`
   en ambos `header.py` — es el mecanismo de configuración propio del repo
   (no hay CLI para esto), no un cambio funcional. Verificado con assert.
3. **Splits**: `split.py` con `split_load=True` carga el split oficial
   (28000/3500/3500) y materializa symlinks (Linux-only → en el kernel).
4. **Métricas de etapa 1 capturadas re-ejecutando `test_neural.py`** con el
   run name descubierto del checkpoint `.best.zip` (el run name incluye
   timestamp+hostname, no es predecible). La métrica es byte-a-byte la del
   paper (TV media + accuracy top-n_k de conceptos).
5. **Entorno pinneado** a `npc-models/requirements.txt` (torch 2.1.2+cu121,
   numpy<2, torch_explain 1.5.1, natsort, sklearn, wandb 0.16.1 en modo
   disabled). PyQt5 omitido (solo lo usa la GUI `interpret.py`).
6. **Limpieza al final del kernel**: la jerarquía `npc/` (35k imágenes) se
   borra para que el output persistido sea solo `results/<condición>/`.

## Comandos (desde la raíz del repo)

```bash
# Subir/actualizar dataset congelado (privado por defecto en CLI 2.2.3)
python -m kaggle datasets create -p data/kaggle_dataset_stage
# (versiones posteriores)
python -m kaggle datasets version -p data/kaggle_dataset_stage -m "mensaje"

# Lanzar etapa 1
python -m kaggle kernels push -p experiments/exp_01_mnist_npc_repro/scripts

# Monitorear / traer resultados
python -m kaggle kernels status bspenad10/exp01-npc-mnist-stage1-seed42
python -m kaggle kernels output bspenad10/exp01-npc-mnist-stage1-seed42 \
    -p experiments/exp_01_mnist_npc_repro/results/_kaggle_output
```

## Criterio de aceptación etapa 1

TV media ≈ 0.0058 y accuracy media de conceptos ≈ 98.99 % (referencia ABM,
Tabla 3) — orden de magnitud; el gate duro es sobre el modelo completo
(Tabla 2: 99.171/99.189 ± std) tras etapas 2-3.

## Desviaciones conocidas respecto al paper

- GPU: Kaggle P100/T4 en lugar del hardware original (no reportado en el
  paper) — wall-clock no comparable con el de los autores, sí entre nuestras
  condiciones.
- `run_mode = "disabled"` para wandb (los autores usaban wandb online); las
  métricas se capturan vía `test_neural` re-ejecutado y `metrics.json`.
