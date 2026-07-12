# Implementation Plan — exp_01_mnist_npc_repro

**Estado**: etapa 1 corriendo en Kaggle (v2, fix PyPI); kernels de etapas 2+3
(Knowledge y Data) escritos y listos, en espera de que termine la etapa 1
(ambos se enlazan a su output vía `kernel_sources`).

## Mapa condición → script/config

| Condición | Carpeta del kernel | Dónde corre | Estado |
|---|---|---|---|
| npc-neural_seed42 (etapa 1) | `scripts/stage1/` | Kaggle GPU | 🟢 corriendo (v2) |
| circuit-knowledge (etapa 2) + npc-knowledge_seed42 (etapa 3) | `scripts/stage2and3_knowledge/` | Kaggle GPU | listo, espera etapa 1 |
| circuit-data (etapa 2) + npc-data_seed42 (etapa 3) | `scripts/stage2and3_data/` | Kaggle GPU | listo, espera etapa 1 |

**Hallazgo clave (evita correr Java/LearnSPN):** el repo oficial `learnspn`
trae **precomputados** tanto `outputs/learnspn/mnist.spn.txt` (estructura
Data-driven, 30,743 líneas) como `outputs/manual/mnist.spn.txt` (circuito
Knowledge, pesos = frecuencia empírica de cada regla, ya final — paper
Proposition 1 prueba que el nodo raíz ya es la distribución conjunta
empírica, sin necesitar CCCP). Por eso ningún kernel de este exp corre Java;
solo se clona `learnspn` para tomar esos dos archivos ya generados.

**Encadenamiento entre kernels:** los kernels de etapa 2+3 declaran
`kernel_sources: ["bspenad10/exp01-npc-mnist-stage1-seed42"]` en su
`kernel-metadata.json`, lo que monta el output de la etapa 1 (el checkpoint
`.best.zip` del reconocedor) en `/kaggle/input/` sin subirlo a mano.

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
7. **Circuito Knowledge sin CCCP**: `header.py`'s `config_pc["file_path_pc"]`
   se parchea de `outputs/learnspn/` a `outputs/manual/` — el circuito ya
   tiene pesos finales (frecuencia empírica, Proposition 1 del paper), así
   que solo se evalúa con `test_pc.py` (sin `train_pc.py`). El circuito Data
   sí corre `train_pc.py` (CCCP) partiendo de la estructura LearnSPN ya
   incluida en el repo.
8. **Métricas de etapas 2-3 capturadas directamente del stdout de
   `train_pc.py`/`train_npc.py`** (ambos ya llaman a su propia evaluación
   final internamente) vía `subprocess.run(capture_output=True)` + regex —
   más simple que el patrón de re-ejecución usado en etapa 1, porque acá no
   hace falta descubrir un run-name previo.
9. **Encadenamiento de checkpoints entre kernels**: el `.best.zip` del
   reconocedor (etapa 1) se localiza con un glob sobre
   `/kaggle/input/*/npc-neural_seed*/*.best.zip` (el mount de
   `kernel_sources` no garantiza un nombre de carpeta 100% predecible) y se
   copia al `checkpoint_dir` que `utility.loadCheckpoint` espera —
   `train_npc.py -w <nombre_de_archivo>` solo acepta el nombre base, no una
   ruta absoluta.

## Comandos (desde la raíz del repo)

```bash
# Subir/actualizar dataset congelado (privado por defecto en CLI 2.2.3)
python -m kaggle datasets create -p data/kaggle_dataset_stage
# (versiones posteriores)
python -m kaggle datasets version -p data/kaggle_dataset_stage -m "mensaje"

# Lanzar etapa 1
python -m kaggle kernels push -p experiments/exp_01_mnist_npc_repro/scripts/stage1

# Lanzar etapas 2+3 (SOLO después de que la etapa 1 esté en estado COMPLETE —
# kernel_sources necesita el output ya generado, no solo el kernel existente)
python -m kaggle kernels push -p experiments/exp_01_mnist_npc_repro/scripts/stage2and3_knowledge
python -m kaggle kernels push -p experiments/exp_01_mnist_npc_repro/scripts/stage2and3_data

# Monitorear / traer resultados (repetir por cada slug de kernel)
python -m kaggle kernels status bspenad10/exp01-npc-mnist-stage1-seed42
python -m kaggle kernels output bspenad10/exp01-npc-mnist-stage1-seed42 \
    -p experiments/exp_01_mnist_npc_repro/results/_kaggle_output_stage1
python -m kaggle kernels output bspenad10/exp01-npc-mnist-stage23-knowledge-seed42 \
    -p experiments/exp_01_mnist_npc_repro/results/_kaggle_output_stage23_knowledge
python -m kaggle kernels output bspenad10/exp01-npc-mnist-stage23-data-seed42 \
    -p experiments/exp_01_mnist_npc_repro/results/_kaggle_output_stage23_data
```

## Criterio de aceptación etapa 1

TV media ≈ 0.0058 y accuracy media de conceptos ≈ 98.99 % (referencia ABM,
Tabla 3) — orden de magnitud; el gate duro es sobre el modelo completo
(Tabla 2: 99.171/99.189 ± std) tras etapas 2-3.

## Bloqueantes resueltos (2026-07-12)

1. **Sin acceso a internet en los kernels** (intentos v1-v2). Fallaba con
   `Temporary failure in name resolution` — primero solo contra
   `download.pytorch.org` (parecía bloqueo de dominio), pero el intento v2
   (ya sin ese dominio, usando PyPI puro) también falló resolviendo
   `pypi.org`. Causa: verificación telefónica de la cuenta pendiente (Kaggle
   desactiva la red en silencio sin ella, pese a `enable_internet: true`).
   **Resuelto**: usuario verificó su teléfono; v3 sí resolvió DNS.
2. **`torch==2.1.2` no tiene wheels para Python 3.12** (intento v3) — la
   imagen de Kaggle corre Python 3.12; el pin exacto de `npc-models` es de
   2023, anterior al soporte de 3.12 en PyTorch (mínimo disponible: 2.2.0).
   **Resuelto**: se sube el pin a **torch==2.2.2 + torchvision==0.17.2** en
   los tres kernels (misma serie 2.x, API idéntica para lo que usa este
   pipeline). Efecto colateral positivo: 2.2.2 también satisface el
   `torch>=2.2` que exige `kdm-torch`, alineando el entorno con el lado KDM
   del proyecto (potencialmente ya no hacen falta dos envs completamente
   separados, a confirmar cuando se arme el exp_02 con KDM).

En ningún momento fue un problema de cuota: `kaggle quota` mostró 30.00h/30.00h
de GPU disponibles durante todos los intentos.

## Desviaciones conocidas respecto al paper

- GPU: Kaggle P100/T4 en lugar del hardware original (no reportado en el
  paper) — wall-clock no comparable con el de los autores, sí entre nuestras
  condiciones.
- `run_mode = "disabled"` para wandb (los autores usaban wandb online); las
  métricas se capturan vía `test_neural` re-ejecutado y `metrics.json`.
