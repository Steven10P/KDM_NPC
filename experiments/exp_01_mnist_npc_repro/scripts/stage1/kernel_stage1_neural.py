#!/usr/bin/env python3
"""exp_01 / etapa 1 — Entrena el reconocedor de atributos de NPC (ResNet34MTL)
en MNIST-Addition, replicando fielmente el pipeline oficial en Kaggle GPU.

Condición: npc-neural_seed<SEED>  (etapa 1 compartida por Knowledge y Data)

Qué hace este kernel:
  1. Instala las versiones pinneadas de npc-models/requirements.txt (torch 2.1.2).
  2. Clona npc-models y npc-dataset-utils y registra los commits exactos.
  3. Monta la jerarquía oficial npc/{datasets,npc-dataset-utils,npc-models}.
  4. Descomprime el dataset congelado (Kaggle Dataset bspenad10/mnist-addition-npc,
     hash global verificado contra MANIFEST.json).
  5. Aplica el único cambio de configuración: dataset_prefix "awa2" -> "mnist"
     (mecanismo de configuración propio del repo, no un cambio funcional).
  6. Genera splits oficiales (split.py, split_load=True) y splits PC (pc.py).
  7. Corre train_neural.py -s <SEED> (batch 256, 150 épocas, SGD lr 0.01 — los
     hiperparámetros oficiales viven en header.py y no se tocan).
  8. Evalúa el mejor checkpoint con la métrica EXACTA de test_neural.test()
     (TV media + accuracy media de conceptos) y escribe metrics.json.

Salidas en /kaggle/working/results/npc-neural_seed<SEED>/:
  metrics.json, checkpoints (*.zip), environment.json
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

# --------------------------------------------------------------------------
# Parámetros de la corrida (editar por condición)
# --------------------------------------------------------------------------
SEED = 42
CONDITION = f"npc-neural_seed{SEED}"
INPUT_DIR = "/kaggle/input/mnist-addition-npc"
WORK = "/kaggle/working"
NPC_ROOT = f"{WORK}/npc"
RESULTS = f"{WORK}/results/{CONDITION}"
EXPECTED_GLOBAL_SHA256 = "4d640365ca4b505d14ae79a83dcdc799d546a121a676e0a26a01c42fb2b9e07d"

WALLCLOCK = {}


def run(cmd, cwd=None, name=None):
    """Ejecuta un comando, mide wall-clock y falla ruidosamente."""
    t0 = time.time()
    print(f"\n[RUN] {' '.join(cmd)} (cwd={cwd})", flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    dt = time.time() - t0
    if name:
        WALLCLOCK[name] = round(dt, 1)
    print(f"[RUN] terminado en {dt:.0f}s (rc={result.returncode})", flush=True)
    if result.returncode != 0:
        sys.exit(f"[FATAL] Falló: {' '.join(cmd)}")


# --------------------------------------------------------------------------
# 1. Entorno (base: npc-models/requirements.txt; sin PyQt5 — GUI no usada)
#
# DOS desviaciones forzadas por la plataforma Kaggle, ambas documentadas en
# IMPLEMENTATION.md:
#  - --index-url https://download.pytorch.org/whl/cu121 falla (DNS no
#    resuelve ese dominio); se instala desde PyPI estándar en su lugar.
#  - torch==2.1.2 (el pin exacto del paper) no tiene wheels para Python 3.12
#    (la imagen de Kaggle corre 3.12; 2.1.2 es pre-3.12). Se usa 2.2.2, el
#    primer 2.2.x estable — misma serie mayor, API idéntica para lo que usa
#    este pipeline (nn.Module, DataParallel, SGD, ReduceLROnPlateau,
#    torchvision.models.resnet34). De paso 2.2.2 también satisface el
#    torch>=2.2 que exige kdm-torch, alineando el entorno con el lado KDM.
# --------------------------------------------------------------------------
t_env = time.time()
run([sys.executable, "-m", "pip", "install", "-q",
     "torch==2.2.2", "torchvision==0.17.2"])
run([sys.executable, "-m", "pip", "install", "-q",
     "numpy<2", "natsort==8.0.2", "torch_explain==1.5.1",
     "scikit-learn==1.3.2", "wandb==0.16.1", "tqdm"])
WALLCLOCK["pip_install"] = round(time.time() - t_env, 1)

# --------------------------------------------------------------------------
# 2. Clonar repos oficiales y registrar commits
# --------------------------------------------------------------------------
os.makedirs(NPC_ROOT, exist_ok=True)
commits = {}
for repo in ("npc-models", "npc-dataset-utils"):
    dest = f"{NPC_ROOT}/{repo}"
    run(["git", "clone", "--depth", "1", f"https://github.com/uiuctml/{repo}.git", dest])
    commits[repo] = subprocess.check_output(
        ["git", "log", "-1", "--format=%H"], cwd=dest).decode().strip()
print("[INFO] commits:", commits, flush=True)

# --------------------------------------------------------------------------
# 3-4. Dataset congelado -> jerarquía oficial + verificación de hash
# --------------------------------------------------------------------------
t_data = time.time()
processed = f"{NPC_ROOT}/datasets/mnist/instances/processed"
os.makedirs(processed, exist_ok=True)
with zipfile.ZipFile(f"{INPUT_DIR}/mnist_addition_processed.zip") as zf:
    zf.extractall(processed)

n_files = sum(len(fs) for _, _, fs in os.walk(processed))
assert n_files == 35000, f"Se esperaban 35000 imágenes, hay {n_files}"

with open(f"{INPUT_DIR}/MANIFEST.json") as f:
    manifest = json.load(f)
assert manifest["global_sha256"] == EXPECTED_GLOBAL_SHA256, \
    "El MANIFEST del Kaggle Dataset no coincide con el hash congelado en DESIGN.md"

# Configs oficiales (mapeos + splits de los autores) donde el pipeline los busca
cfg_dir = f"{NPC_ROOT}/npc-dataset-utils/configs/npc-dataset-utils"
shutil.copy(f"{INPUT_DIR}/mnist.json", cfg_dir)
shutil.copy(f"{INPUT_DIR}/mnist_split.json.gz", cfg_dir)
WALLCLOCK["data_setup"] = round(time.time() - t_data, 1)

# --------------------------------------------------------------------------
# 5. Único cambio de config: dataset_prefix -> "mnist" en ambos header.py
# --------------------------------------------------------------------------
for header_path in (f"{NPC_ROOT}/npc-dataset-utils/src/npc-dataset-utils/header.py",
                    f"{NPC_ROOT}/npc-models/src/npc-models/header.py"):
    with open(header_path) as f:
        content = f.read()
    patched = content.replace('dataset_prefix = "awa2"', 'dataset_prefix = "mnist"')
    assert patched != content, f"No se encontró dataset_prefix en {header_path}"
    with open(header_path, "w") as f:
        f.write(patched)
print("[INFO] dataset_prefix = mnist aplicado en ambos header.py", flush=True)

# --------------------------------------------------------------------------
# 6. Splits oficiales (symlinks) + splits PC
# --------------------------------------------------------------------------
utils_src = f"{NPC_ROOT}/npc-dataset-utils/src/npc-dataset-utils"
run([sys.executable, "split.py"], cwd=utils_src, name="split")
run([sys.executable, "pc.py"], cwd=utils_src, name="pc_splits")

for split_name, expected in (("train", 28000), ("validate", 3500), ("test", 3500)):
    split_dir = f"{NPC_ROOT}/datasets/mnist/splits/instances/{split_name}"
    n = sum(len(fs) for _, _, fs in os.walk(split_dir))
    assert n == expected, f"Split {split_name}: {n} != {expected}"
    print(f"[OK] split {split_name}: {n} instancias", flush=True)

# --------------------------------------------------------------------------
# 7. Etapa 1: entrenar el reconocedor de atributos
# --------------------------------------------------------------------------
models_src = f"{NPC_ROOT}/npc-models/src/npc-models"
run([sys.executable, "train_neural.py", "-s", str(SEED)],
    cwd=models_src, name="train_neural")

# --------------------------------------------------------------------------
# 8. Evaluación del mejor checkpoint con la métrica exacta de test_neural
#    (test_neural.py ya corre dentro de train_neural.py y deja las métricas en
#    el log; aquí lo repetimos de forma capturable para metrics.json)
# --------------------------------------------------------------------------
# El run name real incluye timestamp y hostname (utility.generateRunName:
# <seed>.<dataset>.<type>.<model>.<y>.<m>.<d>.<h>.<min>.<s>.<host>), así que se
# descubre desde el nombre del checkpoint .best.zip que dejó el entrenamiento.
ckpt_dir = f"{NPC_ROOT}/npc-models/outputs/npc-models/checkpoints"
best_ckpts = [fn for fn in os.listdir(ckpt_dir) if fn.endswith(".best.zip")]
assert len(best_ckpts) == 1, f"Se esperaba 1 checkpoint .best.zip: {best_ckpts}"
run_name = best_ckpts[0][:-len(".best.zip")]
assert run_name.split(".")[:4] == [str(SEED), "mnist", "neural", "resnet34mtl"], run_name
print(f"[INFO] run_name detectado: {run_name}", flush=True)

eval_code = r"""
import io, json, sys, contextlib
sys.argv = ["test_neural.py", "-r", "RUN_NAME"]
import test_neural, header

buf = io.StringIO()
with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
    test_neural.main()
log = buf.getvalue()
print(log)

tv = acc = None
for line in log.splitlines():
    if "Testing mean TV distance" in line:
        tv = float(line.rsplit(":", 1)[1].strip().rstrip("."))
    if "Testing mean concept accuracy" in line:
        acc = float(line.rsplit(":", 1)[1].strip().rstrip("."))
with open("/kaggle/working/stage1_eval.json", "w") as f:
    json.dump({"mean_tv_distance": tv, "mean_concept_accuracy": acc}, f)
"""
with open(f"{models_src}/_eval_capture.py", "w") as f:
    f.write(eval_code.replace("RUN_NAME", run_name))
run([sys.executable, "_eval_capture.py"], cwd=models_src, name="test_neural")

# --------------------------------------------------------------------------
# Recolectar resultados
# --------------------------------------------------------------------------
os.makedirs(RESULTS, exist_ok=True)
ckpt_dir = f"{NPC_ROOT}/npc-models/outputs/npc-models/checkpoints"
for fn in os.listdir(ckpt_dir):
    shutil.copy(os.path.join(ckpt_dir, fn), RESULTS)

with open(f"{WORK}/stage1_eval.json") as f:
    eval_metrics = json.load(f)

import torch  # ya instalado arriba
metrics = {
    "experiment": "exp_01_mnist_npc_repro",
    "condition": CONDITION,
    "stage": "1-attribute-recognition",
    "seed": SEED,
    "run_name": run_name,
    "dataset_global_sha256": manifest["global_sha256"],
    "repo_commits": commits,
    "environment": {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    },
    "wallclock_seconds": WALLCLOCK,
    "metrics": eval_metrics,
    "paper_reference": {
        "table3_abm_mean_tv": 0.0058,
        "table3_abm_mean_concept_accuracy": 0.9899,
    },
}
with open(f"{RESULTS}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# La jerarquía npc/ es enorme (35k imágenes + symlinks); no debe persistirse
# como output del kernel. Kaggle solo persiste /kaggle/working, así que la
# limpiamos dejando solo results/.
shutil.rmtree(NPC_ROOT, ignore_errors=True)
os.remove(f"{WORK}/stage1_eval.json")

print("\n[DONE] metrics.json:", flush=True)
print(json.dumps(metrics, indent=2), flush=True)
