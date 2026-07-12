#!/usr/bin/env python3
"""exp_01 / etapas 2+3, variante Data — entrena el circuito data-driven (CCCP)
y corre la optimización conjunta de NPC(Data) en Kaggle GPU.

Condiciones: circuit-data (etapa 2 sola) + npc-data_seed<SEED> (etapa 3)

Por qué NO se corre Java/LearnSPN aquí: la ESTRUCTURA del circuito (aprendida
por LearnSPN sobre D̄) ya viene precomputada en el repo oficial `learnspn`
(`outputs/learnspn/mnist.spn.txt`, 30,743 líneas) — LearnSPN es Java+bash y
solo se necesitaría para regenerar la estructura desde cero, lo cual no hace
falta para replicar los resultados del paper. Lo que SÍ hace falta correr aquí
es CCCP (`train_pc.py`, paper Sec. 3.2.2 "Parameter Learning"): parte de esa
estructura con sus pesos iniciales de LearnSPN y optimiza los pesos por MLE.

Requiere `kernel_sources: ["bspenad10/exp01-npc-mnist-stage1-seed42"]` en
kernel-metadata.json.

Salidas en /kaggle/working/results/npc-data_seed<SEED>/:
  metrics.json (circuit-data + npc-data en un solo archivo)
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

SEED = 42
CONDITION = f"npc-data_seed{SEED}"
INPUT_DIR = "/kaggle/input/mnist-addition-npc"
STAGE1_GLOB = "/kaggle/input/*/npc-neural_seed*/*.best.zip"
WORK = "/kaggle/working"
NPC_ROOT = f"{WORK}/npc"
RESULTS = f"{WORK}/results/{CONDITION}"
EXPECTED_GLOBAL_SHA256 = "4d640365ca4b505d14ae79a83dcdc799d546a121a676e0a26a01c42fb2b9e07d"

WALLCLOCK = {}


def run_captured(cmd, cwd=None, name=None):
    t0 = time.time()
    print(f"\n[RUN] {' '.join(cmd)} (cwd={cwd})", flush=True)
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    dt = time.time() - t0
    if name:
        WALLCLOCK[name] = round(dt, 1)
    print(result.stdout[-4000:], flush=True)
    print(result.stderr[-2000:], file=sys.stderr, flush=True)
    print(f"[RUN] terminado en {dt:.0f}s (rc={result.returncode})", flush=True)
    if result.returncode != 0:
        sys.exit(f"[FATAL] Falló: {' '.join(cmd)}")
    return result.stdout


def run(cmd, cwd=None, name=None):
    t0 = time.time()
    print(f"\n[RUN] {' '.join(cmd)} (cwd={cwd})", flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    dt = time.time() - t0
    if name:
        WALLCLOCK[name] = round(dt, 1)
    print(f"[RUN] terminado en {dt:.0f}s (rc={result.returncode})", flush=True)
    if result.returncode != 0:
        sys.exit(f"[FATAL] Falló: {' '.join(cmd)}")


def extract_metric(text, label):
    m = re.search(re.escape(label) + r":\s*([-\d.eE]+)", text)
    return float(m.group(1)) if m else None


def extract_all(text, label):
    """Todas las ocurrencias (train_pc.py imprime esto por época; nos interesa la última = mejor checkpoint)."""
    return [float(x) for x in re.findall(re.escape(label) + r":\s*([-\d.eE]+)", text)]


# --------------------------------------------------------------------------
# 1. Entorno (mismo fix que etapa 1: PyPI directo + torch 2.2.2 en vez de
#    2.1.2, que no tiene wheels para Python 3.12 — ver kernel de etapa 1)
# --------------------------------------------------------------------------
t_env = time.time()
run([sys.executable, "-m", "pip", "install", "-q", "torch==2.2.2", "torchvision==0.17.2"])
run([sys.executable, "-m", "pip", "install", "-q",
     "numpy<2", "natsort==8.0.2", "torch_explain==1.5.1",
     "scikit-learn==1.3.2", "wandb==0.16.1", "tqdm"])
WALLCLOCK["pip_install"] = round(time.time() - t_env, 1)

# --------------------------------------------------------------------------
# 2. Clonar repos oficiales (incluye learnspn: trae la estructura Data ya aprendida)
# --------------------------------------------------------------------------
os.makedirs(NPC_ROOT, exist_ok=True)
commits = {}
for repo in ("npc-models", "npc-dataset-utils", "learnspn"):
    dest = f"{NPC_ROOT}/{repo}"
    run(["git", "clone", "--depth", "1", f"https://github.com/uiuctml/{repo}.git", dest])
    commits[repo] = subprocess.check_output(
        ["git", "log", "-1", "--format=%H"], cwd=dest).decode().strip()

learnspn_spn = f"{NPC_ROOT}/learnspn/outputs/learnspn/mnist.spn.txt"
assert os.path.isfile(learnspn_spn), f"No se encontró la estructura Data en {learnspn_spn}"

# --------------------------------------------------------------------------
# 3. Dataset congelado -> jerarquía oficial + verificación de hash
# --------------------------------------------------------------------------
t_data = time.time()
processed = f"{NPC_ROOT}/datasets/mnist/instances/processed"
os.makedirs(processed, exist_ok=True)
with zipfile.ZipFile(f"{INPUT_DIR}/mnist_addition_processed.zip") as zf:
    zf.extractall(processed)
assert sum(len(fs) for _, _, fs in os.walk(processed)) == 35000

with open(f"{INPUT_DIR}/MANIFEST.json") as f:
    manifest = json.load(f)
assert manifest["global_sha256"] == EXPECTED_GLOBAL_SHA256

cfg_dir = f"{NPC_ROOT}/npc-dataset-utils/configs/npc-dataset-utils"
shutil.copy(f"{INPUT_DIR}/mnist.json", cfg_dir)
shutil.copy(f"{INPUT_DIR}/mnist_split.json.gz", cfg_dir)
WALLCLOCK["data_setup"] = round(time.time() - t_data, 1)

# --------------------------------------------------------------------------
# 4. Único cambio de config: dataset_prefix -> "mnist"
#    (file_path_pc de config_pc YA apunta a outputs/learnspn/<prefix>.spn.txt
#    por defecto — no hace falta tocarlo para esta variante)
# --------------------------------------------------------------------------
for header_path in (f"{NPC_ROOT}/npc-dataset-utils/src/npc-dataset-utils/header.py",
                    f"{NPC_ROOT}/npc-models/src/npc-models/header.py"):
    with open(header_path) as f:
        content = f.read()
    patched = content.replace('dataset_prefix = "awa2"', 'dataset_prefix = "mnist"')
    assert patched != content, header_path
    with open(header_path, "w") as f:
        f.write(patched)

# --------------------------------------------------------------------------
# 5. Splits oficiales (imágenes, para train_npc.py) + splits PC (texto, para train_pc.py)
# --------------------------------------------------------------------------
utils_src = f"{NPC_ROOT}/npc-dataset-utils/src/npc-dataset-utils"
run([sys.executable, "split.py"], cwd=utils_src, name="split")
run([sys.executable, "pc.py"], cwd=utils_src, name="pc_splits")

for split_name, expected in (("train", 28000), ("validate", 3500), ("test", 3500)):
    n = sum(len(fs) for _, _, fs in os.walk(f"{NPC_ROOT}/datasets/mnist/splits/instances/{split_name}"))
    assert n == expected, f"Split {split_name}: {n} != {expected}"

# --------------------------------------------------------------------------
# 6. Etapa 2: CCCP sobre la estructura LearnSPN -> Circuit(Data), Tabla 4
# --------------------------------------------------------------------------
models_src = f"{NPC_ROOT}/npc-models/src/npc-models"
stdout_pc = run_captured([sys.executable, "train_pc.py", "-s", str(SEED)],
                         cwd=models_src, name="train_pc_data_cccp")

train_lls = extract_all(stdout_pc, "Validation log mean likelihood")
circuit_metrics = {
    "final_test_log_mean_likelihood": extract_metric(stdout_pc, "Testing log mean likelihood"),
    "final_test_mean_likelihood": extract_metric(stdout_pc, "Testing mean likelihood"),
    "best_validation_log_mean_likelihood": extract_metric(stdout_pc, "Best validation log mean likelihood"),
    "n_validation_epochs_logged": len(train_lls),
}
print(f"[INFO] Circuit(Data): {circuit_metrics}", flush=True)

ckpt_dir = f"{NPC_ROOT}/npc-models/outputs/npc-models/checkpoints"
pc_best_ckpts = [fn for fn in os.listdir(ckpt_dir)
                 if ".pc.cccp." in fn and fn.endswith(".best.zip")]
assert len(pc_best_ckpts) == 1, f"Se esperaba 1 checkpoint pc .best.zip: {pc_best_ckpts}"
pc_ckpt_name = pc_best_ckpts[0]
print(f"[INFO] Checkpoint circuito CCCP: {pc_ckpt_name}", flush=True)

# --------------------------------------------------------------------------
# 7. Copiar el checkpoint de la etapa 1 (montado como kernel_sources)
# --------------------------------------------------------------------------
stage1_matches = glob.glob(STAGE1_GLOB)
assert len(stage1_matches) == 1, f"Se esperaba 1 checkpoint de etapa 1, hay {stage1_matches}"
stage1_ckpt = stage1_matches[0]
shutil.copy(stage1_ckpt, ckpt_dir)
stage1_ckpt_name = os.path.basename(stage1_ckpt)
print(f"[INFO] Checkpoint etapa 1: {stage1_ckpt_name}", flush=True)

# --------------------------------------------------------------------------
# 8. Etapa 3: optimización conjunta NPC(Data) -> Tabla 2 y Tabla 4 (Model)
#    -c pasa el checkpoint CCCP-entrenado; el circuito queda congelado en
#    stage 3 (header.npc_pc_backward = False por defecto).
# --------------------------------------------------------------------------
stdout_npc = run_captured(
    [sys.executable, "train_npc.py", "-w", stage1_ckpt_name, "-c", pc_ckpt_name, "-s", str(SEED)],
    cwd=models_src, name="train_npc_data")

npc_metrics = {
    "mean_tv_distance": extract_metric(stdout_npc, "Testing mean TV distance"),
    "mean_concept_accuracy": extract_metric(stdout_npc, "Testing mean concept accuracy"),
    "classification_accuracy": extract_metric(stdout_npc, "Testing classification accuracy"),
}
print(f"[INFO] NPC(Data): {npc_metrics}", flush=True)

# --------------------------------------------------------------------------
# Recolectar resultados
# --------------------------------------------------------------------------
os.makedirs(RESULTS, exist_ok=True)
for fn in os.listdir(ckpt_dir):
    if fn != stage1_ckpt_name:
        shutil.copy(os.path.join(ckpt_dir, fn), RESULTS)

import torch
metrics = {
    "experiment": "exp_01_mnist_npc_repro",
    "condition": CONDITION,
    "variant": "data",
    "seed": SEED,
    "dataset_global_sha256": manifest["global_sha256"],
    "repo_commits": commits,
    "stage1_checkpoint_used": stage1_ckpt_name,
    "environment": {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    },
    "wallclock_seconds": WALLCLOCK,
    "metrics": {
        "circuit_data_stage2": circuit_metrics,
        "npc_data_stage3": npc_metrics,
    },
    "paper_reference": {
        "table4_circuit_data_mean_likelihood": 1.010e-2,
        "table4_model_data_accuracy": 0.9917,
        "table2_npc_data_accuracy_mean": 0.99171,
        "table2_npc_data_accuracy_std": 0.11,
    },
}
with open(f"{RESULTS}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

shutil.rmtree(NPC_ROOT, ignore_errors=True)
print("\n[DONE] metrics.json:", flush=True)
print(json.dumps(metrics, indent=2), flush=True)
