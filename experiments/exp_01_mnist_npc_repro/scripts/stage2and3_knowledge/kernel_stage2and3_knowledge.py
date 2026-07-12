#!/usr/bin/env python3
"""exp_01 / etapas 2+3, variante Knowledge — evalúa el circuito inyectado por
conocimiento y corre la optimización conjunta de NPC(Knowledge) en Kaggle GPU.

Condiciones: circuit-knowledge (etapa 2 sola) + npc-knowledge_seed<SEED> (etapa 3)

Por qué NO se corre Java/LearnSPN aquí: el circuito Knowledge es un depth-2
sum-of-products cuyos pesos son directamente la frecuencia empírica de cada
regla (paper, Sec. 3.2.2, "Parameter Assignment" + Proposition 1: el nodo raíz
YA representa la distribución conjunta empírica) — no hay entrenamiento CCCP
para esta variante. El repo oficial `learnspn` ya trae ese archivo
precomputado (`outputs/manual/mnist.spn.txt`, generado por
`scripts/manual/manual.py`), así que solo se evalúa directamente.

Requiere `kernel_sources: ["bspenad10/exp01-npc-mnist-stage1-seed42"]` en
kernel-metadata.json (monta el checkpoint del reconocedor de atributos ya
entrenado en /kaggle/input/).

Salidas en /kaggle/working/results/npc-knowledge_seed<SEED>/:
  metrics.json (incluye circuit-knowledge Y npc-knowledge en un solo archivo,
  con secciones separadas — misma corrida de kernel, dos condiciones lógicas)
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
CONDITION = f"npc-knowledge_seed{SEED}"
INPUT_DIR = "/kaggle/input/mnist-addition-npc"
STAGE1_GLOB = "/kaggle/input/*/npc-neural_seed*/*.best.zip"
WORK = "/kaggle/working"
NPC_ROOT = f"{WORK}/npc"
RESULTS = f"{WORK}/results/{CONDITION}"
EXPECTED_GLOBAL_SHA256 = "4d640365ca4b505d14ae79a83dcdc799d546a121a676e0a26a01c42fb2b9e07d"

WALLCLOCK = {}


def run_captured(cmd, cwd=None, name=None):
    """Ejecuta un comando, captura stdout+stderr, mide wall-clock y falla ruidosamente."""
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
# 2. Clonar repos oficiales (incluye learnspn: trae el circuito Knowledge ya calculado)
# --------------------------------------------------------------------------
os.makedirs(NPC_ROOT, exist_ok=True)
commits = {}
for repo in ("npc-models", "npc-dataset-utils", "learnspn"):
    dest = f"{NPC_ROOT}/{repo}"
    run(["git", "clone", "--depth", "1", f"https://github.com/uiuctml/{repo}.git", dest])
    commits[repo] = subprocess.check_output(
        ["git", "log", "-1", "--format=%H"], cwd=dest).decode().strip()

manual_spn = f"{NPC_ROOT}/learnspn/outputs/manual/mnist.spn.txt"
assert os.path.isfile(manual_spn), f"No se encontró el circuito Knowledge en {manual_spn}"

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
# --------------------------------------------------------------------------
for header_path in (f"{NPC_ROOT}/npc-dataset-utils/src/npc-dataset-utils/header.py",
                    f"{NPC_ROOT}/npc-models/src/npc-models/header.py"):
    with open(header_path) as f:
        content = f.read()
    patched = content.replace('dataset_prefix = "awa2"', 'dataset_prefix = "mnist"')
    assert patched != content, header_path
    with open(header_path, "w") as f:
        f.write(patched)

# El circuito Knowledge ya tiene pesos finales (frecuencia empírica) —
# apuntar file_path_pc directo al archivo de learnspn, sin pasar por CCCP.
models_header_path = f"{NPC_ROOT}/npc-models/src/npc-models/header.py"
with open(models_header_path) as f:
    content = f.read()
patched = content.replace(
    '"file_path_pc": "../../../learnspn/outputs/learnspn/" + dataset_prefix + ".spn.txt"',
    '"file_path_pc": "../../../learnspn/outputs/manual/" + dataset_prefix + ".spn.txt"',
)
assert patched != content, "No se encontró file_path_pc en header.py"
with open(models_header_path, "w") as f:
    f.write(patched)

# --------------------------------------------------------------------------
# 5. Splits oficiales (imágenes, para train_npc.py) + splits PC (texto, para test_pc.py)
# --------------------------------------------------------------------------
utils_src = f"{NPC_ROOT}/npc-dataset-utils/src/npc-dataset-utils"
run([sys.executable, "split.py"], cwd=utils_src, name="split")
run([sys.executable, "pc.py"], cwd=utils_src, name="pc_splits")

for split_name, expected in (("train", 28000), ("validate", 3500), ("test", 3500)):
    n = sum(len(fs) for _, _, fs in os.walk(f"{NPC_ROOT}/datasets/mnist/splits/instances/{split_name}"))
    assert n == expected, f"Split {split_name}: {n} != {expected}"

# --------------------------------------------------------------------------
# 6. Etapa 2 (sola): evaluar Circuit(Knowledge) -> Tabla 4
# --------------------------------------------------------------------------
models_src = f"{NPC_ROOT}/npc-models/src/npc-models"
stdout_circuit = run_captured([sys.executable, "test_pc.py"], cwd=models_src, name="test_pc_knowledge")
circuit_metrics = {
    "log_mean_likelihood": extract_metric(stdout_circuit, "Testing log mean likelihood"),
    "mean_likelihood": extract_metric(stdout_circuit, "Testing mean likelihood"),
}
print(f"[INFO] Circuit(Knowledge): {circuit_metrics}", flush=True)

# --------------------------------------------------------------------------
# 7. Copiar el checkpoint de la etapa 1 (montado como kernel_sources)
# --------------------------------------------------------------------------
stage1_matches = glob.glob(STAGE1_GLOB)
assert len(stage1_matches) == 1, f"Se esperaba 1 checkpoint de etapa 1, hay {stage1_matches}"
stage1_ckpt = stage1_matches[0]
ckpt_dir = f"{NPC_ROOT}/npc-models/outputs/npc-models/checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(stage1_ckpt, ckpt_dir)
stage1_ckpt_name = os.path.basename(stage1_ckpt)
print(f"[INFO] Checkpoint etapa 1: {stage1_ckpt_name}", flush=True)

# --------------------------------------------------------------------------
# 8. Etapa 3: optimización conjunta NPC(Knowledge) -> Tabla 2 y Tabla 4 (Model)
#    (circuito congelado: header.npc_pc_backward = False por defecto)
# --------------------------------------------------------------------------
stdout_npc = run_captured(
    [sys.executable, "train_npc.py", "-w", stage1_ckpt_name, "-s", str(SEED)],
    cwd=models_src, name="train_npc_knowledge")

npc_metrics = {
    "mean_tv_distance": extract_metric(stdout_npc, "Testing mean TV distance"),
    "mean_concept_accuracy": extract_metric(stdout_npc, "Testing mean concept accuracy"),
    "classification_accuracy": extract_metric(stdout_npc, "Testing classification accuracy"),
}
print(f"[INFO] NPC(Knowledge): {npc_metrics}", flush=True)

# --------------------------------------------------------------------------
# Recolectar resultados
# --------------------------------------------------------------------------
os.makedirs(RESULTS, exist_ok=True)
for fn in os.listdir(ckpt_dir):
    if fn != stage1_ckpt_name:  # no re-subir el checkpoint de entrada
        shutil.copy(os.path.join(ckpt_dir, fn), RESULTS)

import torch
metrics = {
    "experiment": "exp_01_mnist_npc_repro",
    "condition": CONDITION,
    "variant": "knowledge",
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
        "circuit_knowledge_stage2": circuit_metrics,
        "npc_knowledge_stage3": npc_metrics,
    },
    "paper_reference": {
        "table4_circuit_knowledge_mean_likelihood": 1.007e-2,
        "table4_model_knowledge_accuracy": 0.9917,
        "table2_npc_knowledge_accuracy_mean": 0.99189,
        "table2_npc_knowledge_accuracy_std": 0.08,
    },
}
with open(f"{RESULTS}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

shutil.rmtree(NPC_ROOT, ignore_errors=True)
print("\n[DONE] metrics.json:", flush=True)
print(json.dumps(metrics, indent=2), flush=True)
