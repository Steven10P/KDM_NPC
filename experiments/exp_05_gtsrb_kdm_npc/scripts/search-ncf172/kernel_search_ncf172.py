#!/usr/bin/env python3
"""exp_05 -- plantilla de kernel para el barrido de hiperparametros (Fase A)
+ la confirmacion a escala completa (Fase B) de la cascada KDM (variante
Cartesian, unica -- exp_02 ya descarto Distributional) en GTSRB.

NO se corre directo -- _generate_kernel.py sustituye los placeholders
search-ncf172_seed42/42/15/10/172/
0.003/1.0 y escribe una copia por condicion en scripts/<condicion>/.

Generalizacion de exp_03_mnist_kdm_sweep/scripts/_template_kernel.py de 2
cabezas homogeneas (10 valores cada una) a 4 cabezas heterogeneas
(color=3, shape=4, symbol=26, text=10) -- ver
src/models/kdm_cascade_gtsrb.py (misma logica, ya probada localmente con
datos sinteticos en _test_kdm_cascade_gtsrb.py) y
exp_05_gtsrb_kdm_npc/IMPLEMENTATION.md secciones 1-2 para el diseño
completo.

Dataset: bspenad10/gtsrb-npc (39,209 imagenes = 31,367 train + 3,921
validate + 3,921 test; solo el split de entrenamiento oficial de GTSRB
tiene anotaciones de clase, ver
external/npc-dataset-utils/docs/npc-dataset-utils/datasets/gtsrb.md).
EXPECTED_GLOBAL_SHA256 calculado en
experiments/exp_05_gtsrb_kdm_npc/scripts/_package_kaggle_dataset.py.

Nota importante: el mirror de GTSRB que hay que usar es especificamente
meowmeowmeowmeowmeow/gtsrb-german-traffic-sign (Kaggle) -- otra copia de
GTSRB descargada de una fuente distinta (carpetas de clase sin el padding
correcto, extension .ppm en vez de .png) NO calza con los nombres de
archivo de gtsrb_split.json.gz.
"""

import glob
import json
import os
import subprocess
import sys
import time

WORK = "/kaggle/working"
NPC_ROOT = f"{WORK}/npc"
CONDITION = "search-ncf172_seed42"
SEED = 42
EPOCHS = 15
BATCH_SIZE = 256
N_COMP_PER_VALUE = 10
N_COMP_FINAL = 172
LR_KDM = 0.003
SIGMA_MULT = 1.0
RESULTS = f"{WORK}/results/{CONDITION}"
EXPECTED_GLOBAL_SHA256 = "1f398e558be886d4a787c97e85455d5310e912265212f2bcde8856e5a98fbdd0"
VAL_EVERY_N_EPOCHS = 5  # nuevo respecto a exp_03: registrar validacion desde la Fase A

WALLCLOCK = {}


def run(cmd, cwd=None, name=None):
    t0 = time.time()
    print(f"\n[RUN] {' '.join(cmd)} (cwd={cwd})", flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    dt = time.time() - t0
    if name:
        WALLCLOCK[name] = round(dt, 1)
    print(f"[RUN] terminado en {dt:.0f}s (rc={result.returncode})", flush=True)
    if result.returncode != 0:
        sys.exit(f"[FATAL] Fallo: {' '.join(cmd)}")


# --------------------------------------------------------------------------
# 1. Entorno: mismo fix que exp_01/exp_03 (PyPI puro, torch==2.2.2 por
#    Python 3.12 de Kaggle) + kdm-torch (satisface su propio torch>=2.2)
# --------------------------------------------------------------------------
t_env = time.time()
run([sys.executable, "-m", "pip", "install", "-q", "torch==2.2.2", "torchvision==0.17.2"])
run([sys.executable, "-m", "pip", "install", "-q",
     "numpy<2", "natsort==8.0.2", "PyQt5==5.15.11",
     "git+https://github.com/fagonzalezo/kdm.git"])
WALLCLOCK["pip_install"] = round(time.time() - t_env, 1)

# --------------------------------------------------------------------------
# 2. Clonar npc-models + npc-dataset-utils (solo para NPCDataset y splits)
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
# 3. Dataset congelado -> jerarquia oficial (descubrimiento dinamico del mount)
# --------------------------------------------------------------------------
_manifest_matches = glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True)
assert len(_manifest_matches) == 1, f"Se esperaba 1 MANIFEST.json: {_manifest_matches}"
INPUT_DIR = os.path.dirname(_manifest_matches[0])
print(f"[INFO] Dataset montado en: {INPUT_DIR}", flush=True)
print(f"[DEBUG] contenido de INPUT_DIR: {sorted(os.listdir(INPUT_DIR))}", flush=True)

t_data = time.time()
processed = f"{NPC_ROOT}/datasets/gtsrb/instances/processed"
os.makedirs(os.path.dirname(processed), exist_ok=True)
os.symlink(f"{INPUT_DIR}/gtsrb_processed", processed)

with open(f"{INPUT_DIR}/MANIFEST.json") as f:
    manifest = json.load(f)
assert manifest["global_sha256"] == EXPECTED_GLOBAL_SHA256

_class_dirs = sorted(os.listdir(processed))
print(f"[DEBUG] {len(_class_dirs)} carpetas de clase en processed, primeras 3: {_class_dirs[:3]}", flush=True)
_n_processed = sum(len(fs) for _, _, fs in os.walk(processed))
print(f"[DEBUG] {_n_processed} archivos totales en processed", flush=True)
_sample_dir = os.path.join(processed, _class_dirs[0])
print(f"[DEBUG] muestra de archivos en {_class_dirs[0]}: {sorted(os.listdir(_sample_dir))[:5]}", flush=True)
assert _n_processed == manifest["n_images"], f"{_n_processed} != {manifest['n_images']}"

import shutil, gzip  # noqa: E402
cfg_dir = f"{NPC_ROOT}/npc-dataset-utils/configs/npc-dataset-utils"
shutil.copy(f"{INPUT_DIR}/gtsrb.json", cfg_dir)
with open(f"{INPUT_DIR}/gtsrb_split.json", "rb") as f_in:
    with gzip.open(f"{cfg_dir}/gtsrb_split.json.gz", "wb") as f_out:
        f_out.write(f_in.read())
WALLCLOCK["data_setup"] = round(time.time() - t_data, 1)

for header_path in (f"{NPC_ROOT}/npc-dataset-utils/src/npc-dataset-utils/header.py",
                    f"{NPC_ROOT}/npc-models/src/npc-models/header.py"):
    with open(header_path) as f:
        content = f.read()
    patched = content.replace('dataset_prefix = "awa2"', 'dataset_prefix = "gtsrb"')
    assert patched != content, header_path
    with open(header_path, "w") as f:
        f.write(patched)

# dataset_file_extension_images default es ".jpg" (no matchea archivos
# .png -> split.py deja el file_key CON extension, que es lo que
# "accidentalmente" funcionaba para MNIST/.png). Para GTSRB los propios
# autores instruyen explicitamente cambiar esto a ".png" (ver
# external/npc-dataset-utils/docs/npc-dataset-utils/datasets/gtsrb.md) --
# sin este patch, split.py revienta con KeyError porque gtsrb_split.json.gz
# fue construido CON este ajuste (sus claves no llevan extension). Solo
# npc-dataset-utils/header.py declara esta variable (npc-models/header.py no).
dataset_utils_header = f"{NPC_ROOT}/npc-dataset-utils/src/npc-dataset-utils/header.py"
with open(dataset_utils_header) as f:
    content = f.read()
patched = content.replace('dataset_file_extension_images = ".jpg"',
                          'dataset_file_extension_images = ".png"')
assert patched != content, dataset_utils_header
with open(dataset_utils_header, "w") as f:
    f.write(patched)

utils_src = f"{NPC_ROOT}/npc-dataset-utils/src/npc-dataset-utils"
run([sys.executable, "split.py"], cwd=utils_src, name="split")

for split_name, expected in (("train", 31367), ("validate", 3921), ("test", 3921)):
    n = sum(len(fs) for _, _, fs in os.walk(f"{NPC_ROOT}/datasets/gtsrb/splits/instances/{split_name}"))
    assert n == expected, f"Split {split_name}: {n} != {expected}"

# --------------------------------------------------------------------------
# 4. KDMCascadeGTSRB (copia embebida de src/models/kdm_cascade_gtsrb.py, ya
#    probada localmente con datos sinteticos en _test_kdm_cascade_gtsrb.py)
# --------------------------------------------------------------------------
import math  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision  # noqa: E402

from kdm.layers import CosineKernelLayer, KDMLayer  # noqa: E402
from kdm.init import init_kdm_layer  # noqa: E402
from kdm.models import KDMClassModel  # noqa: E402
from kdm.utils import cartesian_product, dm2discrete, pure2dm  # noqa: E402

RESNET_NECK_SIZE = 512
N_CLASSES = 43
ATTRIBUTE_CARDINALITIES = {"color": 3, "shape": 4, "symbol": 26, "text": 10}


def build_shared_trunk() -> nn.Module:
    resnet = torchvision.models.resnet34(weights="IMAGENET1K_V1")
    resnet.fc = nn.Identity()
    return resnet


class KDMCascadeGTSRB(nn.Module):
    def __init__(self, n_comp_per_value, n_comp_final, sigma_head=1.0):
        super().__init__()
        assert n_comp_final % N_CLASSES == 0
        self.n_comp_per_value = n_comp_per_value
        self.n_comp_final = n_comp_final
        self.attribute_names = list(ATTRIBUTE_CARDINALITIES.keys())
        self.trunk = build_shared_trunk()
        self.heads = nn.ModuleDict({
            name: KDMClassModel(RESNET_NECK_SIZE, card, nn.Identity(),
                                n_comp_per_value * card, sigma=sigma_head)
            for name, card in ATTRIBUTE_CARDINALITIES.items()
        })
        dim_x_final = math.prod(ATTRIBUTE_CARDINALITIES.values())
        self.kdm_final = KDMLayer(kernel=CosineKernelLayer(),
                                  dim_x=dim_x_final, dim_y=N_CLASSES, n_comp=n_comp_final)

    def forward(self, image):
        neck = self.trunk(image)
        p = {name: head(neck) for name, head in self.heads.items()}
        joint = cartesian_product([p[name] for name in self.attribute_names])
        p_class = dm2discrete(self.kdm_final(pure2dm(joint)))
        return p, p_class

    @torch.no_grad()
    def init_components(self, images, attribute_labels, class_labels,
                        forward_batch_size=256, sigma_mult=1.0):
        device = next(self.parameters()).device
        neck_chunks = []
        for i in range(0, images.shape[0], forward_batch_size):
            neck_chunks.append(self.trunk(images[i:i + forward_batch_size].to(device)))
        neck = torch.cat(neck_chunks, dim=0)

        def stratified_idx(labels, n_values, n_total):
            per_value = n_total // n_values
            chosen = []
            for value in range(n_values):
                candidates = (labels == value).nonzero(as_tuple=True)[0]
                assert len(candidates) >= per_value, \
                    f"valor {value}: hay {len(candidates)}, se necesitan {per_value}"
                chosen.append(candidates[torch.randperm(len(candidates))[:per_value]])
            return torch.cat(chosen)

        idx_f = stratified_idx(class_labels, N_CLASSES, self.n_comp_final)
        true_onehots = {}
        for name, card in ATTRIBUTE_CARDINALITIES.items():
            n_total = self.n_comp_per_value * card
            idx = stratified_idx(attribute_labels[name], card, n_total)
            y = F.one_hot(attribute_labels[name][idx], card).float()
            init_kdm_layer(self.heads[name].kdm, neck[idx], y, init_sigma=True, sigma_mult=sigma_mult)
            true_onehots[name] = F.one_hot(attribute_labels[name][idx_f], card).float()

        y_f = F.one_hot(class_labels[idx_f], N_CLASSES).float()
        x_f = cartesian_product([true_onehots[name] for name in self.attribute_names])
        init_kdm_layer(self.kdm_final, x_f, y_f, init_sigma=True, sigma_mult=sigma_mult)


# --------------------------------------------------------------------------
# 5. Datos: NPCDataset (npc-models) sobre los splits ya materializados
# --------------------------------------------------------------------------
models_src = f"{NPC_ROOT}/npc-models/src/npc-models"
sys.path.insert(0, models_src)
os.chdir(models_src)  # header.py resuelve rutas relativas al cwd (bug ya corregido en exp_02)
from dataset import NPCDataset  # noqa: E402

tfm = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor(),
])
dataset_dir = f"{NPC_ROOT}/datasets/gtsrb/splits/instances"
ds_train = NPCDataset(f"{dataset_dir}/train", tfm)
ds_validate = NPCDataset(f"{dataset_dir}/validate", tfm)
ds_test = NPCDataset(f"{dataset_dir}/test", tfm)

ATTR_NAMES = list(ATTRIBUTE_CARDINALITIES.keys())

torch.manual_seed(SEED)
device = torch.device("cuda")
model = KDMCascadeGTSRB(n_comp_per_value=N_COMP_PER_VALUE, n_comp_final=N_COMP_FINAL).to(device)

# --------------------------------------------------------------------------
# 6. Inicializacion (batch estratificado grande)
# --------------------------------------------------------------------------
# GTSRB tiene desbalance de clase REAL (a diferencia de MNIST-Addition,
# balanceado por construccion) -- un muestreo de 3000 (como en exp_03/MNIST)
# no siempre trae suficientes ejemplos de los valores de atributo mas raros
# para estratificar (visto en la practica: "valor 7: hay 11, se necesitan
# 15"). 12000 (~38% del train set) da margen holgado sin cargar el train
# set completo en memoria (~7GB de tensores de imagen vs. ~19GB si fuera
# completo).
t_init = time.time()
init_loader = torch.utils.data.DataLoader(ds_train, batch_size=12000, shuffle=True, num_workers=2)
init_images, init_attrs, init_class, _ = next(iter(init_loader))
init_attribute_labels = {name: init_attrs[i].argmax(dim=1).to(device)
                         for i, name in enumerate(ATTR_NAMES)}
model.init_components(init_images.to(device), init_attribute_labels, init_class.to(device),
                      sigma_mult=SIGMA_MULT)
WALLCLOCK["init_components"] = round(time.time() - t_init, 1)
print(f"[INFO] init_components: {WALLCLOCK['init_components']}s", flush=True)

# --------------------------------------------------------------------------
# 7. Entrenamiento end-to-end (con validacion cada VAL_EVERY_N_EPOCHS --
#    nuevo respecto a exp_03, que solo registro train)
# --------------------------------------------------------------------------
loader_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=4, pin_memory=True)
loader_validate = torch.utils.data.DataLoader(ds_validate, batch_size=BATCH_SIZE, shuffle=False,
                                              num_workers=4, pin_memory=True)
loader_test = torch.utils.data.DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False,
                                          num_workers=4, pin_memory=True)

opt_trunk = torch.optim.SGD(model.trunk.parameters(), lr=0.01, momentum=0.9)
kdm_params = [p for n, p in model.named_parameters() if not n.startswith("trunk.")]
opt_kdm = torch.optim.Adam(kdm_params, lr=LR_KDM)


def compute_loss(p, p_class, attrs, classes):
    loss_class = F.nll_loss(torch.log(p_class + 1e-8), classes)
    loss_attr = sum(-(attrs[i] * torch.log(p[name] + 1e-8)).sum(dim=1).mean()
                    for i, name in enumerate(ATTR_NAMES))
    return loss_class + 0.5 * loss_attr


@torch.no_grad()
def evaluate(loader):
    model.eval()
    n_correct_class = n_correct_joint_attr = n_total = 0
    tv_total = 0.0
    loss_total = 0.0
    for images, attrs, classes, _ in loader:
        images, classes = images.to(device), classes.to(device)
        attrs = [a.to(device) for a in attrs]
        p, p_class = model(images)
        loss_total += compute_loss(p, p_class, attrs, classes).item() * images.size(0)
        n_correct_class += (p_class.argmax(dim=1) == classes).sum().item()
        correct_all_attrs = torch.ones(images.size(0), dtype=torch.bool, device=device)
        for i, name in enumerate(ATTR_NAMES):
            correct_all_attrs &= (p[name].argmax(dim=1) == attrs[i].argmax(dim=1))
            tv_total += (0.5 * (p[name] - attrs[i]).abs().sum(dim=1)).sum().item()
        n_correct_joint_attr += correct_all_attrs.sum().item()
        n_total += images.size(0)
    return {
        "loss": loss_total / n_total,
        "classification_accuracy": n_correct_class / n_total,
        "attribute_joint_accuracy": n_correct_joint_attr / n_total,
        "mean_tv_distance": tv_total / (n_total * len(ATTR_NAMES)),
    }


t_train = time.time()
epoch_times = []
train_loss_history = []
val_loss_history = {}
val_accuracy_history = {}
for epoch in range(1, EPOCHS + 1):
    t_epoch = time.time()
    model.train()
    epoch_loss = 0.0
    for images, attrs, classes, _ in loader_train:
        images, classes = images.to(device, non_blocking=True), classes.to(device, non_blocking=True)
        attrs = [a.to(device, non_blocking=True) for a in attrs]

        opt_trunk.zero_grad(); opt_kdm.zero_grad()
        p, p_class = model(images)
        loss = compute_loss(p, p_class, attrs, classes)
        loss.backward()
        opt_trunk.step(); opt_kdm.step()
        epoch_loss += loss.item() * images.size(0)

    epoch_loss /= len(ds_train)
    dt_epoch = time.time() - t_epoch
    epoch_times.append(dt_epoch)
    train_loss_history.append(round(epoch_loss, 4))

    log_line = f"[INFO] epoca {epoch}/{EPOCHS}  loss={epoch_loss:.4f}  ({dt_epoch:.0f}s)"
    if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == EPOCHS:
        val_metrics = evaluate(loader_validate)
        val_loss_history[epoch] = round(val_metrics["loss"], 4)
        val_accuracy_history[epoch] = round(val_metrics["classification_accuracy"], 4)
        log_line += f"  val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['classification_accuracy']:.4f}"
    print(log_line, flush=True)

WALLCLOCK["train_total"] = round(time.time() - t_train, 1)
WALLCLOCK["train_per_epoch_mean"] = round(sum(epoch_times) / len(epoch_times), 1)

# --------------------------------------------------------------------------
# 8. Evaluacion final en test
# --------------------------------------------------------------------------
t_eval = time.time()
metrics_eval = evaluate(loader_test)
WALLCLOCK["eval"] = round(time.time() - t_eval, 1)
print(f"[INFO] eval: {metrics_eval}", flush=True)

# --------------------------------------------------------------------------
# Recolectar resultados
# --------------------------------------------------------------------------
os.makedirs(RESULTS, exist_ok=True)
torch.save(model.state_dict(), f"{RESULTS}/model.pt")

n_params_total = sum(p.numel() for p in model.parameters())
n_params_final = sum(p.numel() for p in model.kdm_final.parameters())

metrics = {
    "experiment": "exp_05_gtsrb_kdm_npc",
    "condition": CONDITION,
    "final_mode": "cartesian",
    "seed": SEED,
    "hyperparameters": {
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "n_comp_per_value": N_COMP_PER_VALUE, "n_comp_final": N_COMP_FINAL,
        "lr_kdm": LR_KDM, "sigma_mult": SIGMA_MULT,
    },
    "dataset_global_sha256": manifest["global_sha256"],
    "repo_commits": commits,
    "environment": {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    },
    "wallclock_seconds": WALLCLOCK,
    "n_parameters": {"total": n_params_total, "kdm_final": n_params_final},
    "train_loss_history": train_loss_history,
    "val_loss_history": val_loss_history,
    "val_accuracy_history": val_accuracy_history,
    "metrics": metrics_eval,
}
with open(f"{RESULTS}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# git_commit.txt -- nuevo respecto a exp_03, que no lo genero (ver DESIGN.md 7)
try:
    tesis_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd="/kaggle/working").decode().strip()
except Exception:
    tesis_commit = "unknown (repo tesis no clonado en este kernel)"
with open(f"{RESULTS}/git_commit.txt", "w") as f:
    f.write(f"npc-models: {commits['npc-models']}\n")
    f.write(f"npc-dataset-utils: {commits['npc-dataset-utils']}\n")
    f.write(f"Tesis_KDM_NPC: {tesis_commit}\n")

shutil.rmtree(NPC_ROOT, ignore_errors=True)
print("\n[DONE] metrics.json:", flush=True)
print(json.dumps(metrics, indent=2), flush=True)
