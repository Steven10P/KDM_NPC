#!/usr/bin/env python3
"""exp_03 — plantilla de kernel para el barrido de hiperparámetros + la
confirmación a escala completa de la cascada KDM (variante Cartesian, ganadora
de exp_02) en MNIST-Addition.

NO se corre directo -- _generate_kernel.py sustituye los placeholders
search-confirm_seed42/42/15/200/380/0.003/
0.5 y escribe una copia por condición en scripts/<condición>/.

Basada en experiments/exp_02_mnist_kdm_base/scripts/cartesian/kernel_cartesian.py
(ya incluye los 2 fixes de Kaggle encontrados ahí: os.chdir para NPCDataset,
loteo del tronco en init_components). Ver DESIGN.md para el diseño del barrido.
"""

import glob
import json
import os
import subprocess
import sys
import time

WORK = "/kaggle/working"
NPC_ROOT = f"{WORK}/npc"
CONDITION = "search-confirm_seed42"
FINAL_MODE = "cartesian"  # exp_02 ya decidió Cartesian; este exp solo ajusta hiperparámetros
SEED = 42
EPOCHS = 15
BATCH_SIZE = 256
N_COMP_HEAD = 200
N_COMP_FINAL = 380
LR_KDM = 0.003
SIGMA_MULT = 0.5
RESULTS = f"{WORK}/results/{CONDITION}"
EXPECTED_GLOBAL_SHA256 = "4d640365ca4b505d14ae79a83dcdc799d546a121a676e0a26a01c42fb2b9e07d"

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
        sys.exit(f"[FATAL] Falló: {' '.join(cmd)}")


# --------------------------------------------------------------------------
# 1. Entorno: mismo fix que exp_01/exp_02 (PyPI puro, torch==2.2.2 por Python
#    3.12 de Kaggle) + kdm-torch (satisface su propio torch>=2.2)
# --------------------------------------------------------------------------
t_env = time.time()
run([sys.executable, "-m", "pip", "install", "-q", "torch==2.2.2", "torchvision==0.17.2"])
run([sys.executable, "-m", "pip", "install", "-q",
     "numpy<2", "natsort==8.0.2", "PyQt5==5.15.11",  # header.py -> type.py de npc-* exige PyQt5
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
# 3. Dataset congelado -> jerarquía oficial (descubrimiento dinámico del mount)
# --------------------------------------------------------------------------
_manifest_matches = glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True)
assert len(_manifest_matches) == 1, f"Se esperaba 1 MANIFEST.json: {_manifest_matches}"
INPUT_DIR = os.path.dirname(_manifest_matches[0])
print(f"[INFO] Dataset montado en: {INPUT_DIR}", flush=True)

t_data = time.time()
processed = f"{NPC_ROOT}/datasets/mnist/instances/processed"
os.makedirs(os.path.dirname(processed), exist_ok=True)
os.symlink(f"{INPUT_DIR}/mnist_addition_processed", processed)
assert sum(len(fs) for _, _, fs in os.walk(processed)) == 35000

with open(f"{INPUT_DIR}/MANIFEST.json") as f:
    manifest = json.load(f)
assert manifest["global_sha256"] == EXPECTED_GLOBAL_SHA256

import shutil, gzip  # noqa: E402
cfg_dir = f"{NPC_ROOT}/npc-dataset-utils/configs/npc-dataset-utils"
shutil.copy(f"{INPUT_DIR}/mnist.json", cfg_dir)
with open(f"{INPUT_DIR}/mnist_split.json", "rb") as f_in:
    with gzip.open(f"{cfg_dir}/mnist_split.json.gz", "wb") as f_out:
        f_out.write(f_in.read())
WALLCLOCK["data_setup"] = round(time.time() - t_data, 1)

for header_path in (f"{NPC_ROOT}/npc-dataset-utils/src/npc-dataset-utils/header.py",
                    f"{NPC_ROOT}/npc-models/src/npc-models/header.py"):
    with open(header_path) as f:
        content = f.read()
    patched = content.replace('dataset_prefix = "awa2"', 'dataset_prefix = "mnist"')
    assert patched != content, header_path
    with open(header_path, "w") as f:
        f.write(patched)

utils_src = f"{NPC_ROOT}/npc-dataset-utils/src/npc-dataset-utils"
run([sys.executable, "split.py"], cwd=utils_src, name="split")

for split_name, expected in (("train", 28000), ("validate", 3500), ("test", 3500)):
    n = sum(len(fs) for _, _, fs in os.walk(f"{NPC_ROOT}/datasets/mnist/splits/instances/{split_name}"))
    assert n == expected, f"Split {split_name}: {n} != {expected}"

# --------------------------------------------------------------------------
# 4. KDMCascade (copia embebida de src/models/kdm_cascade.py, solo variante
#    Cartesian -- exp_02 ya descartó Distributional)
# --------------------------------------------------------------------------
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision  # noqa: E402

from kdm.layers import CosineKernelLayer, KDMLayer  # noqa: E402
from kdm.init import init_kdm_layer  # noqa: E402
from kdm.models import KDMClassModel  # noqa: E402
from kdm.utils import cartesian_product, dm2discrete, pure2dm  # noqa: E402

RESNET_NECK_SIZE = 512
N_DIGIT_VALUES = 10
N_SUM_CLASSES = 19


def build_shared_trunk() -> nn.Module:
    resnet = torchvision.models.resnet34(weights="IMAGENET1K_V1")
    resnet.fc = nn.Identity()
    return resnet


class KDMCascadeCartesian(nn.Module):
    def __init__(self, n_comp_head, n_comp_final, sigma_head=1.0):
        super().__init__()
        self.n_comp_head = n_comp_head
        self.n_comp_final = n_comp_final
        self.trunk = build_shared_trunk()
        self.head1 = KDMClassModel(RESNET_NECK_SIZE, N_DIGIT_VALUES, nn.Identity(), n_comp_head, sigma=sigma_head)
        self.head2 = KDMClassModel(RESNET_NECK_SIZE, N_DIGIT_VALUES, nn.Identity(), n_comp_head, sigma=sigma_head)
        self.kdm_final = KDMLayer(kernel=CosineKernelLayer(),
                                  dim_x=N_DIGIT_VALUES * N_DIGIT_VALUES, dim_y=N_SUM_CLASSES,
                                  n_comp=n_comp_final)

    def forward(self, image):
        neck = self.trunk(image)
        p1 = self.head1(neck)
        p2 = self.head2(neck)
        rho_x = pure2dm(cartesian_product([p1, p2]))
        p_sum = dm2discrete(self.kdm_final(rho_x))
        return p1, p2, p_sum

    @torch.no_grad()
    def init_components(self, images, digit1, digit2, sum_class, forward_batch_size=256, sigma_mult=1.0):
        # correr el tronco en mini-lotes -- una sola pasada de miles de
        # imagenes 224x224 por ResNet-34 agota memoria de GPU (CUDA OOM)
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
                assert len(candidates) >= per_value, f"valor {value}: hay {len(candidates)}, se necesitan {per_value}"
                chosen.append(candidates[torch.randperm(len(candidates))[:per_value]])
            return torch.cat(chosen)

        idx1 = stratified_idx(digit1, N_DIGIT_VALUES, self.n_comp_head)
        init_kdm_layer(self.head1.kdm, neck[idx1],
                       F.one_hot(digit1[idx1], N_DIGIT_VALUES).float(), init_sigma=True, sigma_mult=sigma_mult)
        idx2 = stratified_idx(digit2, N_DIGIT_VALUES, self.n_comp_head)
        init_kdm_layer(self.head2.kdm, neck[idx2],
                       F.one_hot(digit2[idx2], N_DIGIT_VALUES).float(), init_sigma=True, sigma_mult=sigma_mult)

        idx_f = stratified_idx(sum_class, N_SUM_CLASSES, self.n_comp_final)
        y_f = F.one_hot(sum_class[idx_f], N_SUM_CLASSES).float()
        true_d1 = F.one_hot(digit1[idx_f], N_DIGIT_VALUES).float()
        true_d2 = F.one_hot(digit2[idx_f], N_DIGIT_VALUES).float()
        x_f = cartesian_product([true_d1, true_d2])
        init_kdm_layer(self.kdm_final, x_f, y_f, init_sigma=True, sigma_mult=sigma_mult)


# --------------------------------------------------------------------------
# 5. Datos: NPCDataset (npc-models) sobre los splits ya materializados
# --------------------------------------------------------------------------
models_src = f"{NPC_ROOT}/npc-models/src/npc-models"
sys.path.insert(0, models_src)
# header.py resuelve dataset_config_file_path como ruta RELATIVA asumiendo
# cwd=models_src -- hace falta cambiar el cwd a mano antes de instanciar
# NPCDataset (bug encontrado y corregido en exp_02).
os.chdir(models_src)
from dataset import NPCDataset  # noqa: E402

tfm = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor(),
])
dataset_dir = f"{NPC_ROOT}/datasets/mnist/splits/instances"
ds_train = NPCDataset(f"{dataset_dir}/train", tfm)
ds_test = NPCDataset(f"{dataset_dir}/test", tfm)

torch.manual_seed(SEED)
device = torch.device("cuda")
model = KDMCascadeCartesian(n_comp_head=N_COMP_HEAD, n_comp_final=N_COMP_FINAL).to(device)

# --------------------------------------------------------------------------
# 6. Inicialización (batch estratificado grande; cada clase-suma tiene
#    >=280 instancias en el split de train, alcanza para estratificar)
# --------------------------------------------------------------------------
t_init = time.time()
init_loader = torch.utils.data.DataLoader(ds_train, batch_size=3000, shuffle=True, num_workers=2)
init_images, init_attrs, init_class, _ = next(iter(init_loader))
init_d1 = init_attrs[0].argmax(dim=1)
init_d2 = init_attrs[1].argmax(dim=1)
model.init_components(init_images.to(device), init_d1.to(device), init_d2.to(device), init_class.to(device),
                      sigma_mult=SIGMA_MULT)
WALLCLOCK["init_components"] = round(time.time() - t_init, 1)
print(f"[INFO] init_components: {WALLCLOCK['init_components']}s", flush=True)

# --------------------------------------------------------------------------
# 7. Entrenamiento end-to-end
# --------------------------------------------------------------------------
loader_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=4, pin_memory=True)
loader_test = torch.utils.data.DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False,
                                          num_workers=4, pin_memory=True)

opt_trunk = torch.optim.SGD(model.trunk.parameters(), lr=0.01, momentum=0.9)
kdm_params = [p for n, p in model.named_parameters() if not n.startswith("trunk.")]
opt_kdm = torch.optim.Adam(kdm_params, lr=LR_KDM)

t_train = time.time()
epoch_times = []
loss_history = []
for epoch in range(1, EPOCHS + 1):
    t_epoch = time.time()
    model.train()
    epoch_loss = 0.0
    for images, attrs, classes, _ in loader_train:
        images, classes = images.to(device, non_blocking=True), classes.to(device, non_blocking=True)
        d1 = attrs[0].to(device, non_blocking=True)
        d2 = attrs[1].to(device, non_blocking=True)

        opt_trunk.zero_grad(); opt_kdm.zero_grad()
        p1, p2, p_sum = model(images)
        loss_sum = F.nll_loss(torch.log(p_sum + 1e-8), classes)
        loss_attr = -(d1 * torch.log(p1 + 1e-8)).sum(dim=1).mean() - (d2 * torch.log(p2 + 1e-8)).sum(dim=1).mean()
        loss = loss_sum + 0.5 * loss_attr
        loss.backward()
        opt_trunk.step(); opt_kdm.step()
        epoch_loss += loss.item() * images.size(0)

    epoch_loss /= len(ds_train)
    dt_epoch = time.time() - t_epoch
    epoch_times.append(dt_epoch)
    loss_history.append(round(epoch_loss, 4))
    print(f"[INFO] epoca {epoch}/{EPOCHS}  loss={epoch_loss:.4f}  ({dt_epoch:.0f}s)", flush=True)

WALLCLOCK["train_total"] = round(time.time() - t_train, 1)
WALLCLOCK["train_per_epoch_mean"] = round(sum(epoch_times) / len(epoch_times), 1)

# --------------------------------------------------------------------------
# 8. Evaluación en test: accuracy end-to-end + accuracy/TV por dígito
# --------------------------------------------------------------------------
t_eval = time.time()
model.eval()
n_correct_sum = 0
n_correct_joint_attr = 0
tv_d1_total = tv_d2_total = 0.0
n_total = 0
with torch.no_grad():
    for images, attrs, classes, _ in loader_test:
        images, classes = images.to(device), classes.to(device)
        d1_true, d2_true = attrs[0].to(device), attrs[1].to(device)

        p1, p2, p_sum = model(images)
        n_correct_sum += (p_sum.argmax(dim=1) == classes).sum().item()
        correct1 = p1.argmax(dim=1) == d1_true.argmax(dim=1)
        correct2 = p2.argmax(dim=1) == d2_true.argmax(dim=1)
        n_correct_joint_attr += (correct1 & correct2).sum().item()
        tv_d1_total += (0.5 * (p1 - d1_true).abs().sum(dim=1)).sum().item()
        tv_d2_total += (0.5 * (p2 - d2_true).abs().sum(dim=1)).sum().item()
        n_total += images.size(0)

metrics_eval = {
    "classification_accuracy": n_correct_sum / n_total,
    "attribute_joint_accuracy": n_correct_joint_attr / n_total,
    "mean_tv_distance": 0.5 * (tv_d1_total / n_total + tv_d2_total / n_total),
}
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
    "experiment": "exp_03_mnist_kdm_sweep",
    "condition": CONDITION,
    "final_mode": "cartesian",
    "seed": SEED,
    "hyperparameters": {
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "n_comp_head": N_COMP_HEAD, "n_comp_final": N_COMP_FINAL,
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
    "loss_history": loss_history,
    "metrics": metrics_eval,
}
with open(f"{RESULTS}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

shutil.rmtree(NPC_ROOT, ignore_errors=True)
print("\n[DONE] metrics.json:", flush=True)
print(json.dumps(metrics, indent=2), flush=True)
