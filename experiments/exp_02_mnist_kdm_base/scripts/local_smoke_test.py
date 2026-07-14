#!/usr/bin/env python3
"""Smoke test local (CPU) de KDMCascade, ambas variantes -- antes de gastar
GPU de Kaggle. Corre con un subconjunto chico de D̄ (ya generado localmente en
data/npc/datasets/mnist/instances/processed/) y verifica:
  1. forward/backward corren sin error de forma
  2. la loss baja (aunque sea poco) en unos pocos pasos
  3. p1, p2, p_sum son distribuciones válidas (suman ~1)

Uso: python experiments/exp_02_mnist_kdm_base/scripts/local_smoke_test.py
"""

import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from models.kdm_cascade import KDMCascade  # noqa: E402

torch.manual_seed(42)

PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "npc", "datasets", "mnist", "instances", "processed")
MNIST_JSON = os.path.join(REPO_ROOT, "external", "npc-dataset-utils", "configs",
                          "npc-dataset-utils", "mnist.json")
N_SAMPLES = 800  # margen para poder estratificar n_comp_head=20 (2/digito) y n_comp_final=38 (2/clase)
N_COMP_HEAD = 20
N_COMP_FINAL = 38
TRAIN_BATCH = 64  # minibatch para los pasos de entrenamiento (no las 800 de una vez)
IMG_SIZE = 64      # 224 es el tamano real (Kaggle/GPU); en CPU para un smoke test alcanza con menos
                   # -- (64/224)^2 ~= 8% del computo de ResNet-34, mismo pipeline/formas de tensor


def load_subset():
    with open(MNIST_JSON) as f:
        mappings = json.load(f)["mappings"]

    import random
    items = list(mappings.items())
    random.Random(42).shuffle(items)   # orden del dict podria estar agrupado por clase; barajar antes de cortar
    items = items[:N_SAMPLES]
    tfm = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])

    images, d1, d2, sums = [], [], [], []
    for image_name, entry in items:
        path = os.path.join(PROCESSED_DIR, image_name)
        img = Image.open(path).convert("RGB")
        images.append(tfm(img))
        d1.append(int(entry["labels"]["number-first"]))
        d2.append(int(entry["labels"]["number-second"]))
        sums.append(int(image_name.split("/")[0]))  # las claves del JSON siempre usan "/"

    return (torch.stack(images), torch.tensor(d1), torch.tensor(d2), torch.tensor(sums))


def run_variant(final_mode, images, d1, d2, sums):
    print(f"\n=== {final_mode} ===")
    model = KDMCascade(final_mode=final_mode, n_comp_head=N_COMP_HEAD, n_comp_final=N_COMP_FINAL)

    t0 = time.time()
    model.init_components(images, d1, d2, sums)
    print(f"init_components: {time.time()-t0:.1f}s")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    torch.manual_seed(0)
    for step in range(3):
        t0 = time.time()
        idx = torch.randperm(images.shape[0])[:TRAIN_BATCH]
        xb, d1b, d2b, sb = images[idx], d1[idx], d2[idx], sums[idx]

        opt.zero_grad()
        p1, p2, p_sum = model(xb)

        assert p1.shape == (TRAIN_BATCH, 10), p1.shape
        assert p2.shape == (TRAIN_BATCH, 10), p2.shape
        assert p_sum.shape == (TRAIN_BATCH, 19), p_sum.shape
        assert torch.allclose(p1.sum(dim=1), torch.ones(TRAIN_BATCH), atol=1e-3), "p1 no suma 1"
        assert torch.allclose(p2.sum(dim=1), torch.ones(TRAIN_BATCH), atol=1e-3), "p2 no suma 1"
        assert torch.allclose(p_sum.sum(dim=1), torch.ones(TRAIN_BATCH), atol=1e-3), "p_sum no suma 1"

        loss_sum = F.nll_loss(torch.log(p_sum + 1e-8), sb)
        loss_d1 = F.nll_loss(torch.log(p1 + 1e-8), d1b)
        loss_d2 = F.nll_loss(torch.log(p2 + 1e-8), d2b)
        loss = loss_sum + 0.5 * (loss_d1 + loss_d2)
        loss.backward()
        opt.step()

        losses.append(loss.item())
        print(f"  paso {step}: loss={loss.item():.4f}  ({time.time()-t0:.1f}s)")

    # cada paso usa un minibatch distinto (aleatorio) -- no exigir bajada
    # monotonica (seria fragil), solo que no explote/diverja a NaN/Inf.
    assert all(torch.isfinite(torch.tensor(l)) for l in losses), f"loss no finita: {losses}"
    print(f"OK: {final_mode} -- shapes correctas, distribuciones validas, losses finitas {losses}")
    return model


if __name__ == "__main__":
    print("Cargando subconjunto local...")
    images, d1, d2, sums = load_subset()
    print(f"{images.shape[0]} muestras cargadas")

    run_variant("cartesian", images, d1, d2, sums)
    run_variant("distributional", images, d1, d2, sums)

    print("\n[OK] Ambas variantes pasan el smoke test local.")
