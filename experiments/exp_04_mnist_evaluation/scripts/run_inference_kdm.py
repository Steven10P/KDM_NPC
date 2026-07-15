#!/usr/bin/env python3
"""Corre inferencia local (CPU) del KDMCascade (variante Cartesian, exp_03
final-seed42) sobre el split de test congelado de MNIST-Addition, y extrae:

  - p_sum (19-dim), p1/p2 (10-dim cada uno) por muestra -> para matriz de
    confusion / ROC / PR en build_plots.py.
  - Component attribution del kdm_final: el peso normalizado por componente
    (Ec. 12 de kdm/layers/kdm_layer.py::forward, linea 78-85) que se usa para
    combinar los 190 prototipos aprendidos en la prediccion final. Como
    final_mode="cartesian" usa un estado puro (n_comp_in=1), el peso
    normalizado ES literalmente la salida de forward() antes de mezclar con
    c_y -- no hace falta reimplementar nada, solo llamar _compute_mixture y
    normalizar igual que el forward real.
  - Tabla global de los 190 componentes: digit-pair mas cercano (argmax de
    c_x, que vive en la base one-hot de 100-dim del producto cartesiano) y
    clase-suma implicita de c_y -- interpretacion "que representa cada
    prototipo aprendido", independiente de cualquier muestra de test.

No requiere GPU -- ~3500 imagenes 224x224 por ResNet-34 en CPU, un solo pase.
"""
import json
import os
import sys

import numpy as np
import torch
import torchvision

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NPC_MODELS_SRC = os.path.join(REPO_ROOT, "external", "npc-models", "src", "npc-models")
TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "data_local", "mnist_test")
MNIST_JSON = os.path.join(REPO_ROOT, "external", "npc-dataset-utils", "configs",
                          "npc-dataset-utils", "mnist.json")
CHECKPOINT = os.path.join(REPO_ROOT, "experiments", "exp_03_mnist_kdm_sweep",
                          "results", "final-seed42", "checkpoints", "model.pt")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "kdm_final-seed42")

sys.path.insert(0, NPC_MODELS_SRC)
sys.path.insert(0, REPO_ROOT)

import header  # noqa: E402  (npc-models header, solo para NPCDataset)
header.dataset_config_file_path = MNIST_JSON

from dataset import NPCDataset  # noqa: E402

from src.models.kdm_cascade import KDMCascade  # noqa: E402


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cpu")

    tfm = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])
    # dataset.py arma las claves de busqueda en mnist.json["mappings"] con
    # os.path.join(class_name, file_name) -- en Linux (Kaggle) da "8/archivo.png"
    # (coincide con las claves del JSON), pero en Windows da "8\archivo.png" y
    # falla el lookup. Windows acepta "/" en rutas de archivo igual que "\", asi
    # que forzamos os.path.join a usar "/" solo durante la construccion del
    # dataset, sin tocar el codigo vendorizado de npc-models.
    _orig_join = os.path.join
    os.path.join = lambda *a: _orig_join(*a).replace(os.sep, "/")
    try:
        ds_test = NPCDataset(os.path.abspath(TEST_DIR), tfm)
    finally:
        os.path.join = _orig_join
    loader = torch.utils.data.DataLoader(ds_test, batch_size=64, shuffle=False, num_workers=0)
    print(f"[INFO] test set: {len(ds_test)} imagenes")

    model = KDMCascade(final_mode="cartesian", n_comp_head=100, n_comp_final=190)
    state_dict = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] checkpoint cargado: {CHECKPOINT}")

    all_p1, all_p2, all_psum = [], [], []
    all_d1_true, all_d2_true, all_sum_true = [], [], []
    all_paths = []
    all_attribution = []  # (n_comp_final,) por muestra

    from kdm.utils import cartesian_product, pure2dm  # noqa: E402

    with torch.no_grad():
        for batch_idx, (images, attrs, classes, paths) in enumerate(loader):
            d1_true = attrs[0].argmax(dim=1)
            d2_true = attrs[1].argmax(dim=1)

            p1, p2, p_sum = model(images)

            joint = cartesian_product([p1, p2])
            rho_x = pure2dm(joint)
            in_w, out_w = model.kdm_final._compute_mixture(rho_x)
            out_w = out_w.clamp(min=model.kdm_final.eps)
            out_w = out_w / out_w.sum(dim=2, keepdim=True)
            attribution = out_w[:, 0, :]  # (bs, n_comp_final) -- n_comp_in=1 (estado puro)

            all_p1.append(p1.numpy())
            all_p2.append(p2.numpy())
            all_psum.append(p_sum.numpy())
            all_d1_true.append(d1_true.numpy())
            all_d2_true.append(d2_true.numpy())
            all_sum_true.append(classes.numpy())
            all_paths.extend(paths)
            all_attribution.append(attribution.numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"[INFO] batch {batch_idx + 1}/{len(loader)}")

    p1 = np.concatenate(all_p1)
    p2 = np.concatenate(all_p2)
    p_sum = np.concatenate(all_psum)
    d1_true = np.concatenate(all_d1_true)
    d2_true = np.concatenate(all_d2_true)
    sum_true = np.concatenate(all_sum_true)
    attribution = np.concatenate(all_attribution)

    acc_sum = (p_sum.argmax(axis=1) == sum_true).mean()
    acc_attr = ((p1.argmax(axis=1) == d1_true) & (p2.argmax(axis=1) == d2_true)).mean()
    print(f"[INFO] classification_accuracy (recomputada localmente): {acc_sum:.6f}")
    print(f"[INFO] attribute_joint_accuracy (recomputada localmente): {acc_attr:.6f}")

    np.savez(
        os.path.join(OUT_DIR, "predictions.npz"),
        p1=p1, p2=p2, p_sum=p_sum,
        d1_true=d1_true, d2_true=d2_true, sum_true=sum_true,
        attribution=attribution,
        paths=np.array(all_paths),
    )

    # -- tabla global: que representa cada uno de los 190 componentes del KDM final
    c_x = model.kdm_final.c_x.detach()          # (190, 100)
    c_y = model.kdm_final.c_y.detach()          # (190, 19)
    c_w = model.kdm_final._normalized_comp_w().detach()  # (190,)
    nearest_pair_idx = c_x.argmax(dim=1).numpy()          # 0..99 -> (i,j) = divmod(idx, 10)
    implied_sum_class = c_y.argmax(dim=1).numpy()         # 0..18

    components_table = []
    for c in range(c_x.shape[0]):
        i, j = divmod(int(nearest_pair_idx[c]), 10)
        components_table.append({
            "component": c,
            "nearest_digit_pair": [i, j],
            "nearest_pair_implied_sum": i + j,
            "c_y_argmax_sum_class": int(implied_sum_class[c]),
            "prior_weight_c_w": float(c_w[c]),
        })

    with open(os.path.join(OUT_DIR, "components_table.json"), "w") as f:
        json.dump(components_table, f, indent=2)

    summary = {
        "condition": "kdm_final-seed42",
        "checkpoint": CHECKPOINT,
        "n_test": len(ds_test),
        "classification_accuracy": float(acc_sum),
        "attribute_joint_accuracy": float(acc_attr),
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] resultados en {OUT_DIR}")


if __name__ == "__main__":
    main()
