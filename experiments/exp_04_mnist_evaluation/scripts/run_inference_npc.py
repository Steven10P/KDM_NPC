#!/usr/bin/env python3
"""Corre inferencia local (CPU) de NPC (ambas variantes: Knowledge y Data,
seed 42) sobre el split de test congelado de MNIST-Addition, reusando el
codigo NATIVO de npc-models (test_npc.test) en vez de reimplementar la logica
de evaluacion.

Con header.npc_interpret = True, test_npc.test() activa sus propios
mecanismos de interpretabilidad ya implementados por los autores de NPC:
  - MPE (Most Probable Explanation): para cada prediccion, encuentra la
    combinacion de atributos que maximiza matrix_pc * matrix_neural (i.e.
    inferencia MAP EXACTA sobre el circuito) y mide si esa explicacion
    coincide con los atributos reales (findMPE/computeMPEAlignment).
  - CE (Contrastive/Counterfactual Explanation): para las predicciones
    incorrectas, ajusta las probabilidades de atributo via gradiente hasta
    que la prediccion final cambia a la clase correcta, y mide la tasa de
    correccion (findCE).
Esto escribe interpret.json con el detalle por muestra.

Ademas, en un segundo paso liviano (mismo forward del neural + PC, sin
gradientes) se captura la distribucion COMPLETA de 19 clases por muestra
(test_npc.computeNPCOutput ya hace esto internamente pero solo se usa el
argmax en test(); interpret.json solo guarda el top-1) -- necesaria para
matriz de confusion / ROC / precision-recall en build_plots.py.
"""
import json
import os
import shutil
import sys

import numpy as np
import torch
import torchvision

# utility.loadCheckpoint (vendorizado, npc-models) llama torch.load sin
# map_location -- los checkpoints se guardaron desde CUDA en Kaggle, y aqui no
# hay GPU. Forzamos CPU por defecto para todo el proceso (no tocamos el
# archivo vendorizado).
_orig_torch_load = torch.load


def _torch_load_cpu(*args, **kwargs):
    kwargs.setdefault("map_location", torch.device("cpu"))
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_cpu

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NPC_MODELS_SRC = os.path.join(REPO_ROOT, "external", "npc-models", "src", "npc-models")
LEARNSPN_OUT = os.path.join(REPO_ROOT, "external", "learnspn", "outputs")
TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "data_local", "mnist_test")
MNIST_JSON = os.path.join(REPO_ROOT, "external", "npc-dataset-utils", "configs",
                          "npc-dataset-utils", "mnist.json")
EXP01_RESULTS = os.path.join(REPO_ROOT, "experiments", "exp_01_mnist_npc_repro", "results")
OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
STAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_local", "_checkpoints_staging")

VARIANTS = {
    "knowledge": {
        "neural_ckpt": "42.mnist.neural.resnet34mtl.2026.7.14.12.39.76be79a3b6c3.best.zip",
        "pc_ckpt": "42.mnist.pc.pgd.2026.7.14.12.39.76be79a3b6c3.best.zip",
        "ckpt_dir": os.path.join(EXP01_RESULTS, "npc-knowledge_seed42", "checkpoints"),
        "file_path_pc": os.path.join(LEARNSPN_OUT, "manual", "mnist.spn.txt"),
    },
    "data": {
        "neural_ckpt": "42.mnist.neural.resnet34mtl.2026.7.14.12.42.a19eb962fbb9.best.zip",
        "pc_ckpt": "42.mnist.pc.pgd.2026.7.14.12.42.a19eb962fbb9.best.zip",
        "ckpt_dir": os.path.join(EXP01_RESULTS, "npc-data_seed42", "checkpoints"),
        "file_path_pc": os.path.join(LEARNSPN_OUT, "learnspn", "mnist.spn.txt"),
    },
}

sys.path.insert(0, NPC_MODELS_SRC)

import header  # noqa: E402
header.dataset_prefix = "mnist"
header.dataset_config_file_path = MNIST_JSON
header.npc_pc_cpu = True
header.npc_interpret = True

import wandb  # noqa: E402
import utility  # noqa: E402
import model  # noqa: E402
import pc  # noqa: E402
import test_npc  # noqa: E402
from dataset import NPCDataset  # noqa: E402


def load_test_dataset():
    tfm = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])
    # Mismo fix de separador de rutas que run_inference_kdm.py (Windows "\\"
    # vs las claves "/" de mnist.json["mappings"], generadas en Linux).
    _orig_join = os.path.join
    os.path.join = lambda *a: _orig_join(*a).replace(os.sep, "/")
    try:
        ds_test = NPCDataset(os.path.abspath(TEST_DIR), tfm)
    finally:
        os.path.join = _orig_join
    return ds_test


def stage_checkpoints(variant_name, cfg):
    stage_dir = os.path.join(STAGE_DIR, variant_name)
    os.makedirs(stage_dir, exist_ok=True)
    for fname in (cfg["neural_ckpt"], cfg["pc_ckpt"]):
        dst = os.path.join(stage_dir, fname)
        if not os.path.isfile(dst):
            shutil.copy(os.path.join(cfg["ckpt_dir"], fname), dst)
    return stage_dir


def run_variant(variant_name, cfg, ds_test, device):
    print(f"\n{'=' * 70}\n[INFO] variante: {variant_name}\n{'=' * 70}")
    out_dir = os.path.join(OUT_ROOT, f"npc_{variant_name}_seed42")
    os.makedirs(out_dir, exist_ok=True)

    stage_dir = stage_checkpoints(variant_name, cfg)
    header.checkpoint_dir = stage_dir
    header.config_neural["file_name_checkpoint_best"] = cfg["neural_ckpt"]
    header.config_pc["run_name"] = f"{variant_name}-seed42"  # solo necesita ser no-vacio
    header.config_pc["file_name_checkpoint_best"] = cfg["pc_ckpt"]
    header.config_pc["file_path_pc"] = cfg["file_path_pc"]
    header.project_dir_outputs_interpret = out_dir

    data_loader_test = torch.utils.data.DataLoader(
        ds_test, batch_size=64, shuffle=False, num_workers=0)

    # Igual que test_npc.main(): DataParallel para que los nombres de
    # parametros del checkpoint ("module.xxx") calcen al cargar.
    model_neural = model.ResNet34MTL(ds_test.config, device)
    model_neural = torch.nn.DataParallel(model_neural)
    model_neural = model_neural.to(device)

    pc_joint = pc.ProbabilisticCircuit(device)
    pc_marginal = pc.ProbabilisticCircuit(device)

    print(f"[INFO] cargando estructura del circuito: {cfg['file_path_pc']}")
    pc_joint.load(cfg["file_path_pc"])
    pc_marginal.load(cfg["file_path_pc"])
    print(f"[INFO] nodos: total={len(pc_joint.nodes)} suma={len(pc_joint.sum_nodes)} "
          f"producto={len(pc_joint.product_nodes)} hoja={len(pc_joint.leaf_nodes)} "
          f"profundidad={pc_joint.depth}")

    pc_settings_joint = test_npc.generatePCSettings(ds_test.config, device)
    pc_settings_marginal = torch.clone(pc_settings_joint)
    pc_settings_marginal[:, -1] = -1
    pc_joint.set_leaf_nodes_categorical(pc_settings_joint)
    pc_marginal.set_leaf_nodes_categorical(pc_settings_marginal)

    with open(os.path.join(out_dir, "circuit_structure.json"), "w") as f:
        json.dump({
            "variant": variant_name,
            "file_path_pc": cfg["file_path_pc"],
            "n_nodes_total": len(pc_joint.nodes),
            "n_sum_nodes": len(pc_joint.sum_nodes),
            "n_product_nodes": len(pc_joint.product_nodes),
            "n_leaf_nodes": len(pc_joint.leaf_nodes),
            "depth": pc_joint.depth,
        }, f, indent=2)

    wandb.init(config={"neural": header.config_neural, "pc": header.config_pc},
               mode="disabled", reinit=True)

    # --- corre la evaluacion NATIVA de NPC (carga los checkpoints ella misma,
    # calcula accuracy/TV, y con npc_interpret=True escribe interpret.json con
    # MPE alignment + CE correction) ---
    test_npc.test(model_neural, pc_joint, pc_marginal, pc_settings_joint,
                  data_loader_test, device, 1)

    # --- segundo paso liviano: capturar la distribucion completa de 19 clases
    # por muestra (test() solo usa el argmax internamente) ---
    model_neural.eval()
    pc_output_rows = len(data_loader_test.dataset.labels_class)
    pc_output_cols = 1
    for attribute in data_loader_test.dataset.config["attributes"]:
        pc_output_cols *= len(attribute["labels"])

    all_output_npc, all_sum_true, all_d1_true, all_d2_true, all_paths = [], [], [], [], []
    with torch.no_grad():
        for images, labels_attribute, labels_class, paths in data_loader_test:
            (outputs_attribute_original, _) = model_neural(images)
            outputs_attribute = utility.applySoftmaxAttribute(outputs_attribute_original)
            (_, _, output_npc) = test_npc.computeNPCOutput(
                outputs_attribute, pc_joint, pc_marginal,
                pc_output_rows, pc_output_cols, device)

            all_output_npc.append(output_npc.numpy())
            all_sum_true.append(labels_class.numpy())
            all_d1_true.append(labels_attribute[0].argmax(dim=1).numpy())
            all_d2_true.append(labels_attribute[1].argmax(dim=1).numpy())
            all_paths.extend(paths)

    output_npc = np.concatenate(all_output_npc)
    sum_true = np.concatenate(all_sum_true)
    d1_true = np.concatenate(all_d1_true)
    d2_true = np.concatenate(all_d2_true)

    acc_recomputed = (output_npc.argmax(axis=1) == sum_true).mean()
    print(f"[INFO] classification_accuracy (recomputada localmente): {acc_recomputed:.6f}")

    np.savez(
        os.path.join(out_dir, "predictions.npz"),
        p_sum=output_npc, sum_true=sum_true, d1_true=d1_true, d2_true=d2_true,
        paths=np.array(all_paths),
    )

    interpret_path = os.path.join(out_dir, f"{header.dataset_prefix}.json")
    n_interpret = None
    if os.path.isfile(interpret_path):
        with open(interpret_path) as f:
            n_interpret = len(json.load(f))
    print(f"[DONE] variante {variant_name}: interpret.json con {n_interpret} muestras, "
          f"predictions.npz con {output_npc.shape[0]} muestras -> {out_dir}")


def main():
    device = torch.device("cpu")
    ds_test = load_test_dataset()
    print(f"[INFO] test set: {len(ds_test)} imagenes")

    for variant_name, cfg in VARIANTS.items():
        run_variant(variant_name, cfg, ds_test, device)


if __name__ == "__main__":
    main()
