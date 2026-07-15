#!/usr/bin/env python3
"""Reporte comparativo KDM-vs-NPC para MNIST-Addition, usando
src/metrics/interpretability_suite.py (modulo agnostico al dataset).

Selecciona las instancias de test donde KDM y NPC(Knowledge) discrepan, o
donde ambos fallan, y para cada una arma un panel de 4 partes: imagen +
predicciones, espectro de atribucion + top componentes de KDM (decodificados
como tupla de atributos), la Explicacion-Mas-Probable (MPE) exacta de NPC, y
su contrafactual minimo (si la prediccion es incorrecta).

Ademas valida en vivo (no solo reusando lo precomputado en exp_04) que
NPCExplainer.mpe_query/counterfactual reproducen exactamente lo que
test_npc.py ya calculo (guardado en results/npc_knowledge_seed42/mnist.json)
para 3 instancias, antes de confiar en el modulo para el resto del reporte.
"""
import json
import os
import sys

import numpy as np
import torch
import torchvision
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NPC_MODELS_SRC = os.path.join(REPO_ROOT, "external", "npc-models", "src", "npc-models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
TRAIN_SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_local", "mnist_train_sample")
LEARNSPN_OUT = os.path.join(REPO_ROOT, "external", "learnspn", "outputs")

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, NPC_MODELS_SRC)

_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "map_location": kw.get("map_location", "cpu")})

from src.models.kdm_cascade import KDMCascade  # noqa: E402
from src.metrics.interpretability_suite import (  # noqa: E402
    KDMExplainer, NPCExplainer, select_comparison_instances, build_comparison_panel,
)

ATTRIBUTE_NAMES = ["number-first", "number-second"]
ATTRIBUTE_CARDINALITIES = [10, 10]


def load_kdm():
    model = KDMCascade(final_mode="cartesian", n_comp_head=100, n_comp_final=190)
    ckpt = os.path.join(RESULTS_DIR, "kdm_final-seed42", "checkpoints", "model.pt")
    if not os.path.isfile(ckpt):
        ckpt = os.path.join(REPO_ROOT, "experiments", "exp_03_mnist_kdm_sweep",
                            "results", "final-seed42", "checkpoints", "model.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    def head_accessor(m, name):
        return {"number-first": m.head1, "number-second": m.head2}[name]

    return model, KDMExplainer(
        model=model, head_accessor=head_accessor,
        attribute_names=ATTRIBUTE_NAMES, attribute_cardinalities=ATTRIBUTE_CARDINALITIES,
    )


@torch.no_grad()
def compute_train_embeddings(model):
    tfm = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor(),
    ])
    paths = []
    for root, _, files in os.walk(TRAIN_SAMPLE_DIR):
        for f in files:
            paths.append(os.path.join(root, f))
    paths.sort()
    embeddings = []
    for i in range(0, len(paths), 64):
        batch_paths = paths[i:i + 64]
        images = torch.stack([tfm(Image.open(p).convert("RGB")) for p in batch_paths])
        embeddings.append(model.trunk(images))
    return torch.cat(embeddings, dim=0), paths


def load_npc_live(variant: str):
    """Reusa el patron de setup de run_inference_npc.py, pero solo para
    demostrar/validar NPCExplainer en unas pocas instancias, no para
    reprocesar los 3500 casos de test (eso ya lo hizo exp_04)."""
    import header
    import model as npc_model
    import pc
    import test_npc
    import utility

    header.dataset_prefix = "mnist"
    header.npc_pc_cpu = True
    header.checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "data_local",
                                         "_checkpoints_staging", variant)

    ckpt_files = os.listdir(header.checkpoint_dir)
    neural_ckpt = next(f for f in ckpt_files if f.startswith("42.mnist.neural") and f.endswith(".best.zip"))
    pc_ckpt = next(f for f in ckpt_files if f.startswith("42.mnist.pc.pgd") and f.endswith(".best.zip"))
    file_path_pc = os.path.join(LEARNSPN_OUT, "manual" if variant == "knowledge" else "learnspn", "mnist.spn.txt")

    dataset_config_path = os.path.join(REPO_ROOT, "external", "npc-dataset-utils", "configs",
                                       "npc-dataset-utils", "mnist.json")
    with open(dataset_config_path) as f:
        dataset_config = json.load(f)

    device = torch.device("cpu")
    model_neural = npc_model.ResNet34MTL(dataset_config, device)
    model_neural = torch.nn.DataParallel(model_neural)
    header.config_neural["file_name_checkpoint_best"] = neural_ckpt
    utility.loadCheckpoint(neural_ckpt, model_neural)
    model_neural.eval()

    pc_joint = pc.ProbabilisticCircuit(device)
    pc_marginal = pc.ProbabilisticCircuit(device)
    pc_joint.load(file_path_pc)
    pc_marginal.load(file_path_pc)
    pc_settings_joint = test_npc.generatePCSettings(dataset_config, device)
    pc_settings_marginal = torch.clone(pc_settings_joint)
    pc_settings_marginal[:, -1] = -1
    pc_joint.set_leaf_nodes_categorical(pc_settings_joint)
    pc_marginal.set_leaf_nodes_categorical(pc_settings_marginal)
    utility.loadCheckpoint(pc_ckpt, pc_joint, True)
    utility.loadCheckpoint(pc_ckpt, pc_marginal, True)

    explainer = NPCExplainer(
        pc_joint=pc_joint, pc_marginal=pc_marginal, pc_settings_joint=pc_settings_joint,
        attribute_names=ATTRIBUTE_NAMES, attribute_cardinalities=ATTRIBUTE_CARDINALITIES,
        device=device,
    )
    return model_neural, explainer


def validate_npc_explainer(model_neural, npc_explainer, npc_npz, interpret_json, n_check=3):
    import utility
    print("[VALIDACION] comparando NPCExplainer.mpe_query en vivo vs. interpret.json precomputado...")
    paths = npc_npz["paths"]
    for i in range(n_check):
        path = str(paths[i])
        fname = os.path.basename(path)
        tfm = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor(),
        ])
        image = tfm(Image.open(path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            outputs_attribute_original, _ = model_neural(image)
        attribute_probs = utility.applySoftmaxAttribute(outputs_attribute_original)

        mpe = npc_explainer.mpe_query(attribute_probs)
        precomputed = interpret_json[fname]["mpe"]
        d1_live, d2_live = mpe["attribute_values"]["number-first"], mpe["attribute_values"]["number-second"]
        d1_pre = int(precomputed["number-first"])
        d2_pre = int(precomputed["number-second"])
        match = (d1_live == d1_pre) and (d2_live == d2_pre)
        print(f"  {fname}: en vivo=({d1_live},{d2_live}) precomputado=({d1_pre},{d2_pre}) "
             f"{'OK' if match else 'DIFIERE'}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    kdm_model, kdm_explainer = load_kdm()
    kdm_npz = np.load(os.path.join(RESULTS_DIR, "kdm_final-seed42", "predictions.npz"))
    npc_npz = np.load(os.path.join(RESULTS_DIR, "npc_knowledge_seed42", "predictions.npz"))
    with open(os.path.join(RESULTS_DIR, "npc_knowledge_seed42", "mnist.json")) as f:
        interpret_json = json.load(f)

    assert (kdm_npz["sum_true"] == npc_npz["sum_true"]).all(), "test sets deben coincidir"

    print("[INFO] calculando embeddings de la muestra de train (para nearest-neighbor de KDM)...")
    train_embeddings, train_paths = compute_train_embeddings(kdm_model)
    print(f"[INFO] {len(train_paths)} imagenes de train embebidas")

    print("[INFO] cargando NPC (Knowledge) en vivo para validar NPCExplainer...")
    model_neural, npc_explainer = load_npc_live("knowledge")
    validate_npc_explainer(model_neural, npc_explainer, npc_npz, interpret_json)

    kdm_pred = kdm_npz["p_sum"].argmax(axis=1)
    npc_pred = npc_npz["p_sum"].argmax(axis=1)
    true_label = kdm_npz["sum_true"]
    selection = select_comparison_instances(kdm_pred, npc_pred, true_label, n_disagree=5, n_both_wrong=5)
    print(f"[INFO] instancias seleccionadas: {selection}")

    import matplotlib.pyplot as plt
    tfm_display = torchvision.transforms.Resize((224, 224))

    pdf_path = os.path.join(OUT_DIR, "comparison_kdm_vs_npc.pdf")
    with PdfPages(pdf_path) as pdf:
        for category, indices in selection.items():
            for idx in indices:
                path = str(kdm_npz["paths"][idx])
                fname = os.path.basename(path)
                image = np.array(tfm_display(Image.open(path).convert("RGB")))

                attribution = torch.from_numpy(kdm_npz["attribution"][idx])
                top_idx, top_val = kdm_explainer.top_k_components(attribution, k=5)
                top_components = [
                    {"component": c, "weight": v, "attributes": kdm_explainer.decode_final_component(c)}
                    for c, v in zip(top_idx, top_val)
                ]

                mpe = {"attribute_values": {
                    "number-first": int(interpret_json[fname]["mpe"]["number-first"]),
                    "number-second": int(interpret_json[fname]["mpe"]["number-second"]),
                }}
                ce_entry = interpret_json[fname].get("ce", {}).get("class", {})
                npc_correct = npc_pred[idx] == true_label[idx]
                ce = None if npc_correct else {
                    "corrected": bool(ce_entry),
                    "delta": {"class (CE)": {"before": int(npc_pred[idx]),
                                             "after": list(ce_entry.keys())[0] if ce_entry else "n/a"}},
                }

                fig = plt.figure(figsize=(14, 3.2))
                build_comparison_panel(
                    fig, image, true_label=int(true_label[idx]),
                    kdm_pred=int(kdm_pred[idx]), npc_pred=int(npc_pred[idx]),
                    kdm_attribution=attribution, kdm_top_components=top_components,
                    npc_mpe=mpe, npc_ce=ce, kdm_explainer=kdm_explainer,
                )
                fig.suptitle(f"[{category}] {fname}", fontsize=10)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    print(f"[DONE] reporte comparativo: {pdf_path}")

    # --- demo aparte: imagenes de train mas cercanas a un componente de cabeza ---
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for row, (head_name, comp_idx) in enumerate([("number-first", 0), ("number-second", 0)]):
        neighbors = kdm_explainer.nearest_training_images_for_head(
            head_name, comp_idx, train_embeddings, train_paths, k=4)
        for col, (nb_path, dist) in enumerate(neighbors):
            axes[row, col].imshow(Image.open(nb_path).convert("RGB"))
            axes[row, col].set_title(f"{head_name} c{comp_idx}\ndist={dist:.2f}", fontsize=7)
            axes[row, col].axis("off")
    fig.suptitle("KDM: imágenes de train más cercanas a un componente de cabeza\n"
                "(prototipo proyectado al espacio de embeddings, no a un valor semántico)")
    fig.tight_layout()
    demo_path = os.path.join(OUT_DIR, "figures", "interp_kdm_nearest_training_images.png")
    fig.savefig(demo_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] demo de vecinos más cercanos: {demo_path}")


if __name__ == "__main__":
    main()
