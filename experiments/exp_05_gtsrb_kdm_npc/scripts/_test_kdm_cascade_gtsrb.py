#!/usr/bin/env python3
"""Prueba de forma/logica local (CPU, datos sinteticos) de KDMCascadeGTSRB,
antes de subir nada a Kaggle -- per IMPLEMENTATION.md 6, paso 3.

Genera 430 imagenes sinteticas (10 por cada una de las 43 clases, para que
la estratificacion de la capa final -- que exige exactamente n_comp_final=430
componentes, 10/clase -- tenga candidatos suficientes) con atributos
derivados DETERMINISTICAMENTE de la clase real via gtsrb.json (igual que en
GTSRB real: clase -> tupla de atributos fija), para poder verificar
init_components + forward sin necesitar imagenes reales todavia.
"""
import json
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.models.kdm_cascade_gtsrb import KDMCascadeGTSRB, ATTRIBUTE_CARDINALITIES, N_CLASSES  # noqa: E402

GTSRB_JSON = os.path.join(REPO_ROOT, "external", "npc-dataset-utils", "configs",
                          "npc-dataset-utils", "gtsrb.json")
CLASS_MAPPING = os.path.join(os.path.dirname(__file__), "..", "class_mapping.json")
N_PER_CLASS = 10


def build_synthetic_batch():
    with open(GTSRB_JSON) as f:
        cfg = json.load(f)
    with open(CLASS_MAPPING) as f:
        class_mapping = json.load(f)

    attribute_value_to_idx = {}
    for attribute in cfg["attributes"]:
        name = attribute["name"]
        labels = [l for l in attribute["labels"] if l != ""]
        attribute_value_to_idx[name] = {label: i for i, label in enumerate(labels)}

    class_labels = []
    attribute_labels = {name: [] for name in ATTRIBUTE_CARDINALITIES}
    for class_id in range(N_CLASSES):
        semantic_name = class_mapping[f"{class_id:05d}"]
        sample_labels = cfg["mappings"][semantic_name]["labels"]
        for _ in range(N_PER_CLASS):
            class_labels.append(class_id)
            for name in ATTRIBUTE_CARDINALITIES:
                attribute_labels[name].append(attribute_value_to_idx[name][sample_labels[name]])

    n = len(class_labels)
    images = torch.randn(n, 3, 224, 224)  # contenido irrelevante -- solo se prueba forma/logica
    class_labels = torch.tensor(class_labels)
    attribute_labels = {name: torch.tensor(v) for name, v in attribute_labels.items()}

    for name, card in ATTRIBUTE_CARDINALITIES.items():
        counts = torch.bincount(attribute_labels[name], minlength=card)
        assert (counts >= 10).all(), (
            f"atributo {name}: algun valor tiene <10 muestras ({counts.tolist()}) -- "
            f"subir N_PER_CLASS para la prueba"
        )

    return images, attribute_labels, class_labels


def main():
    images, attribute_labels, class_labels = build_synthetic_batch()
    print(f"[INFO] batch sintetico: {images.shape[0]} imagenes, "
          f"{N_CLASSES} clases x {N_PER_CLASS}/clase")

    model = KDMCascadeGTSRB(n_comp_per_value=10, n_comp_final=430)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] modelo construido: {n_params:,} parametros")

    print("[INFO] corriendo init_components (data-driven, estratificado)...")
    model.init_components(images, attribute_labels, class_labels, forward_batch_size=64)
    print("[INFO] init_components OK")

    model.eval()
    with torch.no_grad():
        batch = images[:8]
        p, p_class = model(batch)

    for name, card in ATTRIBUTE_CARDINALITIES.items():
        assert p[name].shape == (8, card), f"{name}: shape {p[name].shape} != (8, {card})"
        assert torch.allclose(p[name].sum(dim=1), torch.ones(8), atol=1e-4), \
            f"{name}: las probabilidades no suman 1"
    assert p_class.shape == (8, N_CLASSES), f"p_class: shape {p_class.shape} != (8, {N_CLASSES})"
    assert torch.allclose(p_class.sum(dim=1), torch.ones(8), atol=1e-4), \
        "p_class: las probabilidades no suman 1"

    print(f"[INFO] forward OK -- p_class shape={p_class.shape}, "
          f"suma por fila (debe ser ~1.0): {p_class.sum(dim=1).tolist()}")
    print("[DONE] KDMCascadeGTSRB pasa la prueba de forma/logica local")


if __name__ == "__main__":
    main()
