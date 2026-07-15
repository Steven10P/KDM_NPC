#!/usr/bin/env python3
"""Interpretabilidad NATIVA de KDM: la prediccion es una combinacion lineal
exacta de 190 prototipos aprendidos (component attribution, Ec. 12 de
kdm/layers/kdm_layer.py::forward) -- no es una aproximacion post-hoc como
SHAP, es la cuenta que el modelo realmente hace.

Dos graficas:
  1. Para unas pocas muestras de test (mezcla de aciertos y errores), el peso
     normalizado por componente (190-dim, suma 1) con los top-5 componentes
     etiquetados por el par de digitos mas cercano (de components_table.json,
     que es una propiedad GLOBAL del modelo entrenado, no de la muestra).
  2. Diagnostico global: entropia de la distribucion de atribucion por
     muestra, separada en predicciones correctas vs incorrectas -- si KDM
     "sabe cuando no sabe", los errores deberian concentrarse en mayor
     entropia (atribucion mas repartida entre componentes en conflicto).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "kdm_final-seed42")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
COLOR = "#2a78d6"
COLOR_ERR = "#e34948"


def load():
    npz = np.load(os.path.join(RESULTS_DIR, "predictions.npz"))
    with open(os.path.join(RESULTS_DIR, "components_table.json")) as f:
        components = json.load(f)
    return npz, components


def label_component(components, idx):
    c = components[idx]
    i, j = c["nearest_digit_pair"]
    return f"c{idx}: ({i},{j})→{c['nearest_pair_implied_sum']}"


def plot_example_attributions(npz, components):
    p_sum = npz["p_sum"]
    sum_true = npz["sum_true"]
    attribution = npz["attribution"]
    pred = p_sum.argmax(axis=1)
    correct_idx = np.where(pred == sum_true)[0]
    wrong_idx = np.where(pred != sum_true)[0]

    rng = np.random.default_rng(0)
    chosen = list(rng.choice(correct_idx, size=min(3, len(correct_idx)), replace=False))
    if len(wrong_idx) > 0:
        chosen += list(rng.choice(wrong_idx, size=min(2, len(wrong_idx)), replace=False))

    fig, axes = plt.subplots(len(chosen), 1, figsize=(8, 2.4 * len(chosen)))
    if len(chosen) == 1:
        axes = [axes]

    for ax, sample_idx in zip(axes, chosen):
        attr = attribution[sample_idx]
        top5 = np.argsort(attr)[::-1][:5]
        is_correct = pred[sample_idx] == sum_true[sample_idx]
        color = COLOR if is_correct else COLOR_ERR
        ax.bar(range(len(attr)), attr, color="#e1e0d9", width=1.0)
        ax.bar(top5, attr[top5], color=color, width=1.0)
        for rank, c_idx in enumerate(top5):
            ax.annotate(label_component(components, int(c_idx)),
                       (c_idx, attr[c_idx]), fontsize=6.5, rotation=60,
                       ha="left", va="bottom")
        status = "correcto" if is_correct else "INCORRECTO"
        ax.set_title(f"muestra {sample_idx}: suma real={sum_true[sample_idx]}, "
                    f"predicho={pred[sample_idx]} ({status})", fontsize=9)
        ax.set_ylabel("peso")
        ax.set_xlim(-1, len(attr))
    axes[-1].set_xlabel("componente del KDM final (0-189)")
    fig.suptitle("Atribución por componente (Ec. 12) — top-5 en color, resto en gris")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "interp_kdm_component_attribution_examples.png"),
               dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chosen


def plot_entropy_diagnostic(npz):
    p_sum = npz["p_sum"]
    sum_true = npz["sum_true"]
    attribution = npz["attribution"]
    pred = p_sum.argmax(axis=1)
    correct = pred == sum_true

    eps = 1e-12
    entropy = -(attribution * np.log(attribution + eps)).sum(axis=1)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bins = np.linspace(0, entropy.max() + 1e-6, 40)
    ax.hist(entropy[correct], bins=bins, alpha=0.65, color=COLOR,
           label=f"correctas (n={correct.sum()})", density=True)
    ax.hist(entropy[~correct], bins=bins, alpha=0.65, color=COLOR_ERR,
           label=f"incorrectas (n={(~correct).sum()})", density=True)
    ax.axvline(entropy[correct].mean(), color=COLOR, linestyle="--", linewidth=1)
    ax.axvline(entropy[~correct].mean(), color=COLOR_ERR, linestyle="--", linewidth=1)
    ax.set_xlabel("entropía de la atribución por componente (nats)")
    ax.set_ylabel("densidad")
    ax.set_title("KDM: ¿la incertidumbre nativa (atribución repartida) predice el error?\n"
                 f"media correctas={entropy[correct].mean():.3f}  "
                 f"media incorrectas={entropy[~correct].mean():.3f}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "interp_kdm_attribution_entropy.png"), dpi=150)
    plt.close(fig)
    return float(entropy[correct].mean()), float(entropy[~correct].mean())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    npz, components = load()
    chosen = plot_example_attributions(npz, components)
    print(f"[INFO] muestras graficadas: {chosen}")
    mean_correct, mean_wrong = plot_entropy_diagnostic(npz)
    print(f"[INFO] entropía media -- correctas: {mean_correct:.4f}  incorrectas: {mean_wrong:.4f}")

    summary = {
        "examples_plotted": [int(i) for i in chosen],
        "attribution_entropy_mean_correct": mean_correct,
        "attribution_entropy_mean_incorrect": mean_wrong,
    }
    with open(os.path.join(RESULTS_DIR, "interpretability_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[DONE] figuras en {OUT_DIR}")


if __name__ == "__main__":
    main()
