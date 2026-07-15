#!/usr/bin/env python3
"""Matriz de confusion, ROC (one-vs-rest, macro-promediada) y precision-recall
(macro-promediada) para las 19 clases-suma de MNIST-Addition, comparando KDM
(exp_03 final-seed42) contra NPC Knowledge y NPC Data (exp_01, seed42).

Requiere que ya existan:
  results/kdm_final-seed42/predictions.npz
  results/npc_knowledge_seed42/predictions.npz
  results/npc_data_seed42/predictions.npz
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
N_CLASSES = 19

# paleta categorica fija (skill dataviz, orden 1-2-3: blue/aqua/yellow)
COLORS = {
    "KDM": "#2a78d6",
    "NPC (Knowledge)": "#1baf7a",
    "NPC (Data)": "#eda100",
}
SEQUENTIAL_BLUE = "Blues"

MODELS = {
    "KDM": os.path.join(RESULTS_DIR, "kdm_final-seed42", "predictions.npz"),
    "NPC (Knowledge)": os.path.join(RESULTS_DIR, "npc_knowledge_seed42", "predictions.npz"),
    "NPC (Data)": os.path.join(RESULTS_DIR, "npc_data_seed42", "predictions.npz"),
}


def load_predictions():
    data = {}
    for name, path in MODELS.items():
        if not os.path.isfile(path):
            print(f"[WARN] falta {path}, se omite {name}")
            continue
        npz = np.load(path)
        data[name] = {"p_sum": npz["p_sum"], "sum_true": npz["sum_true"]}
    return data


def plot_confusion_matrices(data):
    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 4.6))
    if len(data) == 1:
        axes = [axes]
    for ax, (name, d) in zip(axes, data.items()):
        y_pred = d["p_sum"].argmax(axis=1)
        cm = confusion_matrix(d["sum_true"], y_pred, labels=range(N_CLASSES))
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1)
        acc = (y_pred == d["sum_true"]).mean()
        ax.set_title(f"{name}\naccuracy={acc:.4f}", fontsize=10)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(range(N_CLASSES), fontsize=6)
        ax.set_yticklabels(range(N_CLASSES), fontsize=6)
    fig.colorbar(im, ax=axes, shrink=0.8, label="fracción de la clase real")
    fig.suptitle("Matriz de confusión normalizada por fila — 19 clases-suma (test, seed 42)")
    fig.savefig(os.path.join(OUT_DIR, "eval_confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def macro_roc(p_sum, sum_true):
    y_bin = label_binarize(sum_true, classes=range(N_CLASSES))
    fpr_grid = np.linspace(0, 1, 200)
    tpr_interp = []
    for c in range(N_CLASSES):
        if y_bin[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], p_sum[:, c])
        tpr_interp.append(np.interp(fpr_grid, fpr, tpr))
    macro_tpr = np.mean(tpr_interp, axis=0)
    auc_macro = roc_auc_score(y_bin, p_sum, average="macro", multi_class="ovr")
    return fpr_grid, macro_tpr, auc_macro


def macro_pr(p_sum, sum_true):
    y_bin = label_binarize(sum_true, classes=range(N_CLASSES))
    recall_grid = np.linspace(0, 1, 200)
    precision_interp = []
    for c in range(N_CLASSES):
        if y_bin[:, c].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, c], p_sum[:, c])
        order = np.argsort(recall)
        precision_interp.append(np.interp(recall_grid, recall[order], precision[order]))
    macro_precision = np.mean(precision_interp, axis=0)
    ap_macro = average_precision_score(y_bin, p_sum, average="macro")
    return recall_grid, macro_precision, ap_macro


def plot_roc(data):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color="#c3c2b7", linestyle="--", linewidth=1, label="azar")
    for name, d in data.items():
        fpr, tpr, auc = macro_roc(d["p_sum"], d["sum_true"])
        ax.plot(fpr, tpr, color=COLORS[name], linewidth=1.8,
               label=f"{name} (AUC={auc:.4f})")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("ROC macro-promediada (one-vs-rest, 19 clases)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "eval_roc_macro.png"), dpi=150)
    plt.close(fig)


def plot_pr(data):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, d in data.items():
        recall, precision, ap = macro_pr(d["p_sum"], d["sum_true"])
        ax.plot(recall, precision, color=COLORS[name], linewidth=1.8,
               label=f"{name} (AP={ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall macro-promediada (one-vs-rest, 19 clases)")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "eval_precision_recall_macro.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_predictions()
    print(f"[INFO] modelos cargados: {list(data.keys())}")
    if not data:
        print("[WARN] no hay predictions.npz disponibles todavia, nada que graficar")
        return

    plot_confusion_matrices(data)
    plot_roc(data)
    plot_pr(data)

    summary = {}
    for name, d in data.items():
        y_pred = d["p_sum"].argmax(axis=1)
        acc = float((y_pred == d["sum_true"]).mean())
        _, _, auc = macro_roc(d["p_sum"], d["sum_true"])
        _, _, ap = macro_pr(d["p_sum"], d["sum_true"])
        summary[name] = {"accuracy": acc, "roc_auc_macro": float(auc), "ap_macro": float(ap)}
        print(f"[INFO] {name}: acc={acc:.4f} AUC={auc:.4f} AP={ap:.4f}")

    import json
    with open(os.path.join(OUT_DIR, "..", "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] figuras en {OUT_DIR}")


if __name__ == "__main__":
    main()
