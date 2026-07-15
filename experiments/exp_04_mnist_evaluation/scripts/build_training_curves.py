#!/usr/bin/env python3
"""Curvas de entrenamiento por epoca, para diagnostico de estabilidad/overfitting.

Datos realmente disponibles (no se inventa nada que no se haya registrado):
  - KDM (exp_03, Fase B): loss_history de TRAIN por epoca, 60 epocas, 5
    semillas, ya en cada metrics.json. Nunca se registro perdida/accuracy de
    VALIDACION durante el entrenamiento (solo el eval final de test) -> no hay
    curva train-vs-val para KDM, se documenta como limitacion.
  - NPC stage 1 (ResNet34MTL, 150 epocas, compartido por Knowledge y Data):
    el log crudo de Kaggle (_kaggle_output_seed42_v9_COMPLETE) SI quedo
    guardado y contiene "[INFO]: Validation mean concept accuracy: X." por
    cada epoca -- se parsea con una regex simple sobre el texto plano (el
    archivo es JSON-lines con una coma inicial por linea, no hace falta
    parsearlo como JSON completo).
  - NPC stage 2/3 (circuito CCCP + optimizacion conjunta PGD): no se guardo
    ningun log crudo -- no hay curva por epoca posible sin volver a correr el
    entrenamiento. Se documenta como limitacion, no se plotea nada para esto.
"""
import json
import os
import re

import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXP03_RESULTS = os.path.join(REPO_ROOT, "experiments", "exp_03_mnist_kdm_sweep", "results")
STAGE1_LOG = os.path.join(REPO_ROOT, "experiments", "exp_01_mnist_npc_repro", "results",
                          "_kaggle_output_seed42_v9_COMPLETE", "exp01-npc-mnist-stage1-seed42.log")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")

SEEDS = [42, 52, 62, 72, 82]


def load_kdm_loss_histories():
    histories = {}
    for seed in SEEDS:
        path = os.path.join(EXP03_RESULTS, f"final-seed{seed}", "metrics.json")
        with open(path) as f:
            m = json.load(f)
        histories[seed] = m["loss_history"]
    return histories


def parse_npc_stage1_validation_accuracy():
    pattern = re.compile(r"Validation mean concept accuracy: ([\d.]+)\.")
    values = []
    with open(STAGE1_LOG, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                values.append(float(m.group(1)))
    return values


def plot_kdm_loss(histories):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seed, loss_history in histories.items():
        ax.plot(range(1, len(loss_history) + 1), loss_history, label=f"seed {seed}", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Pérdida de entrenamiento (log)")
    ax.set_title("KDM cascada (exp_03, Fase B) — pérdida de train vs. época\n"
                 "(no se registró validación durante entrenamiento)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "training_kdm_loss_vs_epoch.png"), dpi=150)
    plt.close(fig)


def plot_npc_stage1_validation(values):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = range(1, len(values) + 1)
    ax.plot(epochs, values, color="#d77757", linewidth=1.2)
    best_epoch = int(max(range(len(values)), key=lambda i: values[i])) + 1
    ax.axhline(max(values), color="gray", linestyle="--", linewidth=0.8,
              label=f"mejor: {max(values):.4f} (época {best_epoch})")
    ax.set_xlabel("Época")
    ax.set_ylabel("Accuracy de concepto en validación")
    ax.set_title("NPC stage 1 (ResNet34MTL, seed 42) — accuracy de validación vs. época\n"
                 "(150 épocas, compartido por las variantes Knowledge y Data;\n"
                 "no hay curva de entrenamiento equivalente registrada)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "training_npc_stage1_val_accuracy_vs_epoch.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    histories = load_kdm_loss_histories()
    plot_kdm_loss(histories)
    print(f"[INFO] KDM: {len(histories)} semillas, {len(histories[42])} épocas cada una")

    values = parse_npc_stage1_validation_accuracy()
    print(f"[INFO] NPC stage1: {len(values)} épocas de accuracy de validación parseadas del log")
    if values:
        plot_npc_stage1_validation(values)
        print(f"[INFO] mejor accuracy de validación: {max(values):.4f} "
              f"en época {values.index(max(values)) + 1}")
        print(f"[INFO] última (época {len(values)}): {values[-1]:.4f}")

    print(f"[DONE] figuras en {OUT_DIR}")


if __name__ == "__main__":
    main()
