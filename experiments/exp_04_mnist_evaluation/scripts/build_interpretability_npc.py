#!/usr/bin/env python3
"""Interpretabilidad NATIVA de NPC, usando los mecanismos que los propios
autores ya implementaron en npc-models (test_npc.py, activados con
header.npc_interpret=True dentro de run_inference_npc.py):

  - MPE (Most Probable Explanation): para cada muestra, la asignacion exacta
    de atributos que maximiza matrix_pc * matrix_neural (inferencia MAP
    tratable sobre el circuito, no una aproximacion). "MPE alignment rate" =
    fraccion de predicciones correctas donde esa explicacion coincide con los
    atributos reales -- mide si el circuito "razona por el camino correcto",
    no solo si acierta la clase final.
  - CE (Contrastive/Counterfactual Explanation): para muestras mal
    clasificadas, ajusta via gradiente las probabilidades de atributo hasta
    que la prediccion cambia a la clase correcta. "CE correction rate" = de
    los errores, que fraccion se puede corregir con un ajuste minimo de
    atributos (indica si el error viene del reconocedor de atributos o de
    algo mas profundo).
  - Estructura del circuito (nodos suma/producto/hoja, profundidad) --
    interpretabilidad estructural: inferencia tratable por construccion.

Este script:
  1. Parsea el log de texto de run_inference_npc.py (las lineas
     logger.log_info que SI se imprimen, aunque test() no las retorne) para
     sacar TV/accuracy/MPE-alignment/CE-correction por variante.
  2. Lee interpret.json (uno por variante) para armar una tabla cualitativa
     de ejemplos (verdad, prediccion, explicacion MPE, si coincide).
  3. Grafica una comparacion Knowledge vs Data de MPE-alignment y
     CE-correction, y guarda la tabla de ejemplos como JSON+markdown.
"""
import json
import os
import re

import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
LOG_PATH = os.path.join(
    "C:\\Users\\bspd2\\AppData\\Local\\Temp\\claude\\C--Users-bspd2-Maestria-Tesis-KDM-NPC",
    "67d2bfdf-513a-47c5-bd7d-72f7e6e13c41", "scratchpad", "npc_inference.log")

VARIANTS = ["knowledge", "data"]
COLORS = {"knowledge": "#1baf7a", "data": "#eda100"}


def parse_log_metrics(log_path):
    with open(log_path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    blocks = re.split(r"\[INFO\] variante: (knowledge|data)", text)
    # blocks = [preamble, "knowledge", block_knowledge, "data", block_data]
    metrics_by_variant = {}
    for i in range(1, len(blocks), 2):
        variant = blocks[i]
        block = blocks[i + 1]

        def find(pattern):
            m = re.search(pattern, block)
            return float(m.group(1)) if m else None

        metrics_by_variant[variant] = {
            "mean_tv_distance": find(r"Testing mean TV distance: ([\d.]+)\."),
            "mean_concept_accuracy": find(r"Testing mean concept accuracy: ([\d.]+)\."),
            "classification_accuracy": find(r"Testing classification accuracy: ([\d.]+)\."),
            "ce_correction_rate": find(r"CE correction rate: ([\d.]+)\."),
            "mpe_alignment_rate": find(r"MPE alignment rate: ([\d.]+)\."),
        }
    return metrics_by_variant


def load_interpret(variant):
    path = os.path.join(RESULTS_DIR, f"npc_{variant}_seed42", "mnist.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_example_table(interpret, n=8):
    rows = []
    for fname, entry in list(interpret.items())[:n]:
        gt = entry["ground_truth"]
        pred = entry["prediction"]
        mpe = entry.get("mpe", {})
        rows.append({
            "file": fname,
            "true_first": gt.get("number-first"),
            "true_second": gt.get("number-second"),
            "true_class": gt.get("class"),
            "pred_class": list(pred.get("class", {}).keys()),
            "mpe_first": mpe.get("number-first"),
            "mpe_second": mpe.get("number-second"),
            "mpe_aligned": mpe.get("aligned"),
        })
    return rows


def plot_comparison(metrics_by_variant):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    labels = ["MPE alignment\nrate", "CE correction\nrate"]
    for ax, key, title in zip(
        axes, ["mpe_alignment_rate", "ce_correction_rate"],
        ["MPE alignment rate\n(¿la explicación exacta coincide con la verdad?)",
         "CE correction rate\n(¿los errores se corrigen con un ajuste mínimo?)"]
    ):
        values = [metrics_by_variant[v].get(key) for v in VARIANTS]
        bars = ax.bar(VARIANTS, values, color=[COLORS[v] for v in VARIANTS])
        for bar, val in zip(bars, values):
            if val is not None:
                ax.annotate(f"{val:.3f}", (bar.get_x() + bar.get_width() / 2, val),
                           ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("NPC — interpretabilidad nativa del circuito (MPE / CE), Knowledge vs. Data")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "interp_npc_mpe_ce_comparison.png"), dpi=150,
               bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.isfile(LOG_PATH):
        print(f"[WARN] no existe {LOG_PATH} todavía")
        return
    metrics_by_variant = parse_log_metrics(LOG_PATH)
    print(f"[INFO] métricas parseadas: {json.dumps(metrics_by_variant, indent=2)}")

    if not metrics_by_variant or any(
        metrics_by_variant.get(v, {}).get("mpe_alignment_rate") is None for v in VARIANTS
    ):
        print("[WARN] el log no tiene todavía las métricas de ambas variantes -- "
              "¿sigue corriendo run_inference_npc.py?")

    plot_comparison(metrics_by_variant)

    examples_by_variant = {}
    for variant in VARIANTS:
        interpret = load_interpret(variant)
        if interpret is None:
            print(f"[WARN] no existe interpret.json para {variant} todavía")
            continue
        examples_by_variant[variant] = build_example_table(interpret)
        print(f"[INFO] {variant}: interpret.json con {len(interpret)} muestras")

    out = {"metrics_by_variant": metrics_by_variant, "examples": examples_by_variant}
    with open(os.path.join(RESULTS_DIR, "npc_interpretability_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"[DONE] figuras en {OUT_DIR}")


if __name__ == "__main__":
    main()
