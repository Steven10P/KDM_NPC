#!/usr/bin/env python3
"""Materializa una MUESTRA aleatoria (no todo el split train, seria 28000
imagenes) del split de train de MNIST-Addition, para el lookup de "imagen de
entrenamiento mas cercana" de KDMExplainer.nearest_training_images_for_head
-- no hace falta mas que unos cuantos miles de imagenes para que el vecino
mas cercano en el espacio de embeddings de 512-dim sea representativo.

Mismo patron de prepare_test_split.py, pero con split="train" + muestreo.
"""
import gzip
import json
import os
import random

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SPLIT_GZ = os.path.join(REPO_ROOT, "external", "npc-dataset-utils", "configs",
                        "npc-dataset-utils", "mnist_split.json.gz")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "npc", "datasets", "mnist",
                             "instances", "processed")
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_local", "mnist_train_sample"))
SAMPLE_SIZE = 2000
SEED = 0


def build_file_map():
    file_map = {}
    for dir_label in os.listdir(PROCESSED_DIR):
        dir_path = os.path.join(PROCESSED_DIR, dir_label)
        if not os.path.isdir(dir_path):
            continue
        for file_name in os.listdir(dir_path):
            file_key = file_name.split(".jpg")[0]  # ver prepare_test_split.py para el porque
            file_map[file_key] = os.path.join(dir_label, file_name)
    return file_map


def main():
    with gzip.open(SPLIT_GZ, "rb") as f:
        config_split = json.loads(f.read().decode("utf-8"))
    file_keys_train = config_split["train"]

    random.Random(SEED).shuffle(file_keys_train)
    sample_keys = file_keys_train[:SAMPLE_SIZE]
    print(f"[INFO] muestra de train: {len(sample_keys)} de {len(file_keys_train)}")

    file_map = build_file_map()

    if os.path.isdir(OUT_DIR):
        n = sum(len(fs) for _, _, fs in os.walk(OUT_DIR))
        print(f"[INFO] {OUT_DIR} ya existe con {n} archivos, se asume materializado. Saltando.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    use_symlink = True
    for i, file_key in enumerate(sample_keys):
        rel = file_map[file_key]
        dir_label = os.path.dirname(rel)
        os.makedirs(os.path.join(OUT_DIR, dir_label), exist_ok=True)
        src = os.path.join(PROCESSED_DIR, rel)
        dst = os.path.join(OUT_DIR, rel)
        if use_symlink:
            try:
                os.symlink(src, dst)
            except OSError:
                use_symlink = False
                import shutil
                shutil.copy(src, dst)
        else:
            import shutil
            shutil.copy(src, dst)
        if (i + 1) % 500 == 0:
            print(f"[INFO] {i + 1}/{len(sample_keys)}")

    n = sum(len(fs) for _, _, fs in os.walk(OUT_DIR))
    print(f"[DONE] {n} imagenes de muestra de train materializadas en {OUT_DIR}")


if __name__ == "__main__":
    main()
