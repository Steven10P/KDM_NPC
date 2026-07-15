#!/usr/bin/env python3
"""Materializa localmente SOLO el split de test de MNIST-Addition (3500 imagenes)
a partir del split congelado (mnist_split.json.gz) y las imagenes ya procesadas
en data/npc/datasets/mnist/instances/processed/<suma 0-18>/<archivo>.png.

Replica el algoritmo de external/npc-dataset-utils/src/npc-dataset-utils/split.py
(createSymlinks) pero con rutas absolutas propias, sin tocar header.py ni el
repo vendorizado -- solo se necesita el split "test" para evaluacion, no train
ni validate.
"""
import gzip
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SPLIT_GZ = os.path.join(REPO_ROOT, "external", "npc-dataset-utils", "configs",
                        "npc-dataset-utils", "mnist_split.json.gz")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "npc", "datasets", "mnist",
                             "instances", "processed")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data_local", "mnist_test")
OUT_DIR = os.path.abspath(OUT_DIR)


def build_file_map():
    file_map = {}
    for dir_label in os.listdir(PROCESSED_DIR):
        dir_path = os.path.join(PROCESSED_DIR, dir_label)
        if not os.path.isdir(dir_path):
            continue
        for file_name in os.listdir(dir_path):
            # dataset_file_extension_images en header.py = ".jpg" (default para
            # awa2/celeba/gtsrb) -- para archivos .png esto no hace match, asi
            # que split.py deja el file_key IGUAL al nombre completo con
            # extension. Replicamos ese mismo (no-)comportamiento para que las
            # claves calcen con las de mnist_split.json.gz.
            file_key = file_name.split(".jpg")[0]
            file_map[file_key] = os.path.join(dir_label, file_name)
    return file_map


def main():
    with gzip.open(SPLIT_GZ, "rb") as f:
        config_split = json.loads(f.read().decode("utf-8"))
    file_keys_test = config_split["test"]
    print(f"[INFO] file_keys en split 'test': {len(file_keys_test)}")

    file_map = build_file_map()
    print(f"[INFO] imagenes procesadas totales indexadas: {len(file_map)}")

    if os.path.isdir(OUT_DIR):
        print(f"[INFO] {OUT_DIR} ya existe, se asume materializado. Saltando.")
        n = sum(len(fs) for _, _, fs in os.walk(OUT_DIR))
        print(f"[INFO] archivos existentes: {n}")
        assert n == len(file_keys_test), f"esperaba {len(file_keys_test)}, hay {n}"
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    use_symlink = True
    for i, file_key in enumerate(file_keys_test):
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
                print("[WARN] symlink fallo (probablemente falta permiso en Windows), "
                      "se usa copia de archivos en su lugar.")
                import shutil
                shutil.copy(src, dst)
        else:
            import shutil
            shutil.copy(src, dst)
        if (i + 1) % 500 == 0:
            print(f"[INFO] {i + 1}/{len(file_keys_test)}")

    n = sum(len(fs) for _, _, fs in os.walk(OUT_DIR))
    assert n == len(file_keys_test), f"esperaba {len(file_keys_test)}, hay {n}"
    print(f"[DONE] {n} imagenes de test materializadas en {OUT_DIR}")


if __name__ == "__main__":
    main()
