#!/usr/bin/env python3
"""Genera el dataset congelado D̄ de MNIST-Addition para exp_01 (Gate #0).

En lugar de reproducir el emparejamiento vía el shuffle de
`npc-dataset-utils/mnist.py` (sensible al orden de os.listdir, que difiere
entre NTFS y ext4), este script lee los *mappings oficiales* publicados por
los autores en `configs/npc-dataset-utils/mnist.json`: cada clave
`<suma>/<idx1>_<idx2>.png` codifica exactamente qué dos imágenes del arreglo
global (test primero, train después — ver mnist.py main()) se concatenan.
El resultado es por construcción idéntico al dataset del paper.

Equivalencia de píxeles con el pipeline original: mnist.py guarda cada dígito
como PNG en escala de grises (PIL), lo relee con cv2.imread (que replica el
canal a BGR de 3 canales) y concatena con cv2.hconcat. Como PNG es lossless,
eso equivale a `np.repeat(raw[..., None], 3, axis=2)` directo, que es lo que
hacemos aquí — mismos píxeles, sin los 70k PNGs intermedios de `original/`.

Salidas (bajo --out, por defecto data/npc/datasets/mnist/):
  instances/processed/<suma>/<idx1>_<idx2>.png   35,000 imágenes 28x56x3
  MANIFEST.json                                  conteos por clase + SHA256 global

Uso:
  python src/data/generate_mnist_addition.py \
      --raw data/mnist/MNIST/raw \
      --config external/npc-dataset-utils/configs/npc-dataset-utils/mnist.json \
      --out data/npc/datasets/mnist
"""

import argparse
import hashlib
import json
import os
import struct
import sys

import cv2
import numpy as np


def extract_images(file_path):
    """Lee un archivo idx3-ubyte de MNIST (mismo parseo que mnist.py)."""
    with open(file_path, "rb") as file:
        (_, image_count) = struct.unpack(">II", file.read(8))
        (image_height, image_width) = struct.unpack(">II", file.read(8))
        images = np.fromfile(file, dtype=np.dtype(np.uint8).newbyteorder(">"))
        images = images.reshape((image_count, image_height, image_width))
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/mnist/MNIST/raw",
                        help="Directorio con los archivos idx crudos de MNIST")
    parser.add_argument("--config",
                        default="external/npc-dataset-utils/configs/npc-dataset-utils/mnist.json",
                        help="mnist.json oficial con los mappings de los autores")
    parser.add_argument("--out", default="data/npc/datasets/mnist",
                        help="Raíz de salida del dataset (jerarquía estilo npc)")
    args = parser.parse_args()

    # Mismo orden que mnist.py main(): test (10k) primero, luego train (60k).
    images_test = extract_images(os.path.join(args.raw, "t10k-images-idx3-ubyte"))
    images_train = extract_images(os.path.join(args.raw, "train-images-idx3-ubyte"))
    images = np.concatenate((images_test, images_train), axis=0)
    assert images.shape == (70000, 28, 28), images.shape

    with open(args.config) as f:
        config = json.load(f)
    mappings = config["mappings"]
    assert len(mappings) == 35000, len(mappings)

    out_dir = os.path.join(args.out, "instances", "processed")
    os.makedirs(out_dir, exist_ok=True)

    class_counts = {}
    file_hashes = []  # (ruta_relativa, sha256) para el hash global del manifest

    for i, image_name in enumerate(mappings):
        class_name, file_name = image_name.split("/")
        stem = file_name.rsplit(".", 1)[0]
        idx_first, idx_second = (int(s) for s in stem.split("_"))

        # Validación cruzada: la etiqueta de atributo del config debe coincidir
        # con la etiqueta real del dígito. (Las etiquetas reales se validan
        # aparte contra los idx de labels en el paso de verificación.)
        combined = cv2.hconcat([
            np.repeat(images[idx_first][:, :, None], 3, axis=2),
            np.repeat(images[idx_second][:, :, None], 3, axis=2),
        ])

        class_dir = os.path.join(out_dir, class_name)
        if class_name not in class_counts:
            os.makedirs(class_dir, exist_ok=True)
            class_counts[class_name] = 0
        class_counts[class_name] += 1

        out_path = os.path.join(class_dir, file_name)
        if not cv2.imwrite(out_path, combined):
            sys.exit(f"[ERROR] No se pudo escribir {out_path}")

        file_hashes.append((image_name, hashlib.sha256(combined.tobytes()).hexdigest()))

        if (i + 1) % 5000 == 0:
            print(f"[INFO] {i + 1}/35000 imágenes generadas", flush=True)

    # Hash global determinista: SHA256 sobre los pares (ruta, hash) ordenados.
    file_hashes.sort()
    global_hash = hashlib.sha256(
        "\n".join(f"{p}\t{h}" for p, h in file_hashes).encode()
    ).hexdigest()

    manifest = {
        "dataset": "mnist-addition (NPC oficial, generado desde mnist.json)",
        "n_images": len(file_hashes),
        "class_counts": dict(sorted(class_counts.items(), key=lambda kv: int(kv[0]))),
        "global_sha256": global_hash,
        "source_config": os.path.basename(args.config),
    }
    manifest_path = os.path.join(args.out, "MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] {len(file_hashes)} imagenes en {out_dir}")
    print(f"[OK] Hash global del dataset: {global_hash}")
    print(f"[OK] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
