#!/usr/bin/env python3
"""Empaqueta GTSRB para subir a Kaggle, per IMPLEMENTATION.md 1.2.

Fuente: data/gtsrb_official/extracted/Train/<0-42>/<classId>_<track>_<frame>.png
(el mirror OFICIAL usado por npc-dataset-utils -- meowmeowmeowmeowmeow/
gtsrb-german-traffic-sign en Kaggle -- verificado que su convencion de
nombres calza EXACTAMENTE con las claves de gtsrb_split.json.gz; la copia
de GTSRB descargada previamente en data/gtsrb/ era una distribucion
distinta, incompatible, y no se usa aqui).

Construye:
  - gtsrb_processed.zip: <class-semantico>/<archivo>.png (usa
    class_mapping.json, ya verificado contra el gtsrb.py oficial de
    npc-dataset-utils).
  - MANIFEST.json: mismo formato que data/kaggle_dataset_stage/MANIFEST.json
    (hash global determinista = sha256 de los pares (ruta, sha256-del-
    archivo) ordenados).
  - dataset-metadata.json: para `kaggle datasets create`.
  - gtsrb.json + gtsrb_split.json.gz (se sube comprimido -- Kaggle
    auto-descomprime .gz al crear el dataset, el kernel lo re-comprime en
    Kaggle antes de pasarselo a split.py, mismo patron ya usado para MNIST).
"""
import hashlib
import json
import os
import shutil
import zipfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SOURCE_DIR = os.path.join(REPO_ROOT, "data", "gtsrb_official", "extracted", "Train")
CLASS_MAPPING_PATH = os.path.join(os.path.dirname(__file__), "..", "class_mapping.json")
GTSRB_JSON = os.path.join(REPO_ROOT, "external", "npc-dataset-utils", "configs",
                          "npc-dataset-utils", "gtsrb.json")
GTSRB_SPLIT_GZ = os.path.join(REPO_ROOT, "external", "npc-dataset-utils", "configs",
                              "npc-dataset-utils", "gtsrb_split.json.gz")
STAGE_DIR = os.path.join(REPO_ROOT, "data", "kaggle_dataset_stage_gtsrb")


def build_zip_and_manifest():
    with open(CLASS_MAPPING_PATH) as f:
        class_mapping = json.load(f)  # "00000".."00042" -> nombre-semantico

    zip_path = os.path.join(STAGE_DIR, "gtsrb_processed.zip")
    file_hashes = []
    class_counts = {}

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for class_id_padded, semantic_name in class_mapping.items():
            class_id = str(int(class_id_padded))  # carpetas fuente sin padding: "0".."42"
            src_dir = os.path.join(SOURCE_DIR, class_id)
            assert os.path.isdir(src_dir), f"falta carpeta fuente: {src_dir}"

            files = sorted(os.listdir(src_dir))
            class_counts[semantic_name] = len(files)
            for file_name in files:
                src_path = os.path.join(src_dir, file_name)
                # SIN prefijo "gtsrb_processed/" -- Kaggle ya envuelve el
                # contenido extraido en una carpeta con el nombre del zip
                # (verificado en Kaggle: incluir el prefijo aqui produce
                # doble anidamiento gtsrb_processed/gtsrb_processed/<clase>/).
                arcname = f"{semantic_name}/{file_name}"
                zf.write(src_path, arcname)

                with open(src_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                file_hashes.append((f"{semantic_name}/{file_name}", file_hash))

            print(f"[INFO] {semantic_name} (ClassId {class_id}): {len(files)} imagenes")

    file_hashes.sort()
    global_hash = hashlib.sha256(
        "\n".join(f"{p}\t{h}" for p, h in file_hashes).encode()
    ).hexdigest()

    manifest = {
        "dataset": "gtsrb (NPC oficial, mirror meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)",
        "n_images": len(file_hashes),
        "class_counts": class_counts,
        "global_sha256": global_hash,
        "source_config": "gtsrb.json",
    }
    with open(os.path.join(STAGE_DIR, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[OK] {len(file_hashes)} imagenes empaquetadas en {zip_path}")
    print(f"[OK] Hash global del dataset: {global_hash}")
    return global_hash


def stage_config_files():
    shutil.copy(GTSRB_JSON, os.path.join(STAGE_DIR, "gtsrb.json"))
    shutil.copy(GTSRB_SPLIT_GZ, os.path.join(STAGE_DIR, "gtsrb_split.json.gz"))

    metadata = {
        "title": "GTSRB (NPC oficial, congelado)",
        "id": "bspenad10/gtsrb-npc",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(os.path.join(STAGE_DIR, "dataset-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    os.makedirs(STAGE_DIR, exist_ok=True)
    global_hash = build_zip_and_manifest()
    stage_config_files()
    print(f"\n[DONE] staging listo en {STAGE_DIR}")
    print(f"[INFO] EXPECTED_GLOBAL_SHA256 para _template_kernel.py: {global_hash}")


if __name__ == "__main__":
    main()
