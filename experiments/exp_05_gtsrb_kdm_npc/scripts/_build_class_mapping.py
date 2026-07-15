#!/usr/bin/env python3
"""Construye y verifica el mapeo carpeta-numerica (ClassId oficial de GTSRB,
"00000".."00042") <-> nombre-semantico (clave de gtsrb.json["mappings"]),
per IMPLEMENTATION.md 1.1.

El orden de las 43 claves de gtsrb.json["mappings"] coincide con el ClassId
oficial (verificado leyendo el JSON: indice 14 = "regulatory--stop", que es
justamente el ClassId 14 = señal de "Stop" en la numeracion canonica de
GTSRB) -- no hace falta ninguna tabla externa, solo leer el JSON en orden.

Este script:
  1. Construye el mapeo ClassId -> nombre-semantico.
  2. Lo verifica cruzando, para varias carpetas, el ClassId real registrado
     en GT-000NN.csv (columna ClassId) contra el indice de carpeta -- deben
     coincidir trivialmente (son la misma cosa por construccion de GTSRB),
     y ademas imprime los atributos (color/shape/symbol/text) que gtsrb.json
     le asigna a ese nombre-semantico, para inspeccion humana de sensatez
     (ej. ClassId 14 = "regulatory--stop" deberia verse con color rojo).
  3. Guarda el mapeo en class_mapping.json para que el script de empaquetado
     de Kaggle (paso siguiente, IMPLEMENTATION.md 1.2) lo use directamente.
"""
import csv
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GTSRB_JSON = os.path.join(REPO_ROOT, "external", "npc-dataset-utils", "configs",
                          "npc-dataset-utils", "gtsrb.json")
TRAINING_DIR = os.path.join(REPO_ROOT, "data", "gtsrb", "gtsrb", "GTSRB", "Training")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "class_mapping.json")

VERIFY_FOLDERS = ["00000", "00014", "00017", "00025", "00042"]  # primero, stop, no-entry, ~mitad, ultimo


def main():
    with open(GTSRB_JSON) as f:
        cfg = json.load(f)
    class_names = list(cfg["mappings"].keys())
    assert len(class_names) == 43, f"esperaba 43 clases, hay {len(class_names)}"

    class_mapping = {f"{i:05d}": name for i, name in enumerate(class_names)}

    print("[INFO] verificando consistencia ClassId <-> carpeta <-> nombre-semantico:\n")
    for folder in VERIFY_FOLDERS:
        class_id = int(folder)
        semantic_name = class_mapping[folder]

        csv_path = os.path.join(TRAINING_DIR, folder, f"GT-{folder}.csv")
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            row = next(reader)
            csv_class_id = int(row["ClassId"])
        assert csv_class_id == class_id, (
            f"carpeta {folder}: GT-{folder}.csv dice ClassId={csv_class_id}, "
            f"se esperaba {class_id} (inconsistencia real en los datos, investigar)"
        )

        labels = cfg["mappings"][semantic_name]["labels"]
        print(f"  ClassId {class_id:2d}  carpeta={folder}  -> {semantic_name}")
        print(f"    atributos: {labels}")

    with open(OUT_PATH, "w") as f:
        json.dump(class_mapping, f, indent=2)
    print(f"\n[DONE] class_mapping.json ({len(class_mapping)} clases) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
