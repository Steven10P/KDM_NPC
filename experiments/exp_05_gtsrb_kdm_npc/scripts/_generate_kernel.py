#!/usr/bin/env python3
"""Genera carpetas de kernel de Kaggle (scripts/<condicion>/) a partir de
_template_kernel.py, sustituyendo los placeholders __CONDITION__/__SEED__/
__EPOCHS__/__N_COMP_PER_VALUE__/__N_COMP_FINAL__/__LR_KDM__/__SIGMA_MULT__.

Mismo patron que exp_03_mnist_kdm_sweep/scripts/_generate_kernel.py.

Uso:
    python _generate_kernel.py --only search-baseline   # 1 condicion (validar plantilla primero)
    python _generate_kernel.py                          # las 9 condiciones de la Fase A
    python _generate_kernel.py --confirm N M R S         # 10a: config combinada (n_comp_per_value n_comp_final lr_kdm sigma_mult)
"""

import argparse
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "_template_kernel.py")

# Fase A: barrido uno-a-la-vez, baseline = misma densidad que el ganador de
# exp_03 (n_comp_per_value=10 -> n_comp_final=430=43*10, lr_kdm=3e-3,
# sigma_mult=1.0). Ver DESIGN.md 5.
FASE_A_CONDITIONS = [
    {"name": "search-baseline", "n_comp_per_value": 10, "n_comp_final": 430, "lr_kdm": 3e-3, "sigma_mult": 1.0},
    {"name": "search-npv15", "n_comp_per_value": 15, "n_comp_final": 430, "lr_kdm": 3e-3, "sigma_mult": 1.0},
    {"name": "search-npv20", "n_comp_per_value": 20, "n_comp_final": 430, "lr_kdm": 3e-3, "sigma_mult": 1.0},
    {"name": "search-ncf172", "n_comp_per_value": 10, "n_comp_final": 172, "lr_kdm": 3e-3, "sigma_mult": 1.0},
    {"name": "search-ncf645", "n_comp_per_value": 10, "n_comp_final": 645, "lr_kdm": 3e-3, "sigma_mult": 1.0},
    {"name": "search-lr1e3", "n_comp_per_value": 10, "n_comp_final": 430, "lr_kdm": 1e-3, "sigma_mult": 1.0},
    {"name": "search-lr3e4", "n_comp_per_value": 10, "n_comp_final": 430, "lr_kdm": 3e-4, "sigma_mult": 1.0},
    {"name": "search-sig05", "n_comp_per_value": 10, "n_comp_final": 430, "lr_kdm": 3e-3, "sigma_mult": 0.5},
    {"name": "search-sig20", "n_comp_per_value": 10, "n_comp_final": 430, "lr_kdm": 3e-3, "sigma_mult": 2.0},
]
FASE_A_EPOCHS = 15
FASE_A_SEED = 42

# Fase A2: segunda vuelta del barrido uno-a-la-vez, ahora con lr_kdm=3e-4
# FIJO (el ganador de Fase A) -- las corridas originales de estos mismos 4
# valores se hicieron con lr_kdm=3e-3 (roto), asi que no aislaban el efecto
# real del eje. sigma_mult=2.0 se omite: ya hay evidencia fuerte (exp_03 en
# MNIST + search-sig20b acá) de que ensanchar el kernel perjudica
# independientemente de la tasa de aprendizaje, no hace falta reconfirmar.
FASE_A2_CONDITIONS = [
    {"name": "search-lr3e4-ncf172", "n_comp_per_value": 10, "n_comp_final": 172, "lr_kdm": 3e-4, "sigma_mult": 1.0},
    {"name": "search-lr3e4-ncf645", "n_comp_per_value": 10, "n_comp_final": 645, "lr_kdm": 3e-4, "sigma_mult": 1.0},
    {"name": "search-lr3e4-sig05", "n_comp_per_value": 10, "n_comp_final": 430, "lr_kdm": 3e-4, "sigma_mult": 0.5},
    {"name": "search-lr3e4-npv15", "n_comp_per_value": 15, "n_comp_final": 430, "lr_kdm": 3e-4, "sigma_mult": 1.0},
]

# Fase B: confirmacion a escala completa con el ganador final de Fase A2
# (search-lr3e4-sig05: n_comp_per_value=10, n_comp_final=430, lr_kdm=3e-4,
# sigma_mult=0.5 -- misma accuracy que la referencia de Fase A, TV 19.4x
# mejor; la corrida de confirmacion con ncf645 confirmo que combinar ejes no
# ayuda). 60 epocas x 5 semillas, mismo protocolo que exp_03. Ver DESIGN.md.
FASE_B_WINNER = {"n_comp_per_value": 10, "n_comp_final": 430, "lr_kdm": 3e-4, "sigma_mult": 0.5}
FASE_B_SEEDS = [42, 52, 62, 72, 82]
FASE_B_EPOCHS = 60


def generate(name, n_comp_per_value, n_comp_final, lr_kdm, sigma_mult, epochs, seed):
    condition = f"{name}_seed{seed}"
    dest_dir = os.path.join(SCRIPT_DIR, name)
    os.makedirs(dest_dir, exist_ok=True)

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        content = f.read()
    content = (content
        .replace("__CONDITION__", condition)
        .replace("__SEED__", str(seed))
        .replace("__EPOCHS__", str(epochs))
        .replace("__N_COMP_PER_VALUE__", str(n_comp_per_value))
        .replace("__N_COMP_FINAL__", str(n_comp_final))
        .replace("__LR_KDM__", repr(float(lr_kdm)))
        .replace("__SIGMA_MULT__", repr(float(sigma_mult))))
    leftover = [ph for ph in ("__CONDITION__", "__SEED__", "__EPOCHS__", "__N_COMP_PER_VALUE__",
                              "__N_COMP_FINAL__", "__LR_KDM__", "__SIGMA_MULT__") if ph in content]
    assert not leftover, f"Placeholders sin sustituir: {leftover}"

    script_name = f"kernel_{name.replace('-', '_')}.py"
    script_path = os.path.join(dest_dir, script_name)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)

    slug = f"exp05-kdm-gtsrb-{name}-seed{seed}"
    metadata = {
        "id": f"bspenad10/{slug}",
        "title": f"exp05 KDM GTSRB {name} seed{seed}",
        "code_file": script_name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": ["bspenad10/gtsrb-npc"],  # TODO: confirmar slug real tras el upload (IMPLEMENTATION.md 1.2)
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(dest_dir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] {name} -> {dest_dir}  (slug={slug})")
    return slug


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="generar solo esta condicion de Fase A (por nombre)")
    parser.add_argument("--phase2", action="store_true",
                        help="generar las 4 condiciones de Fase A2 (lr_kdm=3e-4 fijo)")
    parser.add_argument("--confirm", nargs=4,
                        metavar=("N_COMP_PER_VALUE", "N_COMP_FINAL", "LR_KDM", "SIGMA_MULT"),
                        help="generar la 10a condicion (confirmacion combinada) con estos valores")
    parser.add_argument("--final", action="store_true",
                        help="generar las 5 corridas de Fase B (60 epocas, ganador search-lr3e4-sig05)")
    parser.add_argument("--epochs", type=int, default=FASE_A_EPOCHS)
    parser.add_argument("--seed", type=int, default=FASE_A_SEED)
    args = parser.parse_args()

    if args.confirm:
        npv, ncf, lr, sig = args.confirm
        generate("search-confirm", int(npv), int(ncf), float(lr), float(sig), args.epochs, args.seed)
    elif args.phase2:
        for c in FASE_A2_CONDITIONS:
            generate(c["name"], c["n_comp_per_value"], c["n_comp_final"], c["lr_kdm"], c["sigma_mult"],
                     args.epochs, args.seed)
    elif args.final:
        for seed in FASE_B_SEEDS:
            generate(f"final-seed{seed}", FASE_B_WINNER["n_comp_per_value"], FASE_B_WINNER["n_comp_final"],
                     FASE_B_WINNER["lr_kdm"], FASE_B_WINNER["sigma_mult"], FASE_B_EPOCHS, seed)
    else:
        conditions = FASE_A_CONDITIONS
        if args.only:
            conditions = [c for c in conditions if c["name"] == args.only]
            assert conditions, f"condicion no encontrada: {args.only}"
        for c in conditions:
            generate(c["name"], c["n_comp_per_value"], c["n_comp_final"], c["lr_kdm"], c["sigma_mult"],
                     args.epochs, args.seed)
