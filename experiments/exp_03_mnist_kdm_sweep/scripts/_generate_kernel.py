#!/usr/bin/env python3
"""Genera carpetas de kernel de Kaggle (scripts/<condición>/) a partir de
_template_kernel.py, sustituyendo los placeholders __CONDITION__/__SEED__/
__EPOCHS__/__N_COMP_HEAD__/__N_COMP_FINAL__/__LR_KDM__/__SIGMA_MULT__.

Uso:
    python _generate_kernel.py --only search-baseline   # 1 condición (validar plantilla primero)
    python _generate_kernel.py                          # las 9 condiciones de la Fase A
    python _generate_kernel.py --confirm N M R S         # 10a: config combinada (n_comp_head n_comp_final lr_kdm sigma_mult)
"""

import argparse
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "_template_kernel.py")

# Fase A: barrido uno-a-la-vez alrededor del baseline de exp_02.
FASE_A_CONDITIONS = [
    {"name": "search-baseline", "n_comp_head": 100, "n_comp_final": 190, "lr_kdm": 1e-3, "sigma_mult": 1.0},
    {"name": "search-nch150", "n_comp_head": 150, "n_comp_final": 190, "lr_kdm": 1e-3, "sigma_mult": 1.0},
    {"name": "search-nch200", "n_comp_head": 200, "n_comp_final": 190, "lr_kdm": 1e-3, "sigma_mult": 1.0},
    {"name": "search-ncf285", "n_comp_head": 100, "n_comp_final": 285, "lr_kdm": 1e-3, "sigma_mult": 1.0},
    {"name": "search-ncf380", "n_comp_head": 100, "n_comp_final": 380, "lr_kdm": 1e-3, "sigma_mult": 1.0},
    {"name": "search-lr3e3", "n_comp_head": 100, "n_comp_final": 190, "lr_kdm": 3e-3, "sigma_mult": 1.0},
    {"name": "search-lr3e4", "n_comp_head": 100, "n_comp_final": 190, "lr_kdm": 3e-4, "sigma_mult": 1.0},
    {"name": "search-sig05", "n_comp_head": 100, "n_comp_final": 190, "lr_kdm": 1e-3, "sigma_mult": 0.5},
    {"name": "search-sig20", "n_comp_head": 100, "n_comp_final": 190, "lr_kdm": 1e-3, "sigma_mult": 2.0},
]
FASE_A_EPOCHS = 15
FASE_A_SEED = 42


def generate(name, n_comp_head, n_comp_final, lr_kdm, sigma_mult, epochs, seed):
    condition = f"{name}_seed{seed}"
    dest_dir = os.path.join(SCRIPT_DIR, name)
    os.makedirs(dest_dir, exist_ok=True)

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        content = f.read()
    content = (content
        .replace("__CONDITION__", condition)
        .replace("__SEED__", str(seed))
        .replace("__EPOCHS__", str(epochs))
        .replace("__N_COMP_HEAD__", str(n_comp_head))
        .replace("__N_COMP_FINAL__", str(n_comp_final))
        .replace("__LR_KDM__", repr(float(lr_kdm)))
        .replace("__SIGMA_MULT__", repr(float(sigma_mult))))
    leftover = [ph for ph in ("__CONDITION__", "__SEED__", "__EPOCHS__", "__N_COMP_HEAD__",
                              "__N_COMP_FINAL__", "__LR_KDM__", "__SIGMA_MULT__") if ph in content]
    assert not leftover, f"Placeholders sin sustituir: {leftover}"

    script_name = f"kernel_{name.replace('-', '_')}.py"
    script_path = os.path.join(dest_dir, script_name)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)

    slug = f"exp03-kdm-mnist-{name}-seed{seed}"
    metadata = {
        "id": f"bspenad10/{slug}",
        "title": f"exp03 KDM MNIST {name} seed{seed}",
        "code_file": script_name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": ["bspenad10/mnist-addition-npc"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(dest_dir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] {name} -> {dest_dir}  (slug={slug})")
    return slug


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="generar solo esta condición de Fase A (por nombre)")
    parser.add_argument("--confirm", nargs=4, metavar=("N_COMP_HEAD", "N_COMP_FINAL", "LR_KDM", "SIGMA_MULT"),
                        help="generar la 10a condición (confirmación combinada) con estos valores")
    parser.add_argument("--epochs", type=int, default=FASE_A_EPOCHS)
    parser.add_argument("--seed", type=int, default=FASE_A_SEED)
    args = parser.parse_args()

    if args.confirm:
        nch, ncf, lr, sig = args.confirm
        generate("search-confirm", int(nch), int(ncf), float(lr), float(sig), args.epochs, args.seed)
    else:
        conditions = FASE_A_CONDITIONS
        if args.only:
            conditions = [c for c in conditions if c["name"] == args.only]
            assert conditions, f"condición no encontrada: {args.only}"
        for c in conditions:
            generate(c["name"], c["n_comp_head"], c["n_comp_final"], c["lr_kdm"], c["sigma_mult"],
                     args.epochs, args.seed)
