"""Cascada KDM para GTSRB: tronco ResNet-34 compartido + 4 cabezas KDM (una
por atributo: color/shape/symbol/text) + un KDM final (coseno) sobre el
producto cartesiano de las 4 -- generalizacion de `kdm_cascade.py` (2
cabezas homogeneas, dim 10 cada una) a K=4 cabezas de cardinalidad
HETEROGENEA (3/4/26/10). Ver experiments/exp_05_gtsrb_kdm_npc/{DESIGN,
IMPLEMENTATION}.md para el diseño completo.

Unica variante: Cartesian (Distributional ya fue descartada en exp_02 sobre
MNIST-Addition, no se repite esa comparacion aqui).
"""
import math

import torch
import torch.nn as nn
import torchvision

from kdm.layers import CosineKernelLayer, KDMLayer
from kdm.init import init_kdm_layer
from kdm.models import KDMClassModel
from kdm.utils import cartesian_product, dm2discrete, pure2dm

RESNET_NECK_SIZE = 512
N_CLASSES = 43

# Cardinalidades reales de gtsrb.json["attributes"] (ver
# exp_05_gtsrb_kdm_npc/scripts/_build_class_mapping.py, verificado leyendo
# el JSON) -- color=3, shape=4, symbol=26, text=10.
ATTRIBUTE_CARDINALITIES = {"color": 3, "shape": 4, "symbol": 26, "text": 10}


def build_shared_trunk() -> nn.Module:
    """Identico a kdm_cascade.py::build_shared_trunk -- mismo tronco que
    MNIST-Addition, deliberado (ver DESIGN.md §3: aislar la variable
    dataset de la variable arquitectura)."""
    resnet = torchvision.models.resnet34(weights="IMAGENET1K_V1")
    resnet.fc = nn.Identity()
    return resnet


class KDMCascadeGTSRB(nn.Module):
    def __init__(self, n_comp_per_value: int, n_comp_final: int, sigma_head: float = 1.0):
        """`n_comp_per_value`: componentes POR VALOR de atributo (no un
        total compartido) -- cada cabeza usa
        `n_comp = n_comp_per_value * cardinalidad_de_esa_cabeza`, para que
        `stratified_idx` (init_components, mas abajo) pueda seguir exigiendo
        muestreo exactamente balanceado por valor con cabezas de
        cardinalidad distinta (ver IMPLEMENTATION.md §2.2 -- un
        `n_comp_head` compartido, como en MNIST, no generaliza: el mcm de
        {3,4,26,10} es 780).
        `n_comp_final`: debe ser multiplo exacto de N_CLASSES=43 (mismo
        motivo, para la capa final).
        """
        super().__init__()
        assert n_comp_final % N_CLASSES == 0, (
            f"n_comp_final={n_comp_final} debe ser multiplo de N_CLASSES={N_CLASSES}")
        self.n_comp_per_value = n_comp_per_value
        self.n_comp_final = n_comp_final
        self.attribute_names = list(ATTRIBUTE_CARDINALITIES.keys())

        self.trunk = build_shared_trunk()
        self.heads = nn.ModuleDict({
            name: KDMClassModel(
                encoded_size=RESNET_NECK_SIZE, dim_y=card, encoder=nn.Identity(),
                n_comp=n_comp_per_value * card, sigma=sigma_head,
            )
            for name, card in ATTRIBUTE_CARDINALITIES.items()
        })

        dim_x_final = math.prod(ATTRIBUTE_CARDINALITIES.values())  # 3*4*26*10 = 3120
        self.kdm_final = KDMLayer(
            kernel=CosineKernelLayer(), dim_x=dim_x_final, dim_y=N_CLASSES, n_comp=n_comp_final,
        )

    def forward(self, image: torch.Tensor):
        neck = self.trunk(image)  # (bs, 512) -- computado una vez, compartido
        p = {name: head(neck) for name, head in self.heads.items()}  # cada uno (bs, card_i)

        joint = cartesian_product([p[name] for name in self.attribute_names])  # (bs, 3120)
        rho_x = pure2dm(joint)  # (bs, 1, 3121)
        rho_y = self.kdm_final(rho_x)
        p_class = dm2discrete(rho_y)  # (bs, 43)
        return p, p_class

    @torch.no_grad()
    def init_components(self, images: torch.Tensor, attribute_labels: dict, class_labels: torch.Tensor,
                        forward_batch_size: int = 256, sigma_mult: float = 1.0) -> None:
        """Inicializacion data-driven de las 5 capas KDM (4 cabezas + final)
        desde un batch real, generalizando kdm_cascade.py::init_components
        de 2 a K atributos.

        `attribute_labels`: dict nombre-atributo -> tensor (bs,) con el
        indice entero del valor real de ese atributo por imagen.
        `class_labels`: tensor (bs,) con el ClassId real (0-42) por imagen.
        """
        device = next(self.parameters()).device
        neck_chunks = []
        for i in range(0, images.shape[0], forward_batch_size):
            chunk = images[i:i + forward_batch_size].to(device)
            neck_chunks.append(self.trunk(chunk))
        neck = torch.cat(neck_chunks, dim=0)

        def stratified_idx(labels: torch.Tensor, n_values: int, n_total: int) -> torch.Tensor:
            # GTSRB tiene desbalance real -- a diferencia de MNIST-Addition,
            # un valor de atributo/clase puede tener menos candidatos que
            # per_value incluso en una muestra grande de init. Se muestrea
            # sin reemplazo hasta agotar los candidatos y se completa CON
            # reemplazo -- son solo puntos de partida del entrenamiento, no
            # una garantia estadistica.
            per_value = n_total // n_values
            assert per_value * n_values == n_total, \
                f"n_comp={n_total} debe ser divisible por n_values={n_values}"
            chosen = []
            for value in range(n_values):
                candidates = (labels == value).nonzero(as_tuple=True)[0]
                assert len(candidates) >= 1, f"valor {value}: no hay ningun candidato"
                if len(candidates) >= per_value:
                    chosen.append(candidates[torch.randperm(len(candidates))[:per_value]])
                else:
                    extra = candidates[torch.randint(len(candidates), (per_value - len(candidates),))]
                    chosen.append(torch.cat([candidates, extra]))
            return torch.cat(chosen)

        # idx_f se calcula UNA sola vez (no por atributo) -- las 4 tuplas
        # one-hot que arman x_f deben venir de las MISMAS imagenes fila a
        # fila para que cartesian_product las combine correctamente (mismo
        # patron que kdm_cascade.py, donde true_d1/true_d2 comparten idx_f).
        idx_f = stratified_idx(class_labels, N_CLASSES, self.n_comp_final)

        true_onehots = {}
        for name, card in ATTRIBUTE_CARDINALITIES.items():
            n_total = self.n_comp_per_value * card
            idx = stratified_idx(attribute_labels[name], card, n_total)
            y = torch.nn.functional.one_hot(attribute_labels[name][idx], card).float()
            init_kdm_layer(self.heads[name].kdm, neck[idx], y, init_sigma=True, sigma_mult=sigma_mult)

            true_onehots[name] = torch.nn.functional.one_hot(
                attribute_labels[name][idx_f], card).float()

        y_f = torch.nn.functional.one_hot(class_labels[idx_f], N_CLASSES).float()
        x_f = cartesian_product([true_onehots[name] for name in self.attribute_names])
        # kdm_final usa kernel coseno (no RBF) -- init_sigma=True no tiene
        # efecto acá, se deja por consistencia con la firma de init_kdm_layer
        # (mismo comentario que kdm_cascade.py).
        init_kdm_layer(self.kdm_final, x_f, y_f, init_sigma=True, sigma_mult=sigma_mult)
