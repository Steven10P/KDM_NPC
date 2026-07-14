"""Cascada KDM para MNIST-Addition: tronco ResNet-34 compartido + 2 cabezas KDM
(una por dígito) + un KDM final que combina ambas en la predicción de la suma.

Dos formas de construir el KDM final, seleccionables vía `final_mode`:
  - "cartesian":      cartesian_product(p1,p2) -> punto único -> KDMLayer(coseno)
  - "distributional": KDM explícita de 100 componentes -> KDMLayer(CrossProductKernelLayer)

Ver experiments/exp_02_mnist_kdm_base/DESIGN.md para la discusión matemática
completa de ambas variantes.
"""

from typing import Literal

import torch
import torch.nn as nn
import torchvision

from kdm.layers import CosineKernelLayer, CrossProductKernelLayer, KDMLayer
from kdm.init import init_kdm_layer
from kdm.models import KDMClassModel
from kdm.utils import cartesian_product, comp2dm, dm2discrete, pure2dm

RESNET_NECK_SIZE = 512
N_DIGIT_VALUES = 10
N_SUM_CLASSES = 19


def build_shared_trunk() -> nn.Module:
    """ResNet-34 con fc->Identity, igual que ResNet34MTL de NPC (para comparación justa)."""
    resnet = torchvision.models.resnet34(weights="IMAGENET1K_V1")
    resnet.fc = nn.Identity()
    return resnet


def _joint_onehot_basis() -> torch.Tensor:
    """Base fija (100,20): fila k=i*10+j = concat(onehot(i), onehot(j)).

    Coincide con el orden de aplanado de cartesian_product (a*b).reshape(bs,-1),
    donde i (primer dígito) varía más lento que j (segundo dígito).
    """
    onehot = torch.eye(N_DIGIT_VALUES)
    first = onehot.repeat_interleave(N_DIGIT_VALUES, dim=0)   # fila k -> onehot(i)
    second = onehot.repeat(N_DIGIT_VALUES, 1)                 # fila k -> onehot(j)
    return torch.cat([first, second], dim=1)                  # (100, 20)


class KDMCascade(nn.Module):
    def __init__(
        self,
        final_mode: Literal["cartesian", "distributional"],
        n_comp_head: int = 100,
        n_comp_final: int = 190,
        sigma_head: float = 1.0,
        sigma_final: float = 1.0,
    ):
        super().__init__()
        assert final_mode in ("cartesian", "distributional")
        self.final_mode = final_mode
        self.n_comp_head = n_comp_head
        self.n_comp_final = n_comp_final

        self.trunk = build_shared_trunk()

        self.head1 = KDMClassModel(
            encoded_size=RESNET_NECK_SIZE, dim_y=N_DIGIT_VALUES,
            encoder=nn.Identity(), n_comp=n_comp_head, sigma=sigma_head,
        )
        self.head2 = KDMClassModel(
            encoded_size=RESNET_NECK_SIZE, dim_y=N_DIGIT_VALUES,
            encoder=nn.Identity(), n_comp=n_comp_head, sigma=sigma_head,
        )

        if final_mode == "cartesian":
            # KDMClassModel trae RBF hardcodeado; el vector de 100-dim es
            # categórico (suma 1), así que se arma la capa a mano con coseno.
            self.kdm_final = KDMLayer(
                kernel=CosineKernelLayer(),
                dim_x=N_DIGIT_VALUES * N_DIGIT_VALUES, dim_y=N_SUM_CLASSES,
                n_comp=n_comp_final,
            )
        else:
            self.kdm_final = KDMLayer(
                kernel=CrossProductKernelLayer(
                    dim1=N_DIGIT_VALUES,
                    kernel1=CosineKernelLayer(), kernel2=CosineKernelLayer(),
                ),
                dim_x=2 * N_DIGIT_VALUES, dim_y=N_SUM_CLASSES,
                n_comp=n_comp_final,
            )
            self.register_buffer("joint_basis", _joint_onehot_basis())  # (100, 20)

    def forward(self, image: torch.Tensor):
        neck = self.trunk(image)          # (bs, 512) -- computado UNA vez, compartido
        p1 = self.head1(neck)             # (bs, 10)
        p2 = self.head2(neck)             # (bs, 10)

        if self.final_mode == "cartesian":
            joint = cartesian_product([p1, p2])          # (bs, 100)
            rho_x = pure2dm(joint)                        # (bs, 1, 101)
        else:
            w = cartesian_product([p1, p2])               # (bs, 100) -- pesos p1[i]*p2[j]
            v = self.joint_basis.unsqueeze(0).expand(neck.shape[0], -1, -1)  # (bs, 100, 20)
            rho_x = comp2dm(w, v)                          # (bs, 100, 21)

        rho_y = self.kdm_final(rho_x)
        p_sum = dm2discrete(rho_y)                         # (bs, 19)
        return p1, p2, p_sum

    @torch.no_grad()
    def init_components(self, images: torch.Tensor, digit1: torch.Tensor,
                        digit2: torch.Tensor, sum_class: torch.Tensor) -> None:
        """Inicializa c_x/c_y/c_w de las 3 capas KDM desde un batch real.

        `images` debe traer, para cada cabeza, al menos `n_comp_head` muestras
        por CADA valor 0-9 (para poder muestrear estratificado), y para el
        final al menos `n_comp_final // 19` muestras por clase-suma.
        `digit1`/`digit2`: enteros (bs,) con el valor real de cada dígito.
        `sum_class`: enteros (bs,) con la clase-suma real (0-18).

        init_kdm_layer exige que encoded_x/samples_y tengan EXACTAMENTE
        `n_comp` filas -- por eso el muestreo estratificado corta al tamaño
        exacto en vez de tomar todo lo disponible.
        """
        device = next(self.parameters()).device
        neck = self.trunk(images.to(device))

        def stratified_idx(labels: torch.Tensor, n_values: int, n_total: int) -> torch.Tensor:
            per_value = n_total // n_values
            assert per_value * n_values == n_total, \
                f"n_comp={n_total} debe ser divisible por n_values={n_values}"
            chosen = []
            for value in range(n_values):
                candidates = (labels == value).nonzero(as_tuple=True)[0]
                assert len(candidates) >= per_value, \
                    f"valor {value}: se necesitan {per_value} muestras, hay {len(candidates)}"
                chosen.append(candidates[torch.randperm(len(candidates))[:per_value]])
            return torch.cat(chosen)

        idx1 = stratified_idx(digit1, N_DIGIT_VALUES, self.n_comp_head)
        y1 = torch.nn.functional.one_hot(digit1[idx1], N_DIGIT_VALUES).float()
        init_kdm_layer(self.head1.kdm, neck[idx1], y1, init_sigma=True)

        idx2 = stratified_idx(digit2, N_DIGIT_VALUES, self.n_comp_head)
        y2 = torch.nn.functional.one_hot(digit2[idx2], N_DIGIT_VALUES).float()
        init_kdm_layer(self.head2.kdm, neck[idx2], y2, init_sigma=True)

        idx_f = stratified_idx(sum_class, N_SUM_CLASSES, self.n_comp_final)
        y_f = torch.nn.functional.one_hot(sum_class[idx_f], N_SUM_CLASSES).float()
        true_d1 = torch.nn.functional.one_hot(digit1[idx_f], N_DIGIT_VALUES).float()
        true_d2 = torch.nn.functional.one_hot(digit2[idx_f], N_DIGIT_VALUES).float()

        if self.final_mode == "cartesian":
            x_f = cartesian_product([true_d1, true_d2])          # (n_comp_final, 100)
        else:
            x_f = torch.cat([true_d1, true_d2], dim=1)            # (n_comp_final, 20)
        init_kdm_layer(self.kdm_final, x_f, y_f, init_sigma=True)
