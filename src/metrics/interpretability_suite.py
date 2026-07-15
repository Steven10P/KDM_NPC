"""Native interpretability tooling for KDM and NPC — dataset-agnostic.

Neither model needs a post-hoc, approximate explainer (SHAP/LIME): both are
interpretable by construction, each in a different way.

- KDM: a prediction is an exact finite mixture over learned prototypes
  (Ec. 12, `kdm/layers/kdm_layer.py::forward`). `KDMExplainer` extracts the
  per-component weights for a given input, decodes the final layer's
  components into the attribute-tuple they are closest to, and — for the
  per-attribute head layers, whose components live in the shared ResNet
  embedding space rather than a semantic one-hot space — finds the real
  training images closest to each component ("this is the training example
  this prototype resembles most").
- NPC: the probabilistic circuit supports exact tractable inference.
  `NPCExplainer` wraps two mechanisms already implemented by the NPC authors
  (`npc-models/test_npc.py::findMPE`/`findCE`) as small, standalone,
  per-instance callables instead of being embedded in the batch test loop:
  MPE (Most Probable Explanation — the exact MAP attribute assignment) and a
  gradient-based minimal counterfactual.

Nothing here is dataset-specific: heads/attributes/classes are passed in as
plain names, cardinalities, and tensors — this runs unchanged for
MNIST-Addition (2 attributes) or GTSRB (4 attributes), or any other NPC
benchmark dataset that follows the same attribute -> class shape.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- KDM
@dataclass
class KDMExplainer:
    """Wraps a trained attribute-cascade KDM model (trunk + K heads + final layer).

    `model` must expose:
      - `model.trunk(images) -> (bs, encoded_size)`
      - one `KDMClassModel` per attribute (each with a `.kdm.c_x` of shape
        `(n_comp_head, encoded_size)`), reachable via `model.heads[name]` or
        equivalent attribute access — pass accessors explicitly so this stays
        agnostic to how the caller's cascade class names its attributes.
      - `model.kdm_final`, a `KDMLayer` whose `.c_x` lives in the cartesian
        product of the attributes' one-hot spaces (dim = prod(cardinalities)).

    `attribute_names`/`attribute_cardinalities` define the joint one-hot basis
    used to decode a final-layer component into the attribute-tuple it is
    closest to (same idea as `kdm_cascade.py::_joint_onehot_basis`, generalized
    from 2 to K attributes).
    """

    model: torch.nn.Module
    head_accessor: Callable[[torch.nn.Module, str], torch.nn.Module]
    attribute_names: list[str]
    attribute_cardinalities: list[int]
    eps: float = 1e-12

    def head(self, name: str) -> torch.nn.Module:
        return self.head_accessor(self.model, name)

    # ---- component attribution (Ec. 12) -----------------------------------
    @torch.no_grad()
    def component_attribution(self, rho_x: torch.Tensor) -> torch.Tensor:
        """Per-sample, per-final-component normalized weight — the exact
        mixture weights `forward()` uses to combine `c_y` (kdm_layer.py:78-85),
        without the final `einsum` that collapses them into a prediction.
        Valid when `rho_x` encodes a pure state (`n_comp_in == 1`), which is
        how every KDM cascade seen so far feeds its final layer.
        """
        in_w, out_w = self.model.kdm_final._compute_mixture(rho_x)
        out_w = out_w.clamp(min=self.eps)
        out_w = out_w / out_w.sum(dim=2, keepdim=True)
        return out_w[:, 0, :]  # (bs, n_comp_final) — requires n_comp_in == 1

    def top_k_components(self, attribution: torch.Tensor, k: int = 5):
        """attribution: (n_comp_final,) for a single sample."""
        values, indices = torch.topk(attribution, k=min(k, attribution.numel()))
        return indices.tolist(), values.tolist()

    # ---- decode a final-layer component into an attribute-tuple -----------
    def decode_final_component(self, component_idx: int) -> dict:
        """Projects `kdm_final.c_x[component_idx]` onto the fixed joint
        one-hot basis (argmax per attribute-block) to name the attribute-tuple
        the component is closest to. Generalizes the 2-attribute
        `nearest_pair_idx = c_x.argmax(dim=1)` trick to K attributes: the
        cartesian product's flattening order (`cartesian_product`,
        `kdm/utils.py:159-179`) means attribute i's block spans
        `prod(cardinalities[i+1:])` contiguous slots, so a single flat argmax
        does not directly separate attributes when K > 2 — instead we
        reshape the component vector into the K-dimensional grid and take the
        per-axis argmax of the marginal (sum over the other axes), which is
        exact when the component is itself (close to) a pure attribute-tuple
        one-hot vector, and a reasonable nearest-tuple summary otherwise.
        """
        c_x = self.model.kdm_final.c_x.detach()[component_idx]
        grid = c_x.reshape(self.attribute_cardinalities)
        tuple_idx = []
        for axis in range(len(self.attribute_cardinalities)):
            other_axes = tuple(a for a in range(grid.ndim) if a != axis)
            marginal = grid.sum(dim=other_axes) if other_axes else grid
            tuple_idx.append(int(marginal.argmax().item()))
        return dict(zip(self.attribute_names, tuple_idx))

    # ---- nearest real training images for a HEAD-level component ----------
    @torch.no_grad()
    def nearest_training_images_for_head(
        self, head_name: str, component_idx: int,
        train_embeddings: torch.Tensor, train_paths: list[str], k: int = 3,
    ) -> list[tuple[str, float]]:
        """Head-level components (`head.kdm.c_x`) live in the shared ResNet
        embedding space, not a semantic one-hot space — "what does this
        prototype look like" is answered by finding real training images
        whose embedding is closest to it, not by decoding a basis.
        `train_embeddings`: (n_train_sample, encoded_size), precomputed once
        by running `model.trunk` over a sample of training images (see
        `prepare_train_sample.py` for how that sample is materialized).
        """
        c_x = self.head(head_name).kdm.c_x.detach()[component_idx]
        dist = torch.linalg.norm(train_embeddings - c_x.unsqueeze(0), dim=1)
        values, indices = torch.topk(dist, k=min(k, dist.numel()), largest=False)
        return [(train_paths[i], float(d)) for i, d in zip(indices.tolist(), values.tolist())]

    # ---- spectral contribution plot ----------------------------------------
    def spectral_contribution_plot(self, attribution: torch.Tensor, ax=None,
                                   color: str = "#2a78d6"):
        """Sorted (descending) component weights — the closest tractable
        analogue to a spectral decomposition for this finite, non-orthogonal
        mixture representation: it shows how concentrated ("sparse
        prototype") or diffuse ("many components in conflict") a prediction
        is, independent of which specific components dominate.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(4.5, 3))
        sorted_w = torch.sort(attribution, descending=True).values.numpy()
        ax.plot(range(1, len(sorted_w) + 1), sorted_w, color=color, linewidth=1.5)
        ax.set_yscale("log")
        ax.set_xlabel("componente (ordenado por peso)")
        ax.set_ylabel("peso (log)")
        ax.grid(alpha=0.3, which="both")
        return ax


# --------------------------------------------------------------------------- NPC
@dataclass
class NPCExplainer:
    """Wraps `npc-models`' own exact-inference mechanisms
    (`test_npc.py::computeNPCOutput`/`findMPE`/`findCE`) as standalone,
    per-instance callables — same math, same code path as the paper's
    authors, just factored out of the batched test loop so a caller can query
    a single instance (or a handful, for a comparison report) without setting
    up the full evaluation harness.

    `pc_joint`/`pc_marginal` must already have `set_leaf_nodes_categorical`
    called (i.e. ready to `.forward()`); `pc_settings_joint` is the same
    leaf-setting tensor used to build them (`test_npc.py::generatePCSettings`).
    """

    pc_joint: object
    pc_marginal: object
    pc_settings_joint: torch.Tensor
    attribute_names: list[str]
    attribute_cardinalities: list[int]
    class_names: Optional[list[str]] = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    ce_learning_rate: float = 5e-2
    ce_steps: int = 100

    def _pc_output_shape(self):
        n_classes = self.pc_settings_joint[:, -1].unique().numel()
        n_cols = self.pc_settings_joint.shape[0] // n_classes
        return n_classes, n_cols

    def _compute_npc_output(self, attribute_probs: list[torch.Tensor]):
        """Same computation as `test_npc.py::computeNPCOutput`, imported
        lazily to avoid a hard dependency on npc-models for callers that only
        need the KDM side of this module.
        """
        import test_npc  # npc-models/src/npc-models, added to sys.path by the caller
        n_rows, n_cols = self._pc_output_shape()
        return test_npc.computeNPCOutput(
            attribute_probs, self.pc_joint, self.pc_marginal, n_rows, n_cols, self.device)

    def predict(self, attribute_probs: list[torch.Tensor]) -> torch.Tensor:
        _, _, output_npc = self._compute_npc_output(attribute_probs)
        return output_npc

    def mpe_query(self, attribute_probs: list[torch.Tensor],
                 predicted_class: Optional[int] = None) -> dict:
        """Most Probable Explanation: the exact attribute-value assignment
        that maximizes `matrix_pc * matrix_neural` for the (given or argmax)
        predicted class — exact MAP inference over the circuit, not an
        approximation. Mirrors `test_npc.py::findMPE`, restricted to a single
        instance.
        """
        matrix_pc, matrix_neural, output_npc = self._compute_npc_output(attribute_probs)
        if predicted_class is None:
            predicted_class = int(output_npc[0].argmax().item())

        attribute_indices_list = self.pc_settings_joint[:matrix_pc.shape[1], :-1].cpu().int().tolist()
        matrix_pc_t = matrix_pc.t()[:, [predicted_class]]  # (n_cols, 1)
        matrix_npc = matrix_pc_t * matrix_neural[:, [0]]
        mpe_col = int(matrix_npc[:, 0].argmax().item())
        mpe_attribute_values = attribute_indices_list[mpe_col]

        return {
            "predicted_class": predicted_class,
            "attribute_values": dict(zip(self.attribute_names, mpe_attribute_values)),
        }

    @staticmethod
    def _project_to_simplex(v: torch.Tensor) -> torch.Tensor:
        """Exact Euclidean projection onto the probability simplex — the same
        operation `test_npc.py::findCE` performs via its `rho`/`lambda` dance
        (sort descending, find the threshold `rho` where `u - (cumsum(u)-1)/i
        > 0` stops holding, shift by `theta = cssv[rho-1]/rho`, clamp at 0).
        Written with the standard closed-form (Duchi et al. 2008) instead of
        the original's index bookkeeping — same result, easier to verify.
        """
        n = v.shape[-1]
        u = torch.sort(v, descending=True, dim=-1).values
        cssv = torch.cumsum(u, dim=-1) - 1
        ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
        cond = (u - cssv / ind) > 0
        rho = cond.sum(dim=-1, keepdim=True).clamp(min=1)
        theta = torch.gather(cssv, -1, rho - 1) / rho
        return torch.clamp(v - theta, min=0)

    def counterfactual(self, attribute_probs: list[torch.Tensor], true_class: int) -> dict:
        """Minimal counterfactual: adjust `attribute_probs` by gradient ascent
        on the log-probability of `true_class` under the circuit, projecting
        back onto the probability simplex after each step (exact same
        objective and update rule as `test_npc.py::findCE`), until the
        prediction flips to `true_class` or `ce_steps` is reached. Returns
        the corrected probabilities and a compact delta (attribute values
        whose top-1 changed) so the "what would need to change" story is
        readable without inspecting raw probability vectors.
        """
        original_top1 = [int(p[0].argmax().item()) for p in attribute_probs]
        probs = [p.detach().clone().requires_grad_(True) for p in attribute_probs]
        corrected = False

        for _ in range(self.ce_steps):
            _, _, output_npc = self._compute_npc_output(probs)
            pred = int(output_npc[0].argmax().item())
            if pred == true_class:
                corrected = True
                break
            log_p_true = torch.log(output_npc[0, true_class] + 1e-12)
            log_p_true.backward()
            with torch.no_grad():
                for i in range(len(probs)):
                    updated = probs[i] + self.ce_learning_rate * probs[i].grad
                    probs[i] = self._project_to_simplex(updated)
            probs = [p.detach().clone().requires_grad_(True) for p in probs]

        new_top1 = [int(p[0].argmax().item()) for p in probs]
        delta = {
            name: {"before": original_top1[i], "after": new_top1[i]}
            for i, name in enumerate(self.attribute_names)
            if original_top1[i] != new_top1[i]
        }
        return {"corrected": corrected, "delta": delta, "corrected_probs": [p.detach() for p in probs]}


# --------------------------------------------------------------------------- Comparison
def select_comparison_instances(
    kdm_pred: np.ndarray, npc_pred: np.ndarray, true_label: np.ndarray,
    n_disagree: int = 5, n_both_wrong: int = 5, seed: int = 0,
) -> dict[str, list[int]]:
    """Picks the instances most worth explaining side by side: where KDM and
    NPC disagree with each other, and where both are wrong (the two
    scenarios where a native-interpretability contrast is actually
    informative — where both agree and are right, there is nothing to
    contrast).
    """
    rng = np.random.default_rng(seed)
    disagree = np.where(kdm_pred != npc_pred)[0]
    both_wrong = np.where((kdm_pred != true_label) & (npc_pred != true_label))[0]

    def sample(pool, n):
        return rng.choice(pool, size=min(n, len(pool)), replace=False).tolist()

    return {
        "disagree": sample(disagree, n_disagree),
        "both_wrong": sample(both_wrong, n_both_wrong),
    }


def build_comparison_panel(
    fig, image: np.ndarray, true_label, kdm_pred, npc_pred,
    kdm_attribution: torch.Tensor, kdm_top_components: list[dict],
    npc_mpe: dict, npc_ce: Optional[dict], kdm_explainer: KDMExplainer,
):
    """One instance -> one figure with 4 panels: input image, KDM spectrum +
    top components, NPC MPE explanation, NPC counterfactual (if wrong).
    Dataset-agnostic: every label comes from the caller's dicts, nothing
    hardcoded to digits or traffic signs.
    """
    axes = fig.subplots(1, 4)

    axes[0].imshow(image)
    axes[0].set_title(f"real={true_label}\nKDM={kdm_pred}  NPC={npc_pred}", fontsize=9)
    axes[0].axis("off")

    kdm_explainer.spectral_contribution_plot(kdm_attribution, ax=axes[1])
    axes[1].set_title("KDM: espectro de atribución", fontsize=9)
    top_str = "\n".join(f"c{c['component']}: {c['attributes']}" for c in kdm_top_components[:3])
    axes[1].annotate(top_str, xy=(0.05, 0.05), xycoords="axes fraction", fontsize=6)

    axes[2].axis("off")
    mpe_str = "\n".join(f"{k}={v}" for k, v in npc_mpe["attribute_values"].items())
    axes[2].set_title("NPC: MPE (inferencia MAP exacta)", fontsize=9)
    axes[2].text(0.02, 0.5, mpe_str, fontsize=8, va="center")

    axes[3].axis("off")
    if npc_ce is None:
        axes[3].set_title("NPC: contrafactual (n/a, ya correcto)", fontsize=9)
    else:
        axes[3].set_title(f"NPC: contrafactual (corregido={npc_ce['corrected']})", fontsize=9)
        delta_str = "\n".join(f"{k}: {v['before']}→{v['after']}" for k, v in npc_ce["delta"].items())
        axes[3].text(0.02, 0.5, delta_str or "(sin cambios registrados)", fontsize=8, va="center")

    return fig
