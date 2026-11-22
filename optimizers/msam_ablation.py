"""
M-SAM Ablation Variants.

Two subclasses of MultimodalChaoticOptimizerMSAM for ablation study:

  MSAMNoOGM    — Shapley + SAM on dominant encoder, but NO OGM-GE modulation.
                 Isolates the SAM perturbation contribution.

  MSAMUniform  — OGM-GE + SAM, but random/uniform modality selection
                 (50/50) instead of Shapley. Shows Shapley matters.

Usage:
    from optimizers.msam_ablation import MSAMNoOGM, MSAMUniform
"""

import torch
from typing import Dict

try:
    from optimizers.chaotic_optimizer_msam import MultimodalChaoticOptimizerMSAM
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from optimizers.chaotic_optimizer_msam import MultimodalChaoticOptimizerMSAM


__all__ = ["MSAMNoOGM", "MSAMUniform", "MSAMShapleyGMOnly"]


class MSAMNoOGM(MultimodalChaoticOptimizerMSAM):
    """
    M-SAM WITHOUT OGM-GE gradient modulation.

    Still uses Shapley values to identify dominant modality and applies
    SAM perturbation to that encoder, but skips the gradient suppression
    and GE noise injection. This isolates SAM's contribution.
    """

    @torch.no_grad()
    def _modulate_gradients(self,
                            contributions: Dict[str, float],
                            dominant: str) -> None:
        """No-op: skip OGM-GE gradient modulation."""
        pass


class MSAMShapleyGMOnly(MultimodalChaoticOptimizerMSAM):
    """
    GM + Shapley WITHOUT SAM perturbation.

    Uses Shapley values to identify dominant modality and applies OGM-GE
    gradient modulation, but skips the SAM perturbation step entirely.
    This isolates the Shapley-guided gradient modulation contribution
    (no sharpness-aware step at all).
    """

    def _sam_perturb(self, dominant: str) -> Dict[str, torch.Tensor]:
        """No-op: skip SAM perturbation."""
        return {}

    def _sam_restore(self, dominant: str, eps_store: Dict[str, torch.Tensor]):
        """No-op: nothing to restore."""
        pass


class MSAMUniform(MultimodalChaoticOptimizerMSAM):
    """
    M-SAM with UNIFORM/RANDOM modality selection instead of Shapley.

    Still applies OGM-GE and SAM perturbation, but selects the "dominant"
    modality randomly (50/50) rather than using Shapley values. This tests whether
    random selection is as effective as Shapley selection.
    """

    @torch.no_grad()
    def _compute_shapley(self,
                         inputs: Dict[str, torch.Tensor],
                         labels: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Randomly select a modality for each step (no Shapley passes).

        One-hot weights ensure the parent's argmax genuinely samples the
        dominant encoder, rather than choosing the first modality on ties.
        """
        selected = int(torch.randint(len(self.modality_names), (1,)).item())
        return {m: float(i == selected)
                for i, m in enumerate(self.modality_names)}


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.joint_model import build_model
    from optimizers.survival_coalition import batch_cox_loss

    B, N_GENES, IMG_DIM = 4, 1000, 256

    model = build_model(n_genes=N_GENES, image_dim=IMG_DIM, verbose=False)

    # Test MSAMNoOGM
    base_opt = torch.optim.SGD(model.parameters(), lr=1e-3,
                                momentum=0.9, weight_decay=1e-4)
    no_ogm = MSAMNoOGM(
        base_opt, model,
        modality_names=["image", "gene"],
        base_lr=1e-3, rho=0.05, alpha=0.5,
    )

    inputs = {"image": torch.randn(B, IMG_DIM), "gene": torch.randn(B, N_GENES)}
    labels = {"survival": torch.randint(0, 2, (B,)), "os_months": torch.rand(B)*24+1, "vital_status": torch.randint(0, 2, (B,)).float()}

    model.train()
    def closure():
        out = model(inputs)
        loss = batch_cox_loss(out["survival"].squeeze(-1), labels["os_months"], labels["vital_status"])
        return loss, out

    loss, _ = no_ogm.step(closure, inputs, labels)
    print(f"MSAMNoOGM — loss={loss.item():.4f}  ✓")

    # Test MSAMUniform
    model2 = build_model(n_genes=N_GENES, image_dim=IMG_DIM, verbose=False)
    base_opt2 = torch.optim.SGD(model2.parameters(), lr=1e-3,
                                 momentum=0.9, weight_decay=1e-4)
    uniform = MSAMUniform(
        base_opt2, model2,
        modality_names=["image", "gene"],
        base_lr=1e-3, rho=0.05, alpha=0.5,
    )

    inputs2 = {"image": torch.randn(B, IMG_DIM), "gene": torch.randn(B, N_GENES)}
    labels2 = {"survival": torch.randint(0, 2, (B,)), "os_months": torch.rand(B)*24+1, "vital_status": torch.randint(0, 2, (B,)).float()}

    model2.train()
    def closure2():
        out = model2(inputs2)
        loss = batch_cox_loss(out["survival"].squeeze(-1), labels2["os_months"], labels2["vital_status"])
        return loss, out

    loss2, _ = uniform.step(closure2, inputs2, labels2)
    hist = uniform.get_contribution_history()
    # Random selection must produce one-hot contribution pairs.
    assert all(v in (0.0, 1.0) for v in hist["image"])
    print(f"MSAMUniform — loss={loss2.item():.4f}  contributions={hist}  ✓")

    # Test MSAMShapleyGMOnly
    model3 = build_model(n_genes=N_GENES, image_dim=IMG_DIM, verbose=False)
    base_opt3 = torch.optim.SGD(model3.parameters(), lr=1e-3,
                                 momentum=0.9, weight_decay=1e-4)
    shapley_gm = MSAMShapleyGMOnly(
        base_opt3, model3,
        modality_names=["image", "gene"],
        base_lr=1e-3, rho=0.05, alpha=0.5,
    )

    inputs3 = {"image": torch.randn(B, IMG_DIM), "gene": torch.randn(B, N_GENES)}
    labels3 = {"survival": torch.randint(0, 2, (B,)), "os_months": torch.rand(B)*24+1, "vital_status": torch.randint(0, 2, (B,)).float()}

    model3.train()
    def closure3():
        out = model3(inputs3)
        loss = batch_cox_loss(out["survival"].squeeze(-1), labels3["os_months"], labels3["vital_status"])
        return loss, out

    loss3, _ = shapley_gm.step(closure3, inputs3, labels3)
    print(f"MSAMShapleyGMOnly — loss={loss3.item():.4f}  ✓")

    print("All ablation self-tests PASSED.")
