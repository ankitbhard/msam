"""
SGM-SAM: Shapley-guided modality-targeted SAM and gradient modulation
for Cox cancer survival prediction.

For each minibatch, estimate how much image and gene coalitions improve
the censored Cox partial likelihood over a zero-risk reference:

    v(S) = CoxLoss(empty) - CoxLoss(S), with v(empty) = 0
    phi_image = 0.5*v(image) + 0.5*(v(full) - v(gene))
    phi_gene  = 0.5*v(gene)  + 0.5*(v(full) - v(image))

Negative Shapley contributions are clipped to zero and the contributions
are normalized for gating. The dominant modality's encoder alone receives
the SAM perturbation, using the gradient of the TOTAL Cox loss restricted
to that encoder's parameters. A second backward pass differentiates the
full loss at the perturbed point; then the perturbation is restored.
Finally, the selected encoder gradient is suppressed when its normalized
contribution exceeds 0.5, optionally with Gaussian noise enhancement.

Cox-Shapley requires labels containing 'os_months' and 'vital_status'.
The scalar survival output has shape [batch_size, 1]. The optimizer
can wrap a standard PyTorch optimizer and supports an LR scheduler.
"""
import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple

try:
    from optimizers.chaotic_lr_scheduler import ChaoticLRScheduler
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from optimizers.chaotic_lr_scheduler import ChaoticLRScheduler


__all__ = ["MultimodalChaoticOptimizerMSAM"]


class MultimodalChaoticOptimizerMSAM:
    """
    Chaotic LR + Modality-Aware SAM + OGM-GE optimizer.

    Args:
        optimizer      : base PyTorch optimizer (SGD recommended)
        model          : JointModel instance
        modality_names : list of modality name strings, e.g. ["image", "gene"]
        base_lr        : base LR for ChaoticLRScheduler
        r              : logistic map parameter (default 3.99, fully chaotic)
        rho            : SAM perturbation radius (default 0.05)
        alpha          : OGM-GE suppression strength (default 0.5)
        use_ge         : inject Gaussian noise on dominant modality (default True)
        compute_every  : apply OGM-GE + SAM every N steps (default 1)
    """

    def __init__(self,
                 optimizer,
                 model:           nn.Module,
                 modality_names:  List[str],
                 base_lr:         float = 1e-3,
                 r:               float = 3.99,
                 rho:             float = 0.05,
                 alpha:           float = 0.5,
                 use_ge:          bool  = True,
                 compute_every:   int   = 1):
        self.optimizer      = optimizer
        self.model          = model
        self.modality_names = modality_names
        self.rho            = rho
        self.alpha          = alpha
        self.use_ge         = use_ge
        self.compute_every  = compute_every
        self._step_count    = 0

        self.scheduler = ChaoticLRScheduler(optimizer, base_lr=base_lr, r=r)

        # Contribution history for analysis
        self.contribution_history: Dict[str, List[float]] = {
            m: [] for m in modality_names
        }

    # ── Shapley contribution estimator ────────────────────────────────────────

    @torch.no_grad()
    def _compute_shapley(self, inputs, labels):
        """Survival-specific Cox-loss improvement Shapley (two modalities)."""
        from optimizers.survival_coalition import cox_shapley_contributions
        return cox_shapley_contributions(
            self.model, inputs, labels, self.modality_names
        )

    # ── Dominant modality encoder parameters ─────────────────────────────────

    def _encoder_params(self, modality: str):
        """Yield (name, param) for the given modality's encoder."""
        key = f"{modality}_encoder"
        for name, param in self.model.named_parameters():
            if key in name and param.requires_grad:
                yield name, param

    # ── SAM perturbation ──────────────────────────────────────────────────────

    def _sam_perturb(self, dominant: str) -> Dict[str, torch.Tensor]:
        """
        Compute and apply SAM perturbation to dominant modality encoder.
        Returns saved perturbations {param_name: epsilon} for restore step.
        """
        grad_norm = torch.norm(
            torch.stack([
                param.grad.norm()
                for _, param in self._encoder_params(dominant)
                if param.grad is not None
            ])
        )

        eps_store = {}
        if grad_norm > 1e-8:
            scale = self.rho / (grad_norm + 1e-12)
            for name, param in self._encoder_params(dominant):
                if param.grad is not None:
                    eps = param.grad * scale
                    eps_store[name] = eps.clone()
                    param.data.add_(eps)

        return eps_store

    def _sam_restore(self, dominant: str, eps_store: Dict[str, torch.Tensor]):
        """Restore dominant modality encoder parameters after SAM second pass."""
        for name, param in self._encoder_params(dominant):
            if name in eps_store:
                param.data.sub_(eps_store[name])

    # ── OGM-GE gradient modulation ────────────────────────────────────────────

    @torch.no_grad()
    def _modulate_gradients(self,
                            contributions: Dict[str, float],
                            dominant: str) -> None:
        """
        Suppress dominant modality gradients and optionally add GE noise.
        Non-dominant modalities are left unchanged (k = 1.0).
        """
        for modality, ratio in contributions.items():
            if modality == dominant and ratio > 0.5:
                k = 1.0 - float(torch.tanh(
                    torch.tensor(self.alpha * ratio)
                ))
            else:
                k = 1.0

            key = f"{modality}_encoder"
            for name, param in self.model.named_parameters():
                if key not in name or param.grad is None:
                    continue
                param.grad.mul_(k)
                if self.use_ge and modality == dominant and k < 1.0:
                    noise_std = param.grad.std() * 0.1
                    param.grad.add_(torch.randn_like(param.grad) * noise_std)

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self,
             closure:   Callable[[], Tuple[torch.Tensor, dict]],
             inputs:    Optional[Dict[str, torch.Tensor]] = None,
             labels:    Optional[Dict[str, torch.Tensor]] = None
             ) -> Tuple[torch.Tensor, dict]:
        """
        Full M-SAM step:
            1. First forward+backward  → get gradients
            2. Compute Shapley contributions
            3. SAM perturb dominant encoder
            4. Second forward+backward → get gradients at perturbed point
            5. Restore dominant encoder
            6. OGM-GE modulation on restored gradients
            7. Base optimizer step + chaotic LR step

        Args:
            closure : callable () → (loss, outputs_dict)
                      Should NOT call zero_grad internally.
            inputs  : modality input dict (for Shapley; can be None to skip)
            labels  : dict with "os_months" and "vital_status" (for Cox-Shapley)

        Returns:
            (loss, outputs) from the second forward pass.
        """
        self._step_count += 1
        apply_msam = (
            inputs is not None and
            labels is not None and
            self._step_count % self.compute_every == 0
        )

        # ── Pass 1: standard forward + backward ──────────────────────────────
        self.optimizer.zero_grad()
        loss, out = closure()
        loss.backward()

        if apply_msam:
            # ── Shapley contributions ─────────────────────────────────────────
            contributions = self._compute_shapley(inputs, labels)
            dominant      = max(contributions, key=contributions.get)

            for m in self.modality_names:
                self.contribution_history[m].append(contributions.get(m, 0.0))

            # ── SAM: perturb dominant encoder ────────────────────────────────
            eps_store = self._sam_perturb(dominant)

            # ── Pass 2: forward + backward at perturbed point ────────────────
            self.optimizer.zero_grad()
            loss, out = closure()
            loss.backward()

            # ── Restore dominant encoder ─────────────────────────────────────
            self._sam_restore(dominant, eps_store)

            # ── OGM-GE modulation ────────────────────────────────────────────
            self._modulate_gradients(contributions, dominant)

        # ── Base optimizer + chaotic LR step ─────────────────────────────────
        self.optimizer.step()
        self.scheduler.step()

        return loss, out

    def get_current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def get_contribution_history(self) -> Dict[str, List[float]]:
        """Returns per-modality Shapley contribution history."""
        return self.contribution_history


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.joint_model import build_model
    from optimizers.survival_coalition import batch_cox_loss

    torch.manual_seed(42)
    B, N_GENES, IMG_DIM = 8, 1000, 256

    model = build_model(n_genes=N_GENES, image_dim=IMG_DIM, verbose=False)
    base_opt = torch.optim.SGD(
        model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4
    )
    msam_opt = MultimodalChaoticOptimizerMSAM(
        base_opt, model, modality_names=["image", "gene"],
        base_lr=1e-3, rho=0.05, alpha=0.5, compute_every=1
    )

    lrs = []
    for step in range(5):
        inputs = {
            "image": torch.randn(B, IMG_DIM),
            "gene": torch.randn(B, N_GENES),
        }
        labels = {
            "os_months": torch.tensor(
                [4., 8., 11., 15., 18., 21., 29., 36.]
            ),
            "vital_status": torch.tensor(
                [1., 0., 1., 1., 0., 1., 0., 1.]
            ),
        }

        model.train()

        def closure():
            output = model(inputs)
            assert output["survival"].shape == (B, 1)
            risk = output["survival"].squeeze(-1)
            loss = batch_cox_loss(
                risk, labels["os_months"], labels["vital_status"]
            )
            return loss, output

        loss, output = msam_opt.step(closure, inputs, labels)
        assert torch.isfinite(loss), "Cox loss became non-finite"
        assert output["survival"].shape == (B, 1)
        lrs.append(msam_opt.get_current_lr())
        print(f"  step {step+1}  Cox loss={loss.item():.4f}"
              f"  lr={lrs[-1]:.6f}")

    hist = msam_opt.get_contribution_history()
    assert len(hist["image"]) == len(hist["gene"]) == 5
    for img, gene in zip(hist["image"], hist["gene"]):
        assert 0.0 <= img <= 1.0 and 0.0 <= gene <= 1.0
        assert abs(img + gene - 1.0) < 1e-6, "Invalid Shapley weights"
    print(f"\nCox-Shapley image: {[f'{v:.3f}' for v in hist['image']]}")
    print(f"Cox-Shapley gene : {[f'{v:.3f}' for v in hist['gene']]}")
    print("All MultimodalChaoticOptimizerMSAM self-tests PASSED.")
