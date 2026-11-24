"""
Vanilla SAM (Sharpness-Aware Minimization) optimizer.

Perturbs ALL parameters (not modality-aware) to find flat minima.
This is a key baseline: shows that M-SAM's modality-awareness is the
critical ingredient, not just SAM perturbation.

Reference: Foret et al., "Sharpness-Aware Minimization for Efficiently
Improving Generalization" (ICLR 2021).

Usage:
    from optimizers.sam_optimizer import SAMOptimizer

    base_opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    sam_opt = SAMOptimizer(base_opt, model, rho=0.05)

    # Training loop — requires closure
    for inputs, labels in loader:
        def closure():
            out = model(inputs)
            loss = criterion(out, labels)
            return loss, out

        loss, out = sam_opt.step(closure)
"""

import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple

try:
    from optimizers.chaotic_lr_scheduler import CosineAnnealingScheduler
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from optimizers.chaotic_lr_scheduler import CosineAnnealingScheduler


__all__ = ["SAMOptimizer"]


class SAMOptimizer:
    """
    Vanilla SAM optimizer — perturbs ALL parameters.

    Args:
        optimizer : base PyTorch optimizer (SGD recommended)
        model     : nn.Module
        rho       : perturbation radius (default 0.05)
    """

    def __init__(self,
                 optimizer,
                 model: nn.Module,
                 rho: float = 0.05):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.scheduler = None  # Set externally

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def _sam_perturb(self) -> Dict[str, torch.Tensor]:
        """
        Compute and apply SAM perturbation to ALL parameters.
        Returns stored perturbations for restore step.
        """
        grad_norms = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norms.append(param.grad.norm())

        if not grad_norms:
            return {}

        grad_norm = torch.norm(torch.stack(grad_norms))

        eps_store = {}
        if grad_norm > 1e-8:
            scale = self.rho / (grad_norm + 1e-12)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    eps = param.grad * scale
                    eps_store[name] = eps.clone()
                    param.data.add_(eps)

        return eps_store

    def _sam_restore(self, eps_store: Dict[str, torch.Tensor]):
        """Restore parameters after SAM second pass."""
        for name, param in self.model.named_parameters():
            if name in eps_store:
                param.data.sub_(eps_store[name])

    def step(self,
             closure: Callable[[], Tuple[torch.Tensor, dict]],
             inputs: Optional[Dict[str, torch.Tensor]] = None,
             labels: Optional[Dict[str, torch.Tensor]] = None,
             ) -> Tuple[torch.Tensor, dict]:
        """
        Full SAM step:
            1. First forward+backward → get gradients
            2. SAM perturb ALL parameters
            3. Second forward+backward → get gradients at perturbed point
            4. Restore parameters
            5. Base optimizer step

        Args:
            closure : callable () → (loss, outputs_dict)
            inputs  : unused (API compatibility with M-SAM)
            labels  : unused (API compatibility with M-SAM)

        Returns:
            (loss, outputs) from the second forward pass.
        """
        # Pass 1: standard forward + backward
        self.optimizer.zero_grad()
        loss, out = closure()
        loss.backward()

        # SAM: perturb all params
        eps_store = self._sam_perturb()

        # Pass 2: forward + backward at perturbed point
        self.optimizer.zero_grad()
        loss, out = closure()
        loss.backward()

        # Restore params
        self._sam_restore(eps_store)

        # Optimizer step
        self.optimizer.step()

        # Scheduler step (per-batch)
        if self.scheduler is not None:
            self.scheduler.step()

        return loss, out

    def get_current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def get_contribution_history(self) -> Dict[str, List[float]]:
        """No Shapley for vanilla SAM — returns empty."""
        return {}


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
    sam_opt = SAMOptimizer(base_opt, model, rho=0.05)

    times = torch.tensor([4., 8., 11., 15., 18., 21., 29., 36.])
    events = torch.tensor([1., 0., 1., 1., 0., 1., 0., 1.])

    for step in range(5):
        inputs = {
            "image": torch.randn(B, IMG_DIM),
            "gene": torch.randn(B, N_GENES),
        }
        model.train()

        def closure():
            output = model(inputs)
            assert output["survival"].shape == (B, 1)
            risk = output["survival"].squeeze(-1)
            loss = batch_cox_loss(risk, times, events)
            return loss, output

        loss, output = sam_opt.step(closure)
        assert torch.isfinite(loss), "Cox loss became non-finite"
        assert output["survival"].shape == (B, 1)
        print(f"  step {step+1}  Cox loss={loss.item():.4f}"
              f"  lr={sam_opt.get_current_lr():.6f}")

    print("All SAMOptimizer self-tests PASSED.")
