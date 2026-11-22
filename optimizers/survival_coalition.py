"""Censoring-aware two-modality Shapley utilities for scalar Cox risk."""
from typing import Dict
import torch


def batch_cox_loss(risk, times, events):
    """Match train_survival.cox_loss normalization: mean over batch size."""
    risk = risk.reshape(-1)
    times = times.reshape(-1)
    events = events.reshape(-1).float()
    order = torch.argsort(times, descending=True)
    risk = risk[order]
    return -((risk - torch.logcumsumexp(risk, dim=0)) *
             events[order]).mean()


@torch.no_grad()
def cox_shapley_contributions(model, inputs: Dict[str, torch.Tensor],
                              labels: Dict[str, torch.Tensor],
                              modality_names):
    """Shapley of batch Cox-loss improvement over a zero-risk coalition.

    Zero masks absent inputs. A full-forward coalition value uses the same
    times, events, and risk sets as unimodal coalitions.
    """
    if len(modality_names) != 2:
        raise ValueError("Cox-Shapley currently supports exactly two modalities.")
    times, events = labels["os_months"], labels["vital_status"]
    if not bool((events.reshape(-1) > 0).any()):
        return {name: 0.5 for name in modality_names}

    was_training = model.training
    model.eval()
    try:
        n = times.numel()
        zero_risk = times.new_zeros(n)
        baseline = batch_cox_loss(zero_risk, times, events).item()
        utilities = {}
        for modality in modality_names:
            masked = {key: value if key == modality else torch.zeros_like(value)
                      for key, value in inputs.items()}
            output = model(masked)
            risk = output["survival"].reshape(-1) if isinstance(output, dict) else output.reshape(-1)
            utilities[modality] = baseline - batch_cox_loss(risk, times, events).item()

        full_output = model(inputs)
        full_risk = (full_output["survival"].reshape(-1)
                     if isinstance(full_output, dict) else full_output.reshape(-1))
        v_full = baseline - batch_cox_loss(full_risk, times, events).item()
    finally:
        model.train(was_training)

    m0, m1 = modality_names
    phi0 = 0.5 * (utilities[m0] + v_full - utilities[m1])
    phi1 = 0.5 * (utilities[m1] + v_full - utilities[m0])
    pos0, pos1 = max(0.0, phi0), max(0.0, phi1)
    total = pos0 + pos1
    return {m0: pos0 / total, m1: pos1 / total} if total > 1e-12 else {m0: 0.5, m1: 0.5}
