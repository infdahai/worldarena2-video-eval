"""Bounded Phase-M objective for Wan v8 direct action band."""
from __future__ import annotations
from torch import Tensor
from .wan_action_loss import weighted_flow_mse
from .wan_v71_cf import support_weighted_fm_energy, smooth_pairwise_ranking

def phase_m_objective(correct:Tensor, wrong:Tensor, target:Tensor, *, condition_support:Tensor, loss_weight:Tensor, valid_mask:Tensor, lambda_cf:float, tau:float=.1)->dict[str,Tensor]:
    if not lambda_cf>0: raise ValueError("lambda_cf must be positive")
    fm=weighted_flow_mse(correct,target,loss_weight=loss_weight,valid_mask=valid_mask)
    correct_e=support_weighted_fm_energy(correct,target,condition_support=condition_support,loss_weight=loss_weight,valid_mask=valid_mask)
    wrong_e=support_weighted_fm_energy(wrong,target,condition_support=condition_support,loss_weight=loss_weight,valid_mask=valid_mask)
    ranking=smooth_pairwise_ranking(correct_e,wrong_e,tau=tau)
    return {"loss":fm+float(lambda_cf)*ranking,"fm_loss":fm,"ranking_loss":ranking,"margin":(wrong_e-correct_e).mean()}
