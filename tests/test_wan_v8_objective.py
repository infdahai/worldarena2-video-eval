import pytest
torch=pytest.importorskip("torch")
from worldarena_baseline.wan_v8_objective import phase_m_objective

def test_phase_m_rewards_lower_correct_energy():
 target=torch.zeros(1,2,21,30,40); correct=torch.full_like(target,.1,requires_grad=True); wrong=torch.full_like(target,.5,requires_grad=True)
 support=torch.ones(1,2,21,15,20); weight=torch.ones(1,1,21,30,40); valid=torch.ones_like(weight)
 result=phase_m_objective(correct,wrong,target,condition_support=support,loss_weight=weight,valid_mask=valid,lambda_cf=1.)
 assert result["margin"]>0 and result["loss"].isfinite()
 result["loss"].backward(); assert correct.grad is not None and wrong.grad is not None
