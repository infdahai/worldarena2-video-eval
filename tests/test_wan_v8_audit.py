from worldarena_baseline.wan_v8_audit import aggregate_v8_audit, hard_shift_energy

def test_hard_shift_is_more_competitive_wrong():
    assert hard_shift_energy({"shift_plus_energy":.9,"shift_minus_energy":.7}) == .7

def test_audit_has_exact_twenty_rows():
    row={"correct_energy":1.,"reverse_energy":1.2,"shift_plus_energy":1.2,"shift_minus_energy":1.1,"swap_energy":1.2,"routing_retention":1.,"fm_regression":0.,"position_regression":0.,"velocity_regression":0.}
    assert aggregate_v8_audit([row]*20,step=100)["decision"]["pass"]
