"""CPU-only contract tests; no official scorer or external data required."""

import copy
import importlib

import numpy as np
import pytest


FEATURES = (
    "Instruction Following", "Interaction Quality", "Perspectivity",
    "Image Quality", "Aesthetic Quality", "Photometric Consistency",
    "Dynamic Degree", "Flow Score", "Motion Smoothness",
)
TARGETS = FEATURES + (
    "Subject Consistency", "Background Consistency", "Depth Accuracy",
    "Trajectory Accuracy", "Semantic Alignment",
)


def api():
    return importlib.import_module("worldarena_baseline.predicted14_ranker")


def features(x):
    return {name: x if index == 0 else 0.5 for index, name in enumerate(FEATURES)}


def row(episode, seed, x, target=None):
    result = {
        "task": "task_a", "episode_id": episode, "seed": seed,
        "features": features(x),
    }
    if target is not None:
        result["targets"] = {name: target for name in TARGETS}
    return result


def balanced_rows():
    return [row("a", 0, 0.0, 0.0)] + [row("b", s, 1.0, 1.0) for s in range(3)]


def inverse_rows():
    result = []
    for episode in range(30):
        for seed, x in enumerate((0.0, 0.3, 0.6, 1.0)):
            item = row(f"fit_{episode}", seed, x)
            item["targets"] = {**features(x), **{name: 1.0 - x for name in TARGETS[9:]}}
            result.append(item)
    return result


def test_missing_five_metrics_can_outweigh_raw9_gain():
    ranker = api()
    model = ranker.fit_ranker(inverse_rows())
    low = ranker.predict_features(model, features(0.0))
    high = ranker.predict_features(model, features(1.0))
    assert sum(features(1.0).values()) > sum(features(0.0).values())
    assert low["predicted_mean14"] > high["predicted_mean14"]
    assert high["predicted_metrics14"]["Instruction Following"] > low["predicted_metrics14"]["Instruction Following"]
    assert set(low["predicted_metrics14"]) == set(TARGETS)
    assert low["score_kind"] == "prediction_not_official"


def test_fit_is_episode_balanced_and_ridge_alpha_ten():
    model = api().fit_ranker(balanced_rows())
    assert model["scaler"]["mean"][0] == pytest.approx(0.5)
    assert model["scaler"]["scale"][0] == pytest.approx(0.5)
    # Two equally weighted episodes: beta=1/(2+10); unpenalized intercept=.5.
    prediction = api().predict_features(model, features(1.0))
    assert prediction["predicted_mean14"] == pytest.approx(7 / 12)


def test_scaler_is_fit_only_and_predictions_are_bounded():
    rows = balanced_rows()
    for item in rows:
        item["features"] = features(0.49 if item["episode_id"] == "a" else 0.51)
    model = api().fit_ranker(rows)
    original = copy.deepcopy(model)
    extreme = features(1.0)
    prediction = api().predict_features(model, extreme)
    assert model == original
    assert prediction["predicted_mean14"] == 1.0
    assert all(0 <= value <= 1 for value in prediction["predicted_metrics14"].values())


@pytest.mark.parametrize("field", ["targets", "gt", "seed", "task", "Subject Consistency"])
def test_inference_rejects_non_deployable_features(field):
    model = api().fit_ranker(balanced_rows())
    bad = {**features(0.5), field: 0.5}
    with pytest.raises(ValueError, match="feature schema"):
        api().predict_features(model, bad)


@pytest.mark.parametrize("mutation", ["jepa", "missing", "duplicate", "one_episode", "nonfinite", "target_range", "feature_range", "extra_field", "feature_schema", "numeric_episode"])
def test_training_rejects_invalid_contracts(mutation):
    rows = balanced_rows()
    if mutation == "jepa":
        rows[0]["targets"]["JEPA"] = 0.5
    elif mutation == "missing":
        rows[0]["targets"].pop("Depth Accuracy")
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "one_episode":
        rows = rows[1:]
    elif mutation == "nonfinite":
        rows[0]["features"]["Flow Score"] = np.nan
    elif mutation == "target_range":
        rows[0]["targets"]["Image Quality"] = 1.01
    elif mutation == "feature_range":
        rows[0]["features"]["Flow Score"] = 1.01
    elif mutation == "extra_field":
        rows[0]["gt_video"] = "private.mp4"
    elif mutation == "feature_schema":
        rows[0]["features"]["Depth Accuracy"] = 0.5
    elif mutation == "numeric_episode":
        rows[0]["episode_id"] = 123
    with pytest.raises(ValueError):
        api().fit_ranker(rows)


def test_task_seed_and_episode_names_are_not_features():
    rows = balanced_rows()
    renamed = copy.deepcopy(rows)
    for item in renamed:
        item.update(task="new_task", episode_id="renamed_" + item["episode_id"], seed=item["seed"] + 100)
    first, second = (api().fit_ranker(data) for data in (rows, renamed))
    assert api().predict_features(first, features(0.7)) == api().predict_features(second, features(0.7))


def test_input_boundary_tolerance_is_clipped():
    rows = balanced_rows()
    rows[0]["features"]["Instruction Following"] = -0.0005
    rows[0]["targets"]["Image Quality"] = -0.0005
    expected = api().fit_ranker(balanced_rows())
    model = api().fit_ranker(rows)
    assert api().predict_features(model, features(1.0005)) == api().predict_features(expected, features(1.0))


def test_constant_features_use_unit_scale_despite_roundoff():
    rows = inverse_rows()
    for item in rows:
        item["features"]["Flow Score"] = 0.1
    model = api().fit_ranker(rows)
    assert model["scaler"]["scale"][7] == 1.0


def test_all_four_seeds_are_scored_and_best_prediction_selected():
    model = api().fit_ranker(inverse_rows())
    candidates = [row("heldout_a", seed, x) for seed, x in ((31, 1.0), (52, 0.6), (89, 0.0), (17, 0.3))]
    result = api().rank_candidates(model, candidates, {"heldout_a": 31}, delta_threshold=0.003)[0]
    assert result["selected_seed"] == 89
    assert result["incumbent_seed"] == 31
    assert result["estimated_delta14"] > 0.003
    assert {item["seed"] for item in result["candidates"]} == {17, 31, 52, 89}
    assert sum(item["selected"] for item in result["candidates"]) == 1
    assert result["score_kind"] == "prediction_not_official"


def test_equal_predictions_keep_incumbent_even_if_not_first_seed():
    model = api().fit_ranker(balanced_rows())
    candidates = [row("heldout_a", seed, 0.5) for seed in range(4)]
    result = api().rank_candidates(model, candidates, {"heldout_a": 3})[0]
    assert result["selected_seed"] == 3
    assert result["estimated_delta14"] == 0.0


def test_improvement_must_be_strictly_greater_than_fixed_threshold():
    model = api().fit_ranker(balanced_rows())
    candidates = [row("heldout_a", seed, seed / 3) for seed in range(4)]
    # Deliberately place threshold on the observed boundary to catch >= selection.
    boundary = api().predict_features(model, features(1.0))["predicted_mean14"] - api().predict_features(model, features(0.0))["predicted_mean14"]
    assert api().rank_candidates(model, candidates, {"heldout_a": 0}, delta_threshold=boundary)[0]["selected_seed"] == 0
    assert api().rank_candidates(model, candidates, {"heldout_a": 0}, delta_threshold=boundary - 0.0001)[0]["selected_seed"] == 3


def test_optional_predicted_protection_can_veto_switch():
    model = api().fit_ranker(inverse_rows())
    candidates = [row("heldout_a", seed, x) for seed, x in enumerate((1.0, 0.6, 0.3, 0.0))]
    result = api().rank_candidates(model, candidates, {"heldout_a": 0}, protected_max_drop={"Instruction Following": 0.0})[0]
    assert result["selected_seed"] == 0
    assert all(item["protected_veto"] for item in result["candidates"] if item["seed"] != 0)


@pytest.mark.parametrize("mutation", ["three", "five", "duplicate", "targets", "missing_incumbent", "nonfinite", "negative_threshold", "jepa_protection", "conflicting_task"])
def test_ranking_rejects_invalid_candidates_and_policy(mutation):
    model = api().fit_ranker(balanced_rows())
    candidates = [row("heldout_a", seed, 0.5) for seed in range(4)]
    incumbents, policy = {"heldout_a": 0}, {}
    if mutation == "three":
        candidates.pop()
    elif mutation == "five":
        candidates.append(row("heldout_a", 4, 0.5))
    elif mutation == "duplicate":
        candidates[-1] = copy.deepcopy(candidates[0])
    elif mutation == "targets":
        candidates[0]["targets"] = {name: 0.5 for name in TARGETS}
    elif mutation == "missing_incumbent":
        incumbents["heldout_a"] = 999
    elif mutation == "nonfinite":
        policy["delta_threshold"] = np.nan
    elif mutation == "negative_threshold":
        policy["delta_threshold"] = -0.1
    elif mutation == "jepa_protection":
        policy["protected_max_drop"] = {"JEPA": 0.1}
    elif mutation == "conflicting_task":
        candidates[0]["task"] = "other_task"
    with pytest.raises(ValueError):
        api().rank_candidates(model, candidates, incumbents, **policy)


def test_serialized_roundtrip_preserves_predictions_and_training_provenance():
    model = api().fit_ranker(balanced_rows())
    loaded = api().loads_model(api().dumps_model(model))
    assert loaded["train_episode_ids"] == ["a", "b"]
    assert len(loaded["data_digest"]) == 64
    assert loaded["feature_schema"] == list(FEATURES)
    assert loaded["target_schema"] == list(TARGETS)
    assert api().predict_features(loaded, features(0.7)) == api().predict_features(model, features(0.7))
    reordered = api().fit_ranker(list(reversed(balanced_rows())))
    assert reordered["data_digest"] == model["data_digest"]
    changed = balanced_rows()
    changed[0]["targets"]["Depth Accuracy"] = 0.2
    assert api().fit_ranker(changed)["data_digest"] != model["data_digest"]


@pytest.mark.parametrize("mutation", ["features", "targets", "alpha", "scale", "coefficients", "nonfinite", "train_ids", "digest"])
def test_model_drift_and_corruption_are_rejected(mutation):
    model = api().fit_ranker(balanced_rows())
    if mutation == "features":
        model["feature_schema"].reverse()
    elif mutation == "targets":
        model["target_schema"][-1] = "JEPA"
    elif mutation == "alpha":
        model["alpha"] = 1
    elif mutation == "scale":
        model["scaler"]["scale"][0] = 0
    elif mutation == "coefficients":
        model["coefficients"].pop()
    elif mutation == "nonfinite":
        model["intercept"][0] = float("inf")
    elif mutation == "train_ids":
        model["train_episode_ids"] = ["a", "a"]
    elif mutation == "digest":
        model["data_digest"] = "invalid"
    with pytest.raises(ValueError):
        api().predict_features(model, features(0.5))
    with pytest.raises(ValueError):
        api().dumps_model(model)
