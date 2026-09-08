import pytest

from apply_ranker_test1000_mvp import format_predictions, require_go


def test_require_go_rejects_any_failed_preregistered_gate():
    result = {"completed": True, "decision": "GO", "gates": {"a": True, "b": False}}
    with pytest.raises(ValueError, match="gate"):
        require_go(result)


def test_format_predictions_uses_native_source_for_selected_seed():
    decisions = [{
        "episode_id": "test1000:episode000001",
        "incumbent_seed": 4,
        "selected_seed": 2,
        "estimated_delta14": 0.02,
        "candidates": [{"seed": seed, "predicted_mean14": seed / 10,
                        "predicted_metrics14": {"metric": seed / 10},
                        "estimated_delta14": (seed - 4) / 10,
                        "incumbent": seed == 4, "selected": seed == 2,
                        "protected_veto": False, "score_kind": "prediction_not_official"}
                       for seed in (1, 4, 2, 3)],
        "score_kind": "prediction_not_official", "task": "test1000",
    }]
    sources = {1: {seed: {"video": f"/native/seed{seed}/episode1.mp4",
                           "sha256": str(seed) * 64} for seed in (1, 4, 2, 3)}}
    rows, details = format_predictions(decisions, sources)
    assert rows == [{"episode_id": 1, "prior_seed": 4, "selected_seed": 2,
                     "selected_video": "/native/seed2/episode1.mp4",
                     "selected_video_sha256": "2" * 64,
                     "estimated_delta14": 0.02}]
    assert details[0]["candidates"][2]["selected"] is True


def test_format_predictions_rejects_wrong_global_identity():
    with pytest.raises(ValueError, match="identity"):
        format_predictions([{"episode_id": "1"}], {})
