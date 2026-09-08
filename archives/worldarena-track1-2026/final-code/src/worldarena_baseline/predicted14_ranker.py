"""Independent CPU generated9 -> corrected14 predictor, NOT official scoring.

``model = fit_ranker(training_rows)`` fits episode-balanced StandardScaler and
multi-target ridge (alpha=10, unpenalized intercept). Training rows contain only
task, globally unique string episode_id, integer seed, features and targets.
Features are the nine *normalized raw generated* scores, never corrected/GT
inputs; targets are the fourteen corrected metrics below, never collection JEPA.
All input scores must be finite in [-.001, 1.001] and are clipped to [0, 1].

``predict_features(model, features)`` accepts ONLY the feature mapping. It returns
clipped predicted metrics and their arithmetic mean, not an official 15-score.
Real collection-level JEPA and full15 acceptance remain an external gate.
No task/episode/seed metadata is ever fitted as a feature.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping
import hashlib
import json
from numbers import Real

import numpy as np


FEATURE_SCHEMA = (
    "Instruction Following", "Interaction Quality", "Perspectivity",
    "Image Quality", "Aesthetic Quality", "Photometric Consistency",
    "Dynamic Degree", "Flow Score", "Motion Smoothness",
)
TARGET_SCHEMA = FEATURE_SCHEMA + (
    "Subject Consistency", "Background Consistency", "Depth Accuracy",
    "Trajectory Accuracy", "Semantic Alignment",
)
ALPHA = 10.0
SCORE_KIND = "prediction_not_official"


def _scores(values, schema, name):
    if not isinstance(values, Mapping) or set(values) != set(schema):
        raise ValueError(f"{name} schema mismatch; expected exactly {schema}")
    if any(isinstance(values[k], bool) or not isinstance(values[k], Real) for k in schema):
        raise ValueError(f"{name} values must be numeric")
    array = np.asarray([values[k] for k in schema], dtype=float)
    if not np.isfinite(array).all() or np.any(array < -0.001) or np.any(array > 1.001):
        raise ValueError(f"{name} values must be finite in [-.001, 1.001]")
    return np.clip(array, 0.0, 1.0)


def _rows(rows, training):
    result, seen, tasks = [], set(), {}
    required = {"task", "episode_id", "seed", "features"} | ({"targets"} if training else set())
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError(f"row schema mismatch; expected exactly {sorted(required)}")
        task, episode, seed = row["task"], row["episode_id"], row["seed"]
        if any(not isinstance(x, str) or not x.strip() for x in (task, episode)):
            raise ValueError("task and globally unique episode_id must be nonempty strings")
        if episode.isdecimal() or type(seed) is not int or seed < 0:
            raise ValueError("episode_id cannot be a numeric id; seed must be a nonnegative integer")
        if (episode, seed) in seen or episode in tasks and tasks[episode] != task:
            raise ValueError("duplicate episode+seed or episode_id shared across tasks")
        seen.add((episode, seed))
        tasks[episode] = task
        item = dict(task=task, episode_id=episode, seed=seed)
        item["features"] = dict(zip(FEATURE_SCHEMA, _scores(row["features"], FEATURE_SCHEMA, "feature")))
        if training:
            item["targets"] = dict(zip(TARGET_SCHEMA, _scores(row["targets"], TARGET_SCHEMA, "target")))
        result.append(item)
    if not result:
        raise ValueError("rows must not be empty")
    return sorted(result, key=lambda row: (row["episode_id"], row["seed"]))


def fit_ranker(rows):
    """Fit only these rows, with each row weight=1/candidates_in_its_episode.

    Returns a JSON-compatible dict recording schemas, normalization, coefficients,
    training episode IDs, and SHA256 of canonical *clipped* training records.
    """
    rows = _rows(rows, training=True)
    counts = Counter(row["episode_id"] for row in rows)
    if len(counts) < 2:
        raise ValueError("training requires at least two episodes")
    weights = np.array([1.0 / counts[row["episode_id"]] for row in rows])
    x = np.array([[row["features"][name] for name in FEATURE_SCHEMA] for row in rows])
    y = np.array([[row["targets"][name] for name in TARGET_SCHEMA] for row in rows])
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    # StandardScaler's numerical constant-feature bound; weighted sample count.
    n_eps = weights.sum() * np.finfo(float).eps
    scale[variance <= n_eps * variance + (n_eps * mean) ** 2] = 1.0
    z = (x - mean) / scale
    intercept = np.average(y, axis=0, weights=weights)
    coefficients = np.linalg.solve(
        z.T @ (weights[:, None] * z) + ALPHA * np.eye(len(FEATURE_SCHEMA)),
        z.T @ (weights[:, None] * (y - intercept)),
    )
    return {
        "version": 1, "score_kind": SCORE_KIND,
        "feature_schema": list(FEATURE_SCHEMA), "target_schema": list(TARGET_SCHEMA),
        "alpha": ALPHA, "input_policy": "finite_[-.001,1.001]_clip_[0,1]",
        "scaler": {"mean": mean.tolist(), "scale": scale.tolist()},
        "coefficients": coefficients.tolist(), "intercept": intercept.tolist(),
        "train_episode_ids": sorted(counts),
        "data_digest": hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
    }


def predict_features(model, features):
    """Predict fourteen metrics using only the exact deployable nine-feature map."""
    mean, scale, coefficients, intercept = _model_arrays(model)
    x = _scores(features, FEATURE_SCHEMA, "feature")
    predictions = np.clip(((x - mean) / scale) @ coefficients + intercept, 0, 1)
    return {"score_kind": SCORE_KIND, "predicted_mean14": float(predictions.mean()),
            "predicted_metrics14": dict(zip(TARGET_SCHEMA, predictions.tolist()))}


def _model_arrays(model):
    try:
        if (model["version"] != 1 or model["alpha"] != ALPHA
                or model["score_kind"] != SCORE_KIND
                or model["feature_schema"] != list(FEATURE_SCHEMA)
                or model["target_schema"] != list(TARGET_SCHEMA)
                or model["input_policy"] != "finite_[-.001,1.001]_clip_[0,1]"):
            raise ValueError("model schema or policy drift")
        ids, digest = model["train_episode_ids"], model["data_digest"]
        if (not isinstance(ids, list) or len(ids) < 2
                or any(not isinstance(x, str) or not x.strip() or x.isdecimal() for x in ids)
                or len(set(ids)) != len(ids)
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid training provenance")
        arrays = [np.asarray(value, dtype=float) for value in (
            model["scaler"]["mean"], model["scaler"]["scale"],
            model["coefficients"], model["intercept"],
        )]
        for array, shape in zip(arrays, ((9,), (9,), (9, 14), (14,))):
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError("invalid model array shape or nonfinite value")
        if np.any(arrays[1] <= 0) or np.any(arrays[0] < 0) or np.any(arrays[0] > 1):
            raise ValueError("invalid scaler range")
        return arrays
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("invalid model") from exc


def dumps_model(model):
    """Validate and serialize to JSON text; caller chooses any output file."""
    _model_arrays(model)
    return json.dumps(model, sort_keys=True, allow_nan=False)


def loads_model(text):
    """Load JSON and reject schema, policy, shape and nonfinite model drift."""
    model = json.loads(text)
    _model_arrays(model)
    return model


def rank_candidates(model, rows, incumbents, *, delta_threshold=0.0, protected_max_drop=None):
    """Score all four distinct seeds per episode without targets or GT inputs.

    ``incumbents`` maps global episode_id to P0 seed; it must match these episodes.
    Select the best non-vetoed candidate only if its predicted_mean14 improvement
    strictly exceeds delta_threshold (fixed by caller, never fitted here). Ties
    keep P0; tied improving challengers use the smallest seed deterministically.
    Optional ``protected_max_drop={metric: tolerance}`` vetoes *predicted* metric
    regressions against P0. It is disabled by default and is not an official gate.
    """
    protected = {} if protected_max_drop is None else protected_max_drop
    if not isinstance(protected, Mapping) or not set(protected) <= set(TARGET_SCHEMA):
        raise ValueError("protected metrics must belong to target schema (no JEPA)")
    for value in [delta_threshold, *protected.values()]:
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
            raise ValueError("thresholds must be finite nonnegative numbers")
    grouped = defaultdict(list)
    for row in _rows(rows, training=False):
        grouped[row["episode_id"]].append(row)
    if not isinstance(incumbents, Mapping) or set(incumbents) != set(grouped):
        raise ValueError("incumbent episode schema mismatch")
    results = []
    for episode, candidates in grouped.items():
        seed = incumbents[episode]
        if len(candidates) != 4 or type(seed) is not int or seed not in {row["seed"] for row in candidates}:
            raise ValueError("each episode needs exactly four seeds including its incumbent")
        scored = [{"seed": row["seed"], **predict_features(model, row["features"])} for row in candidates]
        incumbent = next(item for item in scored if item["seed"] == seed)
        selected = incumbent
        for candidate in scored:
            candidate["estimated_delta14"] = candidate["predicted_mean14"] - incumbent["predicted_mean14"]
            candidate["protected_veto"] = any(
                incumbent["predicted_metrics14"][name] - candidate["predicted_metrics14"][name] > limit
                for name, limit in protected.items()
            )
            if (not candidate["protected_veto"]
                    and candidate["estimated_delta14"] > delta_threshold
                    and candidate["predicted_mean14"] > selected["predicted_mean14"]):
                selected = candidate
        for candidate in scored:
            candidate["incumbent"] = candidate is incumbent
            candidate["selected"] = candidate is selected
        results.append({
            "score_kind": SCORE_KIND, "task": candidates[0]["task"], "episode_id": episode,
            "selected_seed": selected["seed"], "incumbent_seed": seed,
            "estimated_delta14": selected["estimated_delta14"], "candidates": scored,
        })
    return results
