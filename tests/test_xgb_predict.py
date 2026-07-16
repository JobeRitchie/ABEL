"""GPU-trained XGBoost boosters must predict on the CPU, without the device warning.

A booster fitted with device="cuda" keeps that device, so predicting with it on a
NumPy array copies the whole feature matrix host→device on every call.  XGBoost warns
once ("Falling back to prediction using DMatrix due to mismatched devices") and the
copy makes prediction ~7x SLOWER than simply scoring on the CPU.

The GPU tests below are skipped on a machine without CUDA; the wrapper/idempotence
tests run everywhere.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost")
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from abel.utils.xgb_predict import ensure_cpu_prediction, predict_proba  # noqa: E402

DEVICE_WARNING = "mismatched devices"


def _has_cuda() -> bool:
    try:
        x = np.random.default_rng(0).random((32, 4), dtype=np.float32)
        y = (x[:, 0] > 0.5).astype(int)
        XGBClassifier(n_estimators=2, device="cuda", tree_method="hist").fit(x, y)
        return True
    except Exception:
        return False


HAS_CUDA = _has_cuda()
gpu_only = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA-capable XGBoost")


@pytest.fixture()
def data():
    rng = np.random.default_rng(0)
    x = rng.random((400, 12), dtype=np.float32)
    y = (x[:, 0] + 0.2 * rng.random(400) > 0.6).astype(int)
    return x, y


def _fit(device: str, x, y):
    return XGBClassifier(n_estimators=25, max_depth=3, tree_method="hist",
                         device=device).fit(x, y)


def _booster_device(est) -> str:
    cfg = est.get_booster().save_config()
    import json
    return json.loads(cfg)["learner"]["generic_param"]["device"]


@gpu_only
def test_gpu_booster_predicts_without_device_warning(data):
    """The regression: predict_proba on a CUDA booster with CPU input must not warn."""
    x, y = data
    est = _fit("cuda", x, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predict_proba(est, x)
    assert not [w for w in caught if DEVICE_WARNING in str(w.message)], \
        "device-mismatch warning still raised"
    # XGBoost only emits that warning once per process, so an earlier test could use
    # it up and let this one pass vacuously.  The booster's device is the same fact
    # stated deterministically — assert it too.
    assert _booster_device(est) == "cpu"


@gpu_only
def test_probabilities_unchanged_by_moving_to_cpu(data):
    """Scoring on the CPU must not change the model's answers."""
    x, y = data
    est = _fit("cuda", x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = est.predict_proba(x)      # GPU booster, CPU input (the slow path)
    after = predict_proba(est, x)
    assert np.allclose(before, after, atol=1e-5)


@gpu_only
def test_booster_moves_but_estimator_still_trains_on_gpu(data):
    """Only the booster's device changes — a refit still uses the GPU."""
    x, y = data
    est = _fit("cuda", x, y)
    ensure_cpu_prediction(est)
    assert _booster_device(est) == "cpu"
    assert est.get_params()["device"] == "cuda", "refit would no longer use the GPU"


@gpu_only
def test_calibrated_wrapper_is_moved(data):
    """CalibratedClassifierCV(cv='prefit') is how ABEL ships most models."""
    x, y = data
    base = _fit("cuda", x[:300], y[:300])
    cal = CalibratedClassifierCV(base, cv="prefit", method="sigmoid").fit(x[300:], y[300:])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predict_proba(cal, x)
    assert not [w for w in caught if DEVICE_WARNING in str(w.message)]
    assert _booster_device(cal.calibrated_classifiers_[0].estimator) == "cpu"


def test_cpu_model_is_untouched(data):
    """A CPU-trained model already works; the helper must be a no-op on the answers."""
    x, y = data
    est = _fit("cpu", x, y)
    assert np.allclose(est.predict_proba(x), predict_proba(est, x))


def test_pipeline_is_reached(data):
    """A Pipeline ending in XGBoost is unwrapped too."""
    x, y = data
    pipe = Pipeline([("clf", XGBClassifier(n_estimators=10, tree_method="hist"))]).fit(x, y)
    assert ensure_cpu_prediction(pipe) == 1


def test_idempotent_and_safe_on_non_xgboost(data):
    """Repeat calls do no work, and a non-XGBoost model is left alone."""
    from sklearn.linear_model import LogisticRegression

    x, y = data
    est = _fit("cpu", x, y)
    assert ensure_cpu_prediction(est) == 1
    assert ensure_cpu_prediction(est) == 0, "should not re-set the param every call"

    lr = LogisticRegression(max_iter=200).fit(x, y)
    assert ensure_cpu_prediction(lr) == 0
    assert np.allclose(lr.predict_proba(x), predict_proba(lr, x))


def test_unfitted_model_does_not_raise(data):
    """An unfitted estimator has no booster; the helper must not blow up."""
    assert ensure_cpu_prediction(XGBClassifier()) == 0
