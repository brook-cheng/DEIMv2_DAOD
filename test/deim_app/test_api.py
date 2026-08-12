"""Tests for ``deim_app.api.DetectionModel`` — the pure-delegation facade.

The facade wraps a ``DetectionAdapter`` and adds:
  * ``load()`` chaining (returns ``self``);
  * a ``_is_loaded`` gate on ``predict``;
  * ``predict_filtered`` with config-default resolution and class-name
    validation against dataset metadata;
  * pass-through ``box_mode`` / ``metadata`` properties.

All tests use :class:`~test.deim_app.conftest.FakeAdapter` — a spy that
records every call and returns a deterministic 3-detection collection
(scores [0.9, 0.8, 0.3]). No engine objects are constructed.
"""

from __future__ import annotations

import pytest

from deim_app.api import DetectionModel
from deim_app.config.schema import InferenceConfig
from deim_app.errors import AppConfigError, InferenceBackendError
from deim_app.predictions.collection import PredictionCollection

from _facade_fakes import FakeAdapter


# ---------------------------------------------------------------------------
# Brief's exact test
# ---------------------------------------------------------------------------


def test_predict_returns_full_collection_and_filtered_view_is_separate(
    fake_adapter: FakeAdapter,
) -> None:
    """predict() returns the full 3-detection collection; predict_filtered()
    returns an independent filtered view — the full collection is unaffected."""
    model = DetectionModel(fake_adapter)
    full = model.predict("images")
    filtered = model.predict_filtered("images", score_threshold=0.5, top_k=1)
    assert len(full.predictions[0].detections) == 3
    assert len(filtered.predictions[0].detections) == 1


def test_predict_does_not_apply_any_filtering(fake_adapter: FakeAdapter) -> None:
    """predict() is a pure pass-through — no score threshold, no top-k."""
    model = DetectionModel(fake_adapter)
    result = model.predict("images")
    assert isinstance(result, PredictionCollection)
    assert len(result.predictions[0].detections) == 3


# ---------------------------------------------------------------------------
# Not-loaded gate
# ---------------------------------------------------------------------------


def test_predict_before_load_raises_inference_backend_error(
    unloaded_fake_adapter: FakeAdapter,
) -> None:
    """predict() must refuse to run before load() has built the model."""
    model = DetectionModel(unloaded_fake_adapter)
    with pytest.raises(InferenceBackendError, match="not loaded"):
        model.predict("images")


def test_predict_filtered_before_load_raises_inference_backend_error(
    unloaded_fake_adapter: FakeAdapter,
) -> None:
    """predict_filtered() also gates on _is_loaded because it calls predict()."""
    model = DetectionModel(unloaded_fake_adapter)
    with pytest.raises(InferenceBackendError, match="not loaded"):
        model.predict_filtered("images")


# ---------------------------------------------------------------------------
# Config-default resolution in predict_filtered
# ---------------------------------------------------------------------------


def test_predict_filtered_uses_config_score_threshold_when_omitted() -> None:
    """Omitting score_threshold pulls the default from inference config.

    With config score_threshold=0.5 and collection scores [0.9, 0.8, 0.3]:
    two detections survive (>= 0.5), confirming the default was applied
    (without it, all three would pass since filter(None) is a no-op).
    """
    adapter = FakeAdapter(inference=InferenceConfig(score_threshold=0.5, top_k=300))
    model = DetectionModel(adapter)
    result = model.predict_filtered("images")
    assert len(result.predictions[0].detections) == 2


def test_predict_filtered_uses_config_top_k_when_omitted() -> None:
    """Omitting top_k pulls the default from inference config.

    With config top_k=2 and default score_threshold=0.25 (all three pass),
    top_k truncates to the two highest-scoring detections.
    """
    adapter = FakeAdapter(inference=InferenceConfig(top_k=2))
    model = DetectionModel(adapter)
    result = model.predict_filtered("images")
    assert len(result.predictions[0].detections) == 2


def test_predict_filtered_explicit_score_threshold_overrides_config() -> None:
    """Explicit score_threshold takes precedence over the config default."""
    adapter = FakeAdapter(inference=InferenceConfig(score_threshold=0.5))
    model = DetectionModel(adapter)
    result = model.predict_filtered("images", score_threshold=0.85)
    # Only 0.9 >= 0.85 → 1 detection
    assert len(result.predictions[0].detections) == 1


def test_predict_filtered_explicit_top_k_overrides_config() -> None:
    """Explicit top_k takes precedence over the config default."""
    adapter = FakeAdapter(inference=InferenceConfig(top_k=2))
    model = DetectionModel(adapter)
    result = model.predict_filtered("images", top_k=1)
    # top_k=1 keeps only the highest-scoring detection (0.9)
    assert len(result.predictions[0].detections) == 1


def test_predict_filtered_combined_score_and_top_k() -> None:
    """score_threshold then top_k: filter first, then truncate the survivors."""
    model = DetectionModel(FakeAdapter())
    # filter >= 0.5 → [0.9, 0.8], then top_k(1) → [0.9]
    result = model.predict_filtered("images", score_threshold=0.5, top_k=1)
    assert len(result.predictions[0].detections) == 1
    assert result.predictions[0].detections[0].score == 0.9


# ---------------------------------------------------------------------------
# Class-filter validation
# ---------------------------------------------------------------------------


def test_class_filter_accepts_known_names(fake_adapter: FakeAdapter) -> None:
    """A class_filter of known names is accepted and filters accordingly."""
    model = DetectionModel(fake_adapter)
    result = model.predict_filtered("images", class_filter=["c1"])
    assert len(result.predictions[0].detections) == 1
    assert result.predictions[0].detections[0].class_name == "c1"


def test_class_filter_rejects_unknown_names(fake_adapter: FakeAdapter) -> None:
    """Unknown class names raise AppConfigError before inference runs."""
    model = DetectionModel(fake_adapter)
    with pytest.raises(AppConfigError, match="unknown class name"):
        model.predict_filtered("images", class_filter=["totally_unknown"])
    # Inference must not have started.
    assert fake_adapter.predict_calls == []


def test_class_filter_rejects_partial_unknown_names(fake_adapter: FakeAdapter) -> None:
    """A mix of known and unknown names still fails fast on the unknown one."""
    model = DetectionModel(fake_adapter)
    with pytest.raises(AppConfigError, match="c0.*totally_unknown|totally_unknown.*c0"):
        model.predict_filtered("images", class_filter=["c0", "totally_unknown"])


def test_class_filter_from_config_default_is_applied() -> None:
    """When class_filter is omitted, inference.class_filter from config is used."""
    adapter = FakeAdapter(inference=InferenceConfig(class_filter=("c2",)))
    model = DetectionModel(adapter)
    result = model.predict_filtered("images")
    assert len(result.predictions[0].detections) == 1
    assert result.predictions[0].detections[0].class_name == "c2"


def test_class_filter_from_config_default_is_validated() -> None:
    """An unknown name in inference.class_filter is still caught at filter time."""
    adapter = FakeAdapter(
        inference=InferenceConfig(class_filter=("not_a_real_class",)),
    )
    model = DetectionModel(adapter)
    with pytest.raises(AppConfigError, match="unknown class name"):
        model.predict_filtered("images")


# ---------------------------------------------------------------------------
# Delegation: train / evaluate / load / properties
# ---------------------------------------------------------------------------


def test_train_delegates_to_adapter(fake_adapter: FakeAdapter) -> None:
    model = DetectionModel(fake_adapter)
    model.train()
    assert fake_adapter.train_called is True


def test_evaluate_delegates_to_adapter(fake_adapter: FakeAdapter) -> None:
    model = DetectionModel(fake_adapter)
    model.evaluate("ckpt/path.pt")
    assert fake_adapter.evaluate_calls == ["ckpt/path.pt"]


def test_evaluate_delegates_with_none_checkpoint(fake_adapter: FakeAdapter) -> None:
    model = DetectionModel(fake_adapter)
    model.evaluate()
    assert fake_adapter.evaluate_calls == [None]


def test_load_returns_self_for_chaining(fake_adapter: FakeAdapter) -> None:
    model = DetectionModel(fake_adapter)
    result = model.load("weights.pt")
    assert result is model
    assert fake_adapter.load_calls == [("weights.pt", True)]


def test_load_passes_prefer_ema_false(fake_adapter: FakeAdapter) -> None:
    model = DetectionModel(fake_adapter)
    model.load("weights.pt", prefer_ema=False)
    assert fake_adapter.load_calls == [("weights.pt", False)]


def test_load_defaults_pass_none_checkpoint(fake_adapter: FakeAdapter) -> None:
    model = DetectionModel(fake_adapter)
    model.load()
    assert fake_adapter.load_calls == [(None, True)]


def test_box_mode_property_delegates(fake_adapter: FakeAdapter) -> None:
    model = DetectionModel(fake_adapter)
    assert model.box_mode == fake_adapter.box_mode


def test_metadata_property_delegates(fake_adapter: FakeAdapter) -> None:
    model = DetectionModel(fake_adapter)
    assert model.metadata is fake_adapter.metadata


def test_metadata_property_returns_dataset_metadata(fake_adapter: FakeAdapter) -> None:
    """The metadata property returns the full DatasetMetadata from the adapter."""
    model = DetectionModel(fake_adapter)
    md = model.metadata
    assert md.box_mode == "hbb"
    assert md.num_classes == 3
    assert set(md.class_names_by_label.values()) == {"c0", "c1", "c2"}


# ---------------------------------------------------------------------------
# Chaining integration: load → predict
# ---------------------------------------------------------------------------


def test_load_then_predict_works(fake_adapter: FakeAdapter) -> None:
    """After load(), predict() succeeds and returns the full collection."""
    model = DetectionModel(FakeAdapter(is_loaded=False))
    model.load("ckpt")
    result = model.predict("images")
    assert len(result.predictions[0].detections) == 3


def test_predict_passes_batch_size_to_adapter(fake_adapter: FakeAdapter) -> None:
    """The facade forwards batch_size to the adapter's predict()."""
    model = DetectionModel(fake_adapter)
    model.predict("images", batch_size=4)
    assert fake_adapter.predict_calls == [("images", 4)]


def test_predict_filtered_passes_batch_size_to_adapter(
    fake_adapter: FakeAdapter,
) -> None:
    """predict_filtered also forwards batch_size through to the adapter."""
    model = DetectionModel(fake_adapter)
    model.predict_filtered("images", batch_size=2)
    assert fake_adapter.predict_calls[-1] == ("images", 2)
