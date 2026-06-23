"""Dataset construction for center-frame temporal refinement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from abel.temporal_refinement.window_sampler import WindowSample


@dataclass
class CenterFrameDataset:
    """Container for windowed features, labels, and sample metadata."""

    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        raw = self.metadata.iloc[int(idx)].to_dict()
        meta = {str(k): v for k, v in raw.items()}
        return self.X[idx], int(self.y[idx]), meta


@dataclass
class DatasetBuildResult:
    dataset: CenterFrameDataset
    skipped_samples: int


def _window_with_padding(
    frame_features: np.ndarray,
    center_frame: int,
    window_frames: int,
) -> tuple[np.ndarray, int, int]:
    n_frames = int(frame_features.shape[0])
    n_features = int(frame_features.shape[1])
    left = window_frames // 2
    right = window_frames - left - 1

    start = int(center_frame) - left
    end = int(center_frame) + right

    pad_left = max(0, -start)
    pad_right = max(0, end - (n_frames - 1))

    s = max(0, start)
    e = min(n_frames - 1, end)

    window = frame_features[s : e + 1]
    if pad_left > 0 or pad_right > 0:
        window = np.pad(
            window,
            pad_width=((pad_left, pad_right), (0, 0)),
            mode="edge",
        )
    if window.shape[0] != window_frames:
        # Safety net for unusual degenerate windows.
        if window.shape[0] < window_frames:
            missing = window_frames - window.shape[0]
            window = np.pad(window, ((0, missing), (0, 0)), mode="edge")
        else:
            window = window[:window_frames]

    return np.asarray(window, dtype=np.float32), int(pad_left), int(pad_right)


def build_centerframe_dataset(
    sampled_windows: list[WindowSample],
    session_features: dict[str, np.ndarray],
    labels_by_session_frame: dict[tuple[str, int], int],
    window_frames: int,
) -> DatasetBuildResult:
    """Build an in-memory dataset from sampled center frames."""
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    meta_rows: list[dict[str, Any]] = []
    skipped = 0

    for sample in sampled_windows:
        features = session_features.get(sample.session_id)
        if features is None:
            skipped += 1
            continue
        if sample.center_frame < 0 or sample.center_frame >= int(features.shape[0]):
            skipped += 1
            continue

        key = (sample.session_id, int(sample.center_frame))
        if key not in labels_by_session_frame:
            skipped += 1
            continue

        window, pad_left, pad_right = _window_with_padding(
            frame_features=features,
            center_frame=int(sample.center_frame),
            window_frames=int(window_frames),
        )
        # Use the true model label map (not source) to allow explicit exclusions upstream.
        y_value = int(labels_by_session_frame[key])

        X_rows.append(window)
        y_rows.append(y_value)
        meta_rows.append(
            {
                "session_id": sample.session_id,
                "subject_id": sample.subject_id,
                "center_frame": int(sample.center_frame),
                "concept_id": sample.concept_id,
                "source": sample.source,
                "pad_left": int(pad_left),
                "pad_right": int(pad_right),
            }
        )

    if not X_rows:
        empty_x = np.zeros((0, 0, 0), dtype=np.float32)
        empty_y = np.zeros((0,), dtype=np.int32)
        empty_meta = pd.DataFrame(
            columns=["session_id", "subject_id", "center_frame", "concept_id", "source", "pad_left", "pad_right"]
        )
        return DatasetBuildResult(dataset=CenterFrameDataset(X=empty_x, y=empty_y, metadata=empty_meta), skipped_samples=skipped)

    X = np.stack(X_rows, axis=0).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.int32)
    metadata = pd.DataFrame(meta_rows)
    return DatasetBuildResult(dataset=CenterFrameDataset(X=X, y=y, metadata=metadata), skipped_samples=skipped)


def fit_scaler_on_training_split(dataset: CenterFrameDataset) -> StandardScaler:
    """Fit feature normalization on training samples only."""
    scaler = StandardScaler()
    if len(dataset) == 0:
        return scaler
    n, w, f = dataset.X.shape
    scaler.fit(dataset.X.reshape(n * w, f))
    return scaler


def apply_scaler(dataset: CenterFrameDataset, scaler: StandardScaler) -> CenterFrameDataset:
    """Apply an already-fit scaler to a dataset."""
    if len(dataset) == 0:
        return dataset
    n, w, f = dataset.X.shape
    X_scaled = scaler.transform(dataset.X.reshape(n * w, f)).reshape(n, w, f).astype(np.float32)
    return CenterFrameDataset(X=X_scaled, y=dataset.y.copy(), metadata=dataset.metadata.copy())


def save_scaler(scaler: StandardScaler, path: Path) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(scaler, handle)


def load_scaler(path: Path) -> StandardScaler:
    import pickle

    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, StandardScaler):
        raise TypeError("Invalid scaler payload")
    return payload
