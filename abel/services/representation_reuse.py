"""Cross-behavior reuse of the frame/segment representation (Pipeline-All).

During a Pipeline-All run every behavior is trained over the same sessions with
the same window/stride/exclusions, so the frame and segment tables are identical
for all behaviors.  Re-deriving them (and re-reading the multi-GB segment
parquet) per behavior is pure waste.  This module holds the small, pure decision
so it can be unit-tested without importing the Qt UI layer.
"""

from __future__ import annotations

from typing import Callable, Tuple

import pandas as pd

# (signature, frame_df, segment_df)
ReprCache = Tuple[tuple, pd.DataFrame, pd.DataFrame]


def reuse_or_build_representation(
    *,
    active: bool,
    cache: ReprCache | None,
    signature: tuple,
    build_fn: Callable[[], tuple[pd.DataFrame, pd.DataFrame]],
    on_reuse: Callable[[], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, ReprCache | None]:
    """Return (frame_df, segment_df, new_cache).

    If ``active`` and the cached signature matches, return independent copies of
    the cached tables (so downstream mutation cannot corrupt the cache) and leave
    the cache unchanged.  Otherwise call ``build_fn`` and, when ``active``, store
    a fresh copy as the new cache.  When not active, the cache is passed through
    untouched (``None`` stays ``None``).
    """
    if active and cache is not None and cache[0] == signature:
        if on_reuse is not None:
            on_reuse()
        return cache[1].copy(), cache[2].copy(), cache

    frame_df, segment_df = build_fn()
    new_cache: ReprCache | None = cache
    if active:
        new_cache = (signature, frame_df.copy(), segment_df.copy())
    return frame_df, segment_df, new_cache
