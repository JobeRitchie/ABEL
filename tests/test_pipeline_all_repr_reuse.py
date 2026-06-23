"""P2: Pipeline-All must reuse the frame/segment representation across behaviors.

The representation tables are behavior-independent, so building them once and
reusing the in-memory result for the remaining behaviors must (a) avoid calling
the (expensive, multi-GB) representation build more than once and (b) return
identical data each time, while never sharing a cache across single (non
Pipeline-All) runs.  The decision lives in a pure helper so it is testable
without importing the Qt UI layer.
"""

from __future__ import annotations

import pandas as pd

from abel.services.representation_reuse import reuse_or_build_representation


def _make_builder():
    state = {"calls": 0}

    def build():
        state["calls"] += 1
        frame = pd.DataFrame({"session_id": ["s1", "s2"], "f0": [1, 2]})
        seg = pd.DataFrame({"session_id": ["s1", "s2"], "seg": [10, 20]})
        return frame, seg

    return build, state


def test_reuses_across_behaviors_when_active():
    build, state = _make_builder()
    cache = None
    sig = (("s1", "s2"), 30, 15)

    f1, s1, cache = reuse_or_build_representation(
        active=True, cache=cache, signature=sig, build_fn=build
    )
    f2, s2, cache = reuse_or_build_representation(
        active=True, cache=cache, signature=sig, build_fn=build
    )
    f3, s3, cache = reuse_or_build_representation(
        active=True, cache=cache, signature=sig, build_fn=build
    )

    assert state["calls"] == 1, "representation should be built once for the run"
    pd.testing.assert_frame_equal(f1, f2)
    pd.testing.assert_frame_equal(s1, s2)
    pd.testing.assert_frame_equal(f1, f3)


def test_returned_tables_are_independent_copies():
    build, _ = _make_builder()
    sig = (("s1", "s2"), 30, 15)
    f1, _, cache = reuse_or_build_representation(
        active=True, cache=None, signature=sig, build_fn=build
    )
    f1.loc[0, "f0"] = 999  # mutate consumer copy
    f2, _, cache = reuse_or_build_representation(
        active=True, cache=cache, signature=sig, build_fn=build
    )
    assert f2.loc[0, "f0"] != 999, "cache must not be corrupted by consumer mutation"


def test_rebuilds_when_signature_changes():
    build, state = _make_builder()
    _, _, cache = reuse_or_build_representation(
        active=True, cache=None, signature=(("s1", "s2"), 30, 15), build_fn=build
    )
    reuse_or_build_representation(
        active=True, cache=cache, signature=(("s1", "s2", "s3"), 30, 15), build_fn=build
    )
    assert state["calls"] == 2


def test_no_cache_when_not_active():
    build, state = _make_builder()
    sig = (("s1", "s2"), 30, 15)
    _, _, cache = reuse_or_build_representation(
        active=False, cache=None, signature=sig, build_fn=build
    )
    assert cache is None
    reuse_or_build_representation(
        active=False, cache=cache, signature=sig, build_fn=build
    )
    assert state["calls"] == 2, "single runs must not reuse a cross-behavior cache"


def test_on_reuse_callback_only_fires_on_reuse():
    build, _ = _make_builder()
    sig = (("s1", "s2"), 30, 15)
    hits = {"n": 0}

    def on_reuse():
        hits["n"] += 1

    _, _, cache = reuse_or_build_representation(
        active=True, cache=None, signature=sig, build_fn=build, on_reuse=on_reuse
    )
    assert hits["n"] == 0  # first call builds
    reuse_or_build_representation(
        active=True, cache=cache, signature=sig, build_fn=build, on_reuse=on_reuse
    )
    assert hits["n"] == 1  # second call reuses
