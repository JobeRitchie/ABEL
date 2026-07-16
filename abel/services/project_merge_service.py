"""Project Merge Service.

Loads bout data from one or more *external* ABEL project folders and
returns rows/DataFrames that can be appended directly to the host
``BehaviorAnalyticsTab._summary_rows`` / ``_raw_bouts`` collections.

Session IDs are prefixed with ``<tag>::`` to avoid collisions with the
host project. The ``tag`` defaults to the folder name but can be
overridden by the caller.

Behavior *names* from the external project are mapped to behavior IDs by
matching names (case-insensitive) against the host project's behavior
definitions. Behaviors with no match are included with their own ID
(with the tag prefix).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("abel")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json_safe(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_yaml_safe(p: Path) -> dict:
    try:
        import yaml  # type: ignore[import-untyped]
        with p.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

class ProjectMergeEntry:
    """Represents one externally merged project."""

    def __init__(
        self,
        project_root: Path,
        tag: str = "",
        group_override: str = "",
    ) -> None:
        self.project_root = project_root
        self.tag = tag or project_root.name
        self.group_override = group_override  # if set, all sessions get this group


class ProjectMergeService:
    """Loads bout data from external projects for combined analysis."""

    def __init__(self) -> None:
        self._entries: list[ProjectMergeEntry] = []

    # ------------------------------------------------------------------
    # Entry management
    # ------------------------------------------------------------------

    def add_project(
        self,
        project_root: Path,
        tag: str = "",
        group_override: str = "",
    ) -> None:
        for e in self._entries:
            if e.project_root.resolve() == project_root.resolve():
                return  # already added
        self._entries.append(
            ProjectMergeEntry(project_root, tag, group_override)
        )

    def remove_project(self, project_root: Path) -> None:
        self._entries = [
            e for e in self._entries
            if e.project_root.resolve() != project_root.resolve()
        ]

    def clear(self) -> None:
        self._entries.clear()

    @property
    def entries(self) -> list[ProjectMergeEntry]:
        return list(self._entries)

    def is_empty(self) -> bool:
        return not self._entries

    # ------------------------------------------------------------------
    # Behavior name → id mapping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_external_behaviors(project_root: Path) -> dict[str, str]:
        """Return {behavior_id: name} for behaviors defined in the external project."""
        cfg_path = project_root / "config" / "behavior_definitions.yaml"
        if not cfg_path.exists():
            return {}
        data = _read_yaml_safe(cfg_path)
        behaviors = data.get("behaviors", []) or []
        out: dict[str, str] = {}
        for b in behaviors:
            bid  = str(b.get("behavior_id") or b.get("id") or "").strip()
            name = str(b.get("name") or "").strip()
            if bid:
                out[bid] = name or bid
        return out

    @staticmethod
    def _build_name_to_host_bid(host_bid_name_map: dict[str, str]) -> dict[str, str]:
        """Return {lower_name: host_bid} for fuzzy name matching."""
        return {v.lower(): k for k, v in host_bid_name_map.items()}

    # ------------------------------------------------------------------
    # Primary load method
    # ------------------------------------------------------------------

    def load_merged_bouts(
        self,
        host_bid_name_map: dict[str, str],
        host_fps: float,
        use_cached: bool = False,
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, list[dict[str, Any]]],
        dict[str, str],
        dict[str, str],
        dict[str, dict[str, str]],
        dict[str, int],
        dict[str, float],
    ]:
        """Load bouts from all registered external projects.

        Parameters
        ----------
        use_cached:
            When True, each external project's ``derived/analytics_cache/``
            is tried first.  If a valid cache is found the expensive
            re-computation from inference traces is skipped for that project.
            Falls back to full recompute when no cache is available.

        Returns
        -------
        summary_rows : list[dict]
            Rows compatible with ``BehaviorAnalyticsTab._summary_rows``.
        raw_bout_rows : dict[str, list[dict]]
            {bid: [row, ...]} — to be turned into DataFrames and merged
            into ``_raw_bouts``.
        session_label_map : dict[str, str]
            {namespaced_session_id: display_label}
        tag_group_map : dict[str, str]
            {namespaced_session_label: group_name}
            Populated when ``group_override`` is set on the entry.
        ext_factor_map : dict[str, dict[str, str]]
            {host_session_label: {factor_name: level}}
            Factor assignments read from each external project's
            analytics_groups.json, remapped to namespaced host labels.
        ext_session_prechop_map : dict[str, int]
            {namespaced_session_id: prechop_frames} imported from each
            external project's analytics_groups.json subject_prechop_frames.
        ext_session_end_s_map : dict[str, float]
            {namespaced_session_id: session_end_seconds} derived from external
            import manifests (duration_sec or frame_count/fps).
        """
        summary_rows: list[dict[str, Any]] = []
        raw_bout_rows: dict[str, list[dict[str, Any]]] = {}
        session_label_map: dict[str, str] = {}
        tag_group_map: dict[str, str] = {}
        ext_factor_map: dict[str, dict[str, str]] = {}
        ext_session_prechop_map: dict[str, int] = {}
        ext_session_end_s_map: dict[str, float] = {}

        name_to_host = self._build_name_to_host_bid(host_bid_name_map)

        for entry in self._entries:
            try:
                s, r, slm, tgm, efm, epm, ees = self._load_one(
                    entry, host_bid_name_map, name_to_host, host_fps,
                    use_cached=use_cached,
                )
                summary_rows.extend(s)
                for bid, rows in r.items():
                    raw_bout_rows.setdefault(bid, []).extend(rows)
                session_label_map.update(slm)
                tag_group_map.update(tgm)
                ext_factor_map.update(efm)
                ext_session_prechop_map.update(epm)
                ext_session_end_s_map.update(ees)
            except Exception as exc:
                logger.warning(
                    "ProjectMergeService: failed to load %s: %s",
                    entry.project_root, exc, exc_info=True,
                )
        return (
            summary_rows,
            raw_bout_rows,
            session_label_map,
            tag_group_map,
            ext_factor_map,
            ext_session_prechop_map,
            ext_session_end_s_map,
        )

    # ------------------------------------------------------------------
    # Per-project loader
    # ------------------------------------------------------------------

    def _load_one(
        self,
        entry: ProjectMergeEntry,
        host_bid_name_map: dict[str, str],
        name_to_host: dict[str, str],
        host_fps: float,
        use_cached: bool = False,
    ) -> tuple[
        list[dict],
        dict[str, list[dict]],
        dict[str, str],
        dict[str, str],
        dict[str, dict[str, str]],
        dict[str, int],
        dict[str, float],
    ]:
        root = entry.project_root
        tag  = entry.tag

        # Resolve behaviors in the external project
        ext_behaviors = self._read_external_behaviors(root)

        # Map external bid → host bid (fall back to "tag::bid" if no match)
        def resolve_bid(ext_bid: str) -> tuple[str, str]:
            ext_name = ext_behaviors.get(ext_bid, ext_bid)
            host_bid = name_to_host.get(ext_name.lower())
            if host_bid:
                host_name = host_bid_name_map[host_bid]
                return host_bid, host_name
            # No match — use the external behavior as-is with tag namespace
            namespaced = f"{tag}::{ext_bid}"
            return namespaced, ext_name or ext_bid

        # Namespace a session_id
        def ns(sid: str) -> str:
            return f"{tag}::{sid}"

        # Infer FPS from external project (fall back to host fps)
        ext_fps = self._read_project_fps(root) or host_fps

        # Build session→subject map from external manifest
        session_label_local = self._build_session_labels(root, tag)

        summary_rows: list[dict] = []
        raw_bout_rows: dict[str, list[dict]] = {}
        session_label_map: dict[str, str] = {}
        tag_group_map: dict[str, str] = {}
        loaded_keys: set[tuple[str, str]] = set()

        for ext_sid, display_label in session_label_local.items():
            namespaced_sid = ns(ext_sid)
            session_label_map[namespaced_sid] = display_label
            if entry.group_override:
                tag_group_map[display_label] = entry.group_override

        # External session end-times (seconds), namespaced for host use.
        ext_session_end_s_map: dict[str, float] = {}
        sid_to_end_s = self._read_external_session_end_seconds(root, fallback_fps=ext_fps)
        for ext_sid, end_s in sid_to_end_s.items():
            namespaced_sid = ns(ext_sid)
            if namespaced_sid in session_label_map and end_s > 0:
                ext_session_end_s_map[namespaced_sid] = float(end_s)

        # ── Try analytics cache if requested ─────────────────────────────
        if use_cached:
            cached_result = self._load_from_analytics_cache(
                root, tag, ext_behaviors, resolve_bid, ns,
                ext_fps, session_label_map, tag_group_map,
            )
            if cached_result is not None:
                logger.debug(
                    "ProjectMergeService: loaded '%s' from analytics cache.", tag
                )
                return cached_result

        # Read the external project's per-behavior temporal thresholds once.
        # These are applied in both Source 1 and Source 1b below so that every
        # merged project's data is processed with *its own* saved settings.
        ext_thresholds = self._read_tr_thresholds(root)

        # ── Source 1: temporal_refinement — re-apply per-project thresholds ──
        # For each behavior we prefer to re-compute bouts from the raw inference
        # traces using the external project's temporal_review_settings.json.
        # Only when no inference traces are available do we fall back to the
        # pre-stored postprocess bouts (which may have been computed with
        # different settings and therefore should be used only as a last resort).
        tr_root = root / "derived" / "temporal_refinement"
        active_tb_inference = self._read_active_target_behavior_inference_dir(root)
        if tr_root.exists():
            try:
                import numpy as np
                from abel.temporal_refinement.bout_postprocess import (
                    smooth_probabilities,
                    threshold_probabilities,
                    merge_close_bouts,
                    remove_short_bouts,
                    binary_trace_to_intervals,
                )
                _bout_postprocess_available = True
            except Exception:
                _bout_postprocess_available = False

            for ext_bid, ext_name in ext_behaviors.items():
                host_bid, host_name = resolve_bid(ext_bid)
                token = self._safe_name(ext_bid)
                latest_path = tr_root / token / "latest.json"
                if not latest_path.exists():
                    continue
                latest = _read_json_safe(latest_path)

                # Skip stale postprocess artifacts when they were generated from
                # a missing or non-active inference run.
                inference_dir_raw = str(latest.get("inference_dir", "") or "").strip()
                if inference_dir_raw and not Path(inference_dir_raw).exists():
                    continue
                if (
                    active_tb_inference
                    and inference_dir_raw
                    and inference_dir_raw != active_tb_inference
                ):
                    continue

                # Resolve per-behavior threshold params
                defaults_p = ext_thresholds.get("__defaults__", {
                    "onset_threshold": 0.5,
                    "min_bout_duration_frames": 6,
                    "merge_gap_frames": 3,
                })
                params = ext_thresholds.get(ext_bid, defaults_p)
                onset     = float(params.get("onset_threshold",          defaults_p.get("onset_threshold", 0.5)))
                min_bout  = int(params.get("min_bout_duration_frames",   defaults_p.get("min_bout_duration_frames", 6)))
                merge_gap = int(params.get("merge_gap_frames",           defaults_p.get("merge_gap_frames", 3)))

                # Resolve smoothing settings from the postprocess manifest
                sm_method = "moving_average"
                sm_window = 5
                post_dir_raw = str(latest.get("postprocess_dir", "") or "").strip()
                if post_dir_raw:
                    mp = Path(post_dir_raw) / "postprocess_manifest.json"
                    if mp.exists():
                        try:
                            pm_data = _read_json_safe(mp)
                            sm_method = str(pm_data.get("smoothing_method", sm_method))
                            sm_window = int(pm_data.get("smoothing_window", sm_window))
                        except Exception:
                            pass

                # ── Preferred path: re-compute from inference traces ──────────
                inf_trace_paths: dict[str, str] = {}
                inf_dir_raw = str(latest.get("inference_dir", "") or "").strip()
                if inf_dir_raw:
                    ip = Path(inf_dir_raw) / "inference_manifest.json"
                    if ip.exists():
                        try:
                            im = _read_json_safe(ip)
                            inf_trace_paths = {
                                str(k): str(v)
                                for k, v in (im.get("trace_paths", {}) or {}).items()
                            }
                        except Exception:
                            pass

                if inf_trace_paths and _bout_postprocess_available:
                    for ext_sid, tp_str in inf_trace_paths.items():
                        if not tp_str or not Path(tp_str).exists():
                            continue
                        try:
                            trace_df = pd.read_parquet(tp_str)
                        except Exception:
                            continue
                        if trace_df.empty:
                            continue
                        # Per-behavior traces have a "probability" column;
                        # multi-behavior traces may use "prob_{ext_bid}".
                        prob_col = f"prob_{ext_bid}"
                        if prob_col not in trace_df.columns:
                            prob_col = "probability" if "probability" in trace_df.columns else None
                        if prob_col is None:
                            continue
                        frame_arr = (
                            trace_df["frame"].to_numpy(dtype=int)
                            if "frame" in trace_df.columns
                            else np.arange(len(trace_df), dtype=int)
                        )
                        raw = (
                            pd.to_numeric(trace_df[prob_col], errors="coerce")
                            .fillna(0.0)
                            .to_numpy(dtype=float)
                        )
                        smoothed = smooth_probabilities(raw, method=sm_method, window=sm_window)
                        binary   = threshold_probabilities(smoothed, onset_thresh=onset)
                        binary   = merge_close_bouts(binary, max_gap_frames=merge_gap)
                        binary   = remove_short_bouts(binary, min_duration_frames=min_bout)
                        intervals = binary_trace_to_intervals(binary)

                        namespaced_sid = ns(ext_sid)
                        if namespaced_sid not in session_label_map:
                            continue
                        display_label = session_label_map[namespaced_sid]

                        bout_rows_here: list[dict] = []
                        for s_idx, e_idx in intervals:
                            sf = int(frame_arr[s_idx]) if s_idx < len(frame_arr) else s_idx
                            ef = int(frame_arr[min(e_idx, len(frame_arr) - 1)])
                            bout_rows_here.append({"start_frame": sf, "end_frame": ef})
                            raw_bout_rows.setdefault(host_bid, []).append({
                                "session_id":  namespaced_sid,
                                "start_frame": sf,
                                "end_frame":   ef,
                                "behavior_id": host_bid,
                                "behavior":    host_name,
                            })

                        n_bouts = len(bout_rows_here)
                        if n_bouts == 0:
                            continue
                        total_frames = float(sum(r["end_frame"] - r["start_frame"] + 1 for r in bout_rows_here))
                        time_s   = total_frames / ext_fps
                        mean_dur = time_s / n_bouts
                        latency_s = float(bout_rows_here[0]["start_frame"]) / ext_fps
                        summary_rows.append(self._make_summary_row(
                            namespaced_sid, display_label, host_bid, host_name,
                            n_bouts, time_s, mean_dur, latency_s,
                        ))
                        loaded_keys.add((host_bid, namespaced_sid))

                else:
                    # ── Fallback: stored postprocess bouts ───────────────────
                    # Used only when inference traces are not available.
                    # Note: these bouts were computed with whatever settings were
                    # active at the time of the last postprocess run and may not
                    # reflect the current temporal_review_settings.json.
                    if not post_dir_raw:
                        continue
                    manifest_path = Path(post_dir_raw) / "postprocess_manifest.json"
                    if not manifest_path.exists():
                        continue
                    pm = _read_json_safe(manifest_path)
                    bout_paths = {str(k): str(v) for k, v in (pm.get("bout_paths", {}) or {}).items()}
                    for ext_sid, bp_str in bout_paths.items():
                        bp_path = Path(bp_str)
                        if not bp_path.exists():
                            continue
                        try:
                            bout_df = pd.read_parquet(bp_path)
                        except Exception:
                            continue
                        if bout_df.empty or not {"start_frame", "end_frame"}.issubset(bout_df.columns):
                            continue
                        namespaced_sid = ns(ext_sid)
                        display_label  = session_label_map.get(namespaced_sid, namespaced_sid)
                        for _, bout in bout_df.iterrows():
                            raw_bout_rows.setdefault(host_bid, []).append({
                                "session_id":  namespaced_sid,
                                "start_frame": int(bout["start_frame"]),
                                "end_frame":   int(bout["end_frame"]),
                                "behavior_id": host_bid,
                                "behavior":    host_name,
                            })
                        n_bouts      = len(bout_df)
                        total_frames = float((bout_df["end_frame"] - bout_df["start_frame"] + 1).sum())
                        time_s       = total_frames / ext_fps
                        mean_dur     = time_s / n_bouts if n_bouts else 0.0
                        latency_s    = float(bout_df["start_frame"].min()) / ext_fps if n_bouts else float("nan")
                        summary_rows.append(self._make_summary_row(
                            namespaced_sid, display_label, host_bid, host_name,
                            n_bouts, time_s, mean_dur, latency_s,
                        ))
                        loaded_keys.add((host_bid, namespaced_sid))

        # ── Source 1b: target_behavior TR probability traces ──────────
        # Mirrors what BehaviorAnalyticsTab.Source-2 does for the host project.
        # Reads the external project's target_behavior inference traces and
        # re-applies the external project's own per-behavior thresholds.
        # ext_thresholds is already loaded above and reused here.
        tb_trace_paths, tb_sm_method, tb_sm_window = self._read_target_behavior_traces(root)
        if tb_trace_paths:
            tb_bouts = self._recompute_bouts_from_traces(
                tb_trace_paths,
                ext_behaviors,
                resolve_bid,
                ext_thresholds,
                ext_fps,
                tb_sm_method,
                tb_sm_window,
            )
            for (host_bid, host_name, ext_sid), bout_rows in tb_bouts.items():
                namespaced_sid = ns(ext_sid)
                if (host_bid, namespaced_sid) in loaded_keys:
                    continue  # already loaded from postprocess artifacts
                if namespaced_sid not in session_label_map:
                    continue  # session not in this project's manifest
                display_label = session_label_map[namespaced_sid]
                for row in bout_rows:
                    raw_bout_rows.setdefault(host_bid, []).append({
                        "session_id":  namespaced_sid,
                        "start_frame": row["start_frame"],
                        "end_frame":   row["end_frame"],
                        "behavior_id": host_bid,
                        "behavior":    host_name,
                    })
                if not bout_rows:
                    continue
                sdf = pd.DataFrame(bout_rows)
                n_bouts = len(sdf)
                total_frames = float((sdf["end_frame"] - sdf["start_frame"] + 1).sum())
                time_s = total_frames / ext_fps
                mean_dur = time_s / n_bouts if n_bouts else 0.0
                latency_s = float(sdf["start_frame"].min()) / ext_fps if n_bouts else float("nan")
                summary_rows.append(self._make_summary_row(
                    namespaced_sid, display_label, host_bid, host_name,
                    n_bouts, time_s, mean_dur, latency_s,
                ))
                loaded_keys.add((host_bid, namespaced_sid))

        # ── Source 2: behavior_bouts parquets (fallback) ─────────────
        bouts_dir = root / "derived" / "behavior_bouts"
        if bouts_dir.exists():
            for ext_bid, ext_name in ext_behaviors.items():
                host_bid, host_name = resolve_bid(ext_bid)
                bout_path = bouts_dir / f"{ext_bid}_bouts.parquet"
                if not bout_path.exists():
                    continue
                try:
                    df = pd.read_parquet(bout_path)
                except Exception:
                    continue
                if df.empty:
                    continue
                for ext_sid, grp in df.groupby("session_id"):
                    namespaced_sid = ns(str(ext_sid))
                    if (host_bid, namespaced_sid) in loaded_keys:
                        continue
                    if namespaced_sid not in session_label_map:
                        session_label_map[namespaced_sid] = namespaced_sid
                    display_label = session_label_map[namespaced_sid]
                    for _, bout in grp.iterrows():
                        raw_bout_rows.setdefault(host_bid, []).append({
                            "session_id":  namespaced_sid,
                            "start_frame": int(bout["start_frame"]),
                            "end_frame":   int(bout["end_frame"]),
                            "behavior_id": host_bid,
                            "behavior":    host_name,
                        })
                    n_bouts = len(grp)
                    if "duration_frames" in grp.columns:
                        total_frames = float(grp["duration_frames"].sum())
                    else:
                        total_frames = float((grp["end_frame"] - grp["start_frame"] + 1).sum())
                    time_s   = total_frames / ext_fps
                    mean_dur = time_s / n_bouts if n_bouts else 0.0
                    latency_s = float(grp["start_frame"].min()) / ext_fps if n_bouts else float("nan")
                    summary_rows.append(self._make_summary_row(
                        namespaced_sid, display_label, host_bid, host_name,
                        n_bouts, time_s, mean_dur, latency_s,
                    ))
                    loaded_keys.add((host_bid, namespaced_sid))

        # ── Factor assignments from external analytics_groups.json ──────────
        ext_factor_map: dict[str, dict[str, str]] = {}
        analytics_path = root / "derived" / "analytics_groups.json"
        if analytics_path.exists():
            analytics_data = _read_json_safe(analytics_path)
            ext_session_factors = analytics_data.get("session_factors") or {}
            valid_host_labels = set(session_label_local.values())
            for ext_label, facs in ext_session_factors.items():
                if not isinstance(facs, dict):
                    continue
                host_label = f"{tag}/{ext_label}"
                if host_label in valid_host_labels:
                    ext_factor_map[host_label] = {
                        k: v for k, v in facs.items()
                        if isinstance(k, str) and isinstance(v, str) and v
                    }

        # ── Per-session prechop from external subject_prechop_frames ───────
        ext_session_prechop_map: dict[str, int] = {}
        sid_to_prechop = self._read_external_subject_prechop_by_session(root)
        for ext_sid, frames in sid_to_prechop.items():
            namespaced_sid = ns(ext_sid)
            if namespaced_sid not in session_label_map:
                continue
            ext_session_prechop_map[namespaced_sid] = int(frames)

        return (
            summary_rows,
            raw_bout_rows,
            session_label_map,
            tag_group_map,
            ext_factor_map,
            ext_session_prechop_map,
            ext_session_end_s_map,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_name(s: str) -> str:
        import re
        return re.sub(r"[^\w\-]", "_", s)

    @staticmethod
    def _load_from_analytics_cache(
        root: Path,
        tag: str,
        ext_behaviors: dict[str, str],
        resolve_bid: Any,
        ns: Any,
        ext_fps: float,
        session_label_map: dict[str, str],
        tag_group_map: dict[str, str],
    ) -> tuple[
        list[dict],
        dict[str, list[dict]],
        dict[str, str],
        dict[str, str],
        dict[str, dict[str, str]],
        dict[str, int],
        dict[str, float],
    ] | None:
        """Try to load merged-project data from the project's analytics cache.

        Returns the same 5-tuple as ``_load_one`` on success, or ``None``
        when the cache is absent / invalid so the caller can fall back to
        full recomputation.
        """
        import re as _re
        cache_dir = root / "derived" / "analytics_cache"
        meta_path = cache_dir / "analytics_cache.json"
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if meta.get("version") != 2:
            return None
        cache_summary: list[dict] = meta.get("summary_rows", [])
        if not cache_summary:
            return None

        summary_rows: list[dict] = []
        raw_bout_rows: dict[str, list[dict]] = {}

        for row in cache_summary:
            ext_sid = str(row.get("session_id", "")).strip()
            ext_bid = str(row.get("behavior_id", "")).strip()
            if not ext_sid or not ext_bid:
                continue
            host_bid, host_name = resolve_bid(ext_bid)
            namespaced_sid = ns(ext_sid)
            if namespaced_sid not in session_label_map:
                continue
            display_label = session_label_map[namespaced_sid]
            new_row = dict(row)
            new_row["session_id"]    = namespaced_sid
            new_row["subject"]       = display_label
            new_row["session_label"] = display_label
            new_row["behavior_id"]   = host_bid
            new_row["behavior"]      = host_name
            summary_rows.append(new_row)

        # Load raw bout parquets per behavior (for motif / density analysis).
        for ext_bid in ext_behaviors:
            host_bid, host_name = resolve_bid(ext_bid)
            safe_bid = _re.sub(r"[^\w\-]", "_", ext_bid)
            p = cache_dir / f"bouts_{safe_bid}.parquet"
            if not p.exists():
                continue
            try:
                df = pd.read_parquet(p)
            except Exception:
                continue
            if df.empty or not {"start_frame", "end_frame"}.issubset(df.columns):
                continue
            for _, bout_row in df.iterrows():
                ext_sid = str(bout_row.get("session_id", "")).strip()
                if not ext_sid:
                    continue
                namespaced_sid = ns(ext_sid)
                if namespaced_sid not in session_label_map:
                    continue
                raw_bout_rows.setdefault(host_bid, []).append({
                    "session_id":  namespaced_sid,
                    "start_frame": int(bout_row["start_frame"]),
                    "end_frame":   int(bout_row["end_frame"]),
                    "behavior_id": host_bid,
                    "behavior":    host_name,
                })

        if not summary_rows:
            return None

        # Factor assignments (same as _load_one Source 4 block)
        ext_factor_map: dict[str, dict[str, str]] = {}
        analytics_path = root / "derived" / "analytics_groups.json"
        if analytics_path.exists():
            try:
                analytics_data = json.loads(analytics_path.read_text(encoding="utf-8"))
                for ext_label, facs in (analytics_data.get("session_factors") or {}).items():
                    if not isinstance(facs, dict):
                        continue
                    host_label = f"{tag}/{ext_label}"
                    if host_label in set(session_label_map.values()):
                        ext_factor_map[host_label] = {
                            k: v for k, v in facs.items()
                            if isinstance(k, str) and isinstance(v, str) and v
                        }
            except Exception:
                pass

        # Per-session prechop from external subject_prechop_frames
        ext_session_prechop_map: dict[str, int] = {}
        sid_to_prechop = ProjectMergeService._read_external_subject_prechop_by_session(root)
        for ext_sid, frames in sid_to_prechop.items():
            namespaced_sid = ns(ext_sid)
            if namespaced_sid not in session_label_map:
                continue
            ext_session_prechop_map[namespaced_sid] = int(frames)

        # Per-session end-time (seconds) from external import manifest
        ext_session_end_s_map: dict[str, float] = {}
        sid_to_end_s = ProjectMergeService._read_external_session_end_seconds(root, fallback_fps=ext_fps)
        for ext_sid, end_s in sid_to_end_s.items():
            namespaced_sid = ns(ext_sid)
            if namespaced_sid in session_label_map and end_s > 0:
                ext_session_end_s_map[namespaced_sid] = float(end_s)

        return (
            summary_rows,
            raw_bout_rows,
            session_label_map,
            tag_group_map,
            ext_factor_map,
            ext_session_prechop_map,
            ext_session_end_s_map,
        )

    @staticmethod
    def _make_summary_row(
        session_id: str,
        session_label: str,
        bid: str,
        bname: str,
        n_bouts: int,
        time_s: float,
        mean_dur: float,
        latency_s: float,
    ) -> dict[str, Any]:
        return {
            "session_id":    session_id,
            "subject":       session_label,
            "session_label": session_label,
            "session_type":  "",
            "behavior_id":   bid,
            "behavior":      bname,
            "n_bouts":       float(n_bouts),
            "time_spent_s":  time_s,
            "mean_bout_s":   mean_dur,
            "latency_s":     latency_s,
            "distance_cm":   0.0,
        }

    @staticmethod
    def _read_project_fps(root: Path) -> float:
        """Try to read fps from the project config, return 0 on failure."""
        for rel in (
            "config/preprocessing.yaml",
            "config/app_settings.yaml",
        ):
            p = root / rel
            if p.exists():
                d = _read_yaml_safe(p)
                fps = d.get("fps") or d.get("video_fps") or d.get("frame_rate")
                if fps:
                    return float(fps)
        return 0.0

    @staticmethod
    def _read_active_target_behavior_inference_dir(root: Path) -> str:
        """Return active target_behavior inference_dir, or ''."""
        p = root / "derived" / "temporal_refinement" / "target_behavior" / "latest.json"
        if not p.exists():
            return ""
        data = _read_json_safe(p)
        return str(data.get("inference_dir", "") or "").strip()

    @staticmethod
    def _build_session_labels(root: Path, tag: str) -> dict[str, str]:
        """Return {ext_session_id: display_label} for sessions in the external project."""
        # Try to read the import manifest to get subject names
        manifest_dir = root / "derived" / "manifests"
        if not manifest_dir.exists():
            manifest_dir = root / "derived"

        # Look for the newest import_manifest*.json
        candidates = sorted(
            list((root / "derived").rglob("import_manifest*.json")),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        if not candidates:
            return {}

        data = _read_json_safe(candidates[0])
        sessions = data.get("linked_sessions") or data.get("sessions") or []
        out: dict[str, str] = {}
        for s in sessions:
            sid     = str(s.get("session_id") or "").strip()
            subject = str(s.get("subject_id") or s.get("subject") or "").strip() or sid
            if not sid:
                continue
            out[sid] = f"{tag}/{subject}"
        return out

    @staticmethod
    def _read_external_subject_prechop_by_session(root: Path) -> dict[str, int]:
        """Return {ext_session_id: prechop_frames} from external analytics state."""
        analytics_path = root / "derived" / "analytics_groups.json"
        if not analytics_path.exists():
            return {}
        analytics_data = _read_json_safe(analytics_path)
        raw_by_subject = analytics_data.get("subject_prechop_frames") or {}
        if not isinstance(raw_by_subject, dict):
            return {}

        # Build ext_session_id -> ext_subject from import manifest.
        candidates = sorted(
            list((root / "derived").rglob("import_manifest*.json")),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        if not candidates:
            return {}
        data = _read_json_safe(candidates[0])
        sessions = data.get("linked_sessions") or data.get("sessions") or []

        out: dict[str, int] = {}
        for s in sessions:
            sid = str(s.get("session_id") or "").strip()
            subject = str(s.get("subject_id") or s.get("subject") or "").strip()
            if not sid or not subject:
                continue
            try:
                frames = max(0, int(raw_by_subject.get(subject, 0)))
            except Exception:
                continue
            if frames > 0:
                out[sid] = frames
        return out

    @staticmethod
    def _read_external_session_end_seconds(root: Path, fallback_fps: float) -> dict[str, float]:
        """Return {ext_session_id: end_seconds} from external import manifest."""
        candidates = sorted(
            list((root / "derived").rglob("import_manifest*.json")),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        if not candidates:
            return {}

        data = _read_json_safe(candidates[0])
        videos = data.get("videos") or []
        linked_sessions = data.get("linked_sessions") or data.get("sessions") or []

        video_by_asset: dict[str, dict[str, Any]] = {}
        for v in videos:
            aid = str(v.get("asset_id") or "").strip()
            if aid:
                video_by_asset[aid] = v

        out: dict[str, float] = {}
        for s in linked_sessions:
            sid = str(s.get("session_id") or "").strip()
            if not sid:
                continue
            vid = video_by_asset.get(str(s.get("video_asset_id") or "").strip(), {})
            dur = vid.get("duration_sec")
            if dur is not None:
                try:
                    d = float(dur)
                    if d > 0:
                        out[sid] = d
                        continue
                except Exception:
                    pass

            frame_count = vid.get("frame_count")
            fps = vid.get("fps") or fallback_fps
            try:
                n = int(frame_count)
                f = float(fps)
                if n > 0 and f > 0:
                    out[sid] = float(n) / float(f)
            except Exception:
                continue

        return out

    @staticmethod
    def _read_tr_thresholds(root: Path) -> dict[str, dict[str, Any]]:
        """Read per-behavior bout-detection thresholds from the external project.

        Returns a dict with optional key ``"__defaults__"`` and per-behavior
        entries keyed by the external project's behavior_id.  Each value is a
        dict with ``onset_threshold``, ``min_bout_duration_frames``, and
        ``merge_gap_frames``.
        """
        settings_path = root / "config" / "temporal_review_settings.json"
        defaults: dict[str, Any] = {
            "onset_threshold": 0.5,
            "min_bout_duration_frames": 6,
            "merge_gap_frames": 3,
        }
        if not settings_path.exists():
            return {"__defaults__": defaults}
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception:
            return {"__defaults__": defaults}
        raw_defaults = data.get("__all__", {})
        defaults = {
            "onset_threshold": float(raw_defaults.get("onset_threshold", 0.5)),
            "min_bout_duration_frames": int(raw_defaults.get("min_bout_duration_frames", 6)),
            "merge_gap_frames": int(raw_defaults.get("merge_gap_frames", 3)),
        }
        result: dict[str, dict[str, Any]] = {"__defaults__": defaults}
        for bid, vals in (data.get("by_behavior") or {}).items():
            if bid == "target_behavior":
                continue
            result[bid] = {
                "onset_threshold": float(vals.get("onset_threshold", defaults["onset_threshold"])),
                "min_bout_duration_frames": int(vals.get("min_bout_duration_frames", defaults["min_bout_duration_frames"])),
                "merge_gap_frames": int(vals.get("merge_gap_frames", defaults["merge_gap_frames"])),
            }
        return result

    @staticmethod
    def _read_target_behavior_traces(root: Path) -> tuple[dict[str, str], str, int]:
        """Return (trace_paths, smoothing_method, smoothing_window) for the external project.

        ``trace_paths`` maps session_id → absolute path to the per-session
        probability trace parquet.  Returns empty dict when not available.
        """
        trace_paths: dict[str, str] = {}
        smoothing_method = "moving_average"
        smoothing_window = 5
        tb_latest = (
            root / "derived" / "temporal_refinement" / "target_behavior" / "latest.json"
        )
        if not tb_latest.exists():
            return trace_paths, smoothing_method, smoothing_window
        try:
            latest = json.loads(tb_latest.read_text(encoding="utf-8"))
        except Exception:
            return trace_paths, smoothing_method, smoothing_window

        inf_dir_raw = str(latest.get("inference_dir", "") or "").strip()
        if inf_dir_raw:
            ip = Path(inf_dir_raw) / "inference_manifest.json"
            if ip.exists():
                try:
                    im = json.loads(ip.read_text(encoding="utf-8"))
                    trace_paths = {
                        str(k): str(v)
                        for k, v in (im.get("trace_paths", {}) or {}).items()
                    }
                except Exception:
                    pass

        post_dir_raw = str(latest.get("postprocess_dir", "") or "").strip()
        if post_dir_raw:
            mp = Path(post_dir_raw) / "postprocess_manifest.json"
            if mp.exists():
                try:
                    pm = json.loads(mp.read_text(encoding="utf-8"))
                    smoothing_method = str(pm.get("smoothing_method", smoothing_method))
                    smoothing_window = int(pm.get("smoothing_window", smoothing_window))
                except Exception:
                    pass

        return trace_paths, smoothing_method, smoothing_window

    @staticmethod
    def _recompute_bouts_from_traces(
        trace_paths: dict[str, str],
        ext_behaviors: dict[str, str],
        resolve_bid: Any,
        thresholds: dict[str, dict[str, Any]],
        fps: float,
        smoothing_method: str,
        smoothing_window: int,
    ) -> dict[tuple[str, str, str], list[dict]]:
        """Recompute bouts from target_behavior probability traces.

        Returns a dict keyed by ``(host_bid, host_name, namespaced_sid)``
        with a list of raw bout row dicts.
        """
        try:
            import numpy as np
            from abel.temporal_refinement.bout_postprocess import (
                smooth_probabilities,
                threshold_probabilities,
                merge_close_bouts,
                remove_short_bouts,
                binary_trace_to_intervals,
            )
        except Exception:
            return {}

        defaults = thresholds.get("__defaults__", {
            "onset_threshold": 0.5,
            "min_bout_duration_frames": 6,
            "merge_gap_frames": 3,
        })

        result: dict[tuple[str, str, str], list[dict]] = {}

        for ext_sid, tp_str in trace_paths.items():
            if not tp_str or not Path(tp_str).exists():
                continue
            try:
                trace_df = pd.read_parquet(tp_str)
            except Exception:
                continue
            if trace_df.empty:
                continue

            frame_arr = (
                trace_df["frame"].to_numpy(dtype=int)
                if "frame" in trace_df.columns
                else np.arange(len(trace_df), dtype=int)
            )

            for ext_bid, _ext_name in ext_behaviors.items():
                prob_col = f"prob_{ext_bid}"
                if prob_col not in trace_df.columns:
                    continue

                host_bid, host_name = resolve_bid(ext_bid)

                raw = (
                    pd.to_numeric(trace_df[prob_col], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                smoothed = smooth_probabilities(
                    raw, method=smoothing_method, window=smoothing_window,
                )

                params = thresholds.get(ext_bid, defaults)
                onset = float(params.get("onset_threshold", defaults["onset_threshold"]))
                min_bout = int(params.get("min_bout_duration_frames", defaults["min_bout_duration_frames"]))
                merge_gap = int(params.get("merge_gap_frames", defaults["merge_gap_frames"]))

                binary = threshold_probabilities(smoothed, onset_thresh=onset)
                binary = merge_close_bouts(binary, max_gap_frames=merge_gap)
                binary = remove_short_bouts(binary, min_duration_frames=min_bout)
                intervals = binary_trace_to_intervals(binary)

                # Use tag-namespaced session id — caller must have built session_label_map
                # before this call, but we only have ext_sid here; let the caller namespace
                key = (host_bid, host_name, ext_sid)  # caller will namespace ext_sid
                rows = result.setdefault(key, [])
                for s_idx, e_idx in intervals:
                    sf = int(frame_arr[s_idx]) if s_idx < len(frame_arr) else s_idx
                    ef = int(frame_arr[min(e_idx, len(frame_arr) - 1)])
                    rows.append({
                        "start_frame": sf,
                        "end_frame":   ef,
                        "behavior_id": host_bid,
                        "behavior":    host_name,
                    })

        return result

    # ------------------------------------------------------------------
    # Serialisation (persist merged project list with the host project)
    # ------------------------------------------------------------------

    def save(self, project_root: Path) -> None:
        p = project_root / "config" / "merged_projects.json"
        try:
            data = [
                {
                    "project_root":   str(e.project_root),
                    "tag":            e.tag,
                    "group_override": e.group_override,
                }
                for e in self._entries
            ]
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to save merged project list: %s", exc)

    def load(self, project_root: Path) -> None:
        p = project_root / "config" / "merged_projects.json"
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for entry in data:
                ep = Path(entry["project_root"])
                if ep.exists():
                    self.add_project(
                        ep,
                        tag=entry.get("tag", ""),
                        group_override=entry.get("group_override", ""),
                    )
        except Exception as exc:
            logger.warning("Failed to load merged project list: %s", exc)
