"""Subprocess entrypoint for temporal refinement jobs with streamed logging."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from abel.core.project_manager import ProjectManager
from abel.temporal_refinement.temporal_refinement_service import TemporalRefinementConfig


def _print_line(message: str) -> None:
    print(str(message), flush=True)


def _run_train(args: argparse.Namespace) -> int:
    """Legacy train entry point — returns immediately (training removed)."""
    result = {"status": "skipped", "reason": "training_removed"}
    print("RESULT_JSON:" + json.dumps(result), flush=True)
    return 0


def _run_infer(args: argparse.Namespace) -> int:
    project_root = Path(str(args.project_root)).resolve()
    concept_id = str(args.concept_id)
    cfg_raw = json.loads(str(args.config_json or "{}"))
    cfg = TemporalRefinementConfig(**cfg_raw)

    manager = ProjectManager(project_root)
    result = manager.run_temporal_refinement_inference(
        concept_id=concept_id,
        config=cfg,
        progress_cb=_print_line,
    )
    print("RESULT_JSON:" + json.dumps(result), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run temporal refinement jobs in a subprocess")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="(Legacy) temporal training — no-op")
    train.add_argument("--project-root", required=True)
    train.add_argument("--concept-id", required=True)
    train.add_argument("--config-json", required=True)
    train.add_argument("--model-name", default="")

    infer = sub.add_parser("infer", help="Run dense temporal inference")
    infer.add_argument("--project-root", required=True)
    infer.add_argument("--concept-id", required=True)
    infer.add_argument("--config-json", required=True)

    args = parser.parse_args()

    try:
        if args.command == "train":
            return _run_train(args)
        if args.command == "infer":
            return _run_infer(args)
        _print_line(f"Unsupported command: {args.command}")
        return 2
    except Exception:
        _print_line("Temporal job failed:")
        _print_line(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
