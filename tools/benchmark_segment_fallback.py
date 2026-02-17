"""Benchmark segmented-display fallback overhead on a single clip.

Usage:
    uv run python tools/benchmark_segment_fallback.py --input /path/to/clip.mp4
"""

import argparse
import copy
import time
import sys
from pathlib import Path

# Allow running as a path script (`python tools/benchmark_segment_fallback.py`).
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from vidqc.config import load_config
from vidqc.features.text_inconsistency import extract_text_features


def _run_once(video_path: str, config: dict) -> float:
    start = time.perf_counter()
    extract_text_features(video_path, config=config)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark segment fallback overhead")
    parser.add_argument("--input", required=True, help="Input clip path")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument(
        "--max-overhead",
        type=float,
        default=0.25,
        help="Allowed overhead fraction (default: 0.25 => 25%%)",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)

    cfg_disabled = copy.deepcopy(base_cfg)
    cfg_disabled.setdefault("text", {}).setdefault("segment_display_fallback", {})[
        "enabled"
    ] = False

    cfg_enabled = copy.deepcopy(base_cfg)
    cfg_enabled.setdefault("text", {}).setdefault("segment_display_fallback", {})[
        "enabled"
    ] = True

    t_disabled = _run_once(args.input, cfg_disabled)
    t_enabled = _run_once(args.input, cfg_enabled)

    overhead = 0.0
    if t_disabled > 0:
        overhead = (t_enabled - t_disabled) / t_disabled

    print(f"baseline_disabled_s={t_disabled:.3f}")
    print(f"fallback_enabled_s={t_enabled:.3f}")
    print(f"overhead_fraction={overhead:.3f}")

    if overhead > args.max_overhead:
        raise SystemExit(
            f"Overhead {overhead:.3f} exceeds allowed {args.max_overhead:.3f}"
        )


if __name__ == "__main__":
    main()
