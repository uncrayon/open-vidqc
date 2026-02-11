"""
vidqc CLI entry point.

Usage:
    python -m vidqc train --config config.yaml
    python -m vidqc eval --config config.yaml
    python -m vidqc predict --input clip.mp4
    python -m vidqc batch --input samples/ --output predictions.jsonl
"""

from vidqc.cli import main

if __name__ == "__main__":
    main()
