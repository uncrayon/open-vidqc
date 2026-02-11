"""CLI interface for vidqc."""

import argparse
import sys


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="vidqc",
        description="Video artifact detection for AI-generated clips",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the classifier")
    train_parser.add_argument("--config", default="config.yaml", help="Config file path")
    train_parser.add_argument("--labels", default="labels.csv", help="Labels CSV path")
    train_parser.add_argument("--samples", default="samples/", help="Samples directory")
    train_parser.add_argument("--output-dir", default="models/", help="Output directory")
    train_parser.add_argument(
        "--cache/--no-cache",
        dest="cache",
        action="store_true",
        default=True,
        help="Cache extracted features",
    )

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate the classifier")
    eval_parser.add_argument("--config", default="config.yaml", help="Config file path")
    eval_parser.add_argument("--labels", default="labels.csv", help="Labels CSV path")
    eval_parser.add_argument("--samples", default="samples/", help="Samples directory")
    eval_parser.add_argument("--model-dir", default="models/", help="Model directory")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict on a single clip")
    predict_parser.add_argument("--input", required=True, help="Input video file")
    predict_parser.add_argument("--model-dir", default="models/", help="Model directory")
    predict_parser.add_argument("--config", default="config.yaml", help="Config file path")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch predict on multiple clips")
    batch_parser.add_argument("--input", required=True, help="Input directory")
    batch_parser.add_argument(
        "--output", default="predictions.jsonl", help="Output JSONL path"
    )
    batch_parser.add_argument("--model-dir", default="models/", help="Model directory")
    batch_parser.add_argument("--config", default="config.yaml", help="Config file path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Stub implementations - will be filled in later milestones
    print(f"Command '{args.command}' not yet implemented.")
    print("Milestone 1: CLI scaffolding complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
