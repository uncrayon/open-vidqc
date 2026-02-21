"""CLI interface for vidqc.

Usage:
    python -m vidqc train --config config.yaml
    python -m vidqc eval --config config.yaml
    python -m vidqc predict --input clip.mp4
    python -m vidqc batch --input samples/ --output predictions.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from vidqc.utils.logging import get_logger

logger = get_logger(__name__)


def _load_config(config_path: str) -> dict:
    """Load config with error handling and initialize logging.

    Args:
        config_path: Path to config YAML file

    Returns:
        Config dict
    """
    from vidqc.config import load_config
    from vidqc.utils.logging import setup_logging

    try:
        config = load_config(config_path)
        
        # Initialize logging based on loaded config
        logging_config = config.get("logging", {})
        level = logging_config.get("level", "INFO")
        log_file = logging_config.get("file", None)
        
        setup_logging(level=level, log_file=log_file)
        
        return config
    except ValueError as e:
        # Fallback logging if config parsing fails
        setup_logging(level="INFO")
        logger.error(f"Invalid configuration: {e}")
        sys.exit(1)


def cmd_train(args: argparse.Namespace) -> None:
    """Handle the 'train' command."""
    from vidqc.train import train_model

    config = _load_config(args.config)

    # Validate labels file exists
    if not Path(args.labels).exists():
        logger.error(f"Labels file not found: {args.labels}")
        sys.exit(1)

    cache_path = "features_cache.pkl" if args.cache else ""

    try:
        classifier, manifest = train_model(
            labels_csv=args.labels,
            output_dir=args.output_dir,
            config=config,
            cache_path=cache_path if args.cache else "features_cache.pkl",
        )
        print("\nTraining complete!")
        print(f"Model: {manifest['model_path']}")
        print(f"CV F1: {manifest['cv_mean_f1']:.3f} +/- {manifest['cv_std_f1']:.3f}")
        print(f"Artifacts saved to: {args.output_dir}")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


def cmd_eval(args: argparse.Namespace) -> None:
    """Handle the 'eval' command."""
    from vidqc.eval import evaluate_model

    config = _load_config(args.config)

    # Resolve model path
    model_path = str(Path(args.model_dir) / "model.json")
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    if not Path(args.labels).exists():
        logger.error(f"Labels file not found: {args.labels}")
        sys.exit(1)

    # Check for training manifest (for CV std criterion)
    manifest_path = str(Path(args.model_dir) / "training_manifest.json")
    if not Path(manifest_path).exists():
        manifest_path = None

    output_path = str(Path(args.model_dir) / "eval_results.json")

    try:
        results = evaluate_model(
            labels_csv=args.labels,
            model_path=model_path,
            output_path=output_path,
            config=config,
            training_manifest_path=manifest_path,
        )
        print("\n=== Evaluation Complete ===")
        print(f"Results saved to: {output_path}")
        print(f"Success criteria met: {results['success_criteria_met']}")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


def cmd_predict(args: argparse.Namespace) -> None:
    """Handle the 'predict' command."""
    from vidqc.predict import predict_clip

    config = _load_config(args.config)

    # Validate input
    if not Path(args.input).exists():
        logger.error(f"Video file not found: {args.input}")
        sys.exit(1)

    # Resolve model path
    model_path = str(Path(args.model_dir) / "model.json")
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    try:
        prediction = predict_clip(args.input, model_path, config)
        print(json.dumps(prediction.model_dump(), indent=2, default=str))
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


def cmd_batch(args: argparse.Namespace) -> None:
    """Handle the 'batch' command."""
    from vidqc.predict import predict_clip

    config = _load_config(args.config)

    # Resolve model path
    model_path = str(Path(args.model_dir) / "model.json")
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    # Find video files in input directory
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path not found: {args.input}")
        sys.exit(1)

    if input_path.is_dir():
        video_files = sorted(input_path.glob("*.mp4"))
        if not video_files:
            logger.error(f"No .mp4 files found in {args.input}")
            sys.exit(1)
    else:
        logger.error(f"Input must be a directory: {args.input}")
        sys.exit(1)

    logger.info(f"Batch prediction on {len(video_files)} clips")

    # Process each clip, write JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed = 0

    with open(output_path, "w") as f:
        for i, video_file in enumerate(video_files):
            logger.info(f"[{i+1}/{len(video_files)}] {video_file.name}")
            try:
                prediction = predict_clip(str(video_file), model_path, config)
                line = json.dumps(prediction.model_dump(), default=str)
                f.write(line + "\n")
                succeeded += 1
            except Exception as e:
                logger.error(f"Failed: {video_file.name}: {e}")
                error_line = json.dumps({
                    "clip_id": video_file.name,
                    "error": str(e),
                })
                f.write(error_line + "\n")
                failed += 1

    print("\nBatch prediction complete.")
    print(f"Results: {succeeded} succeeded, {failed} failed")
    print(f"Output: {args.output}")


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
    train_parser.add_argument(
        "--labels", default="samples/labels.csv", help="Labels CSV path"
    )
    train_parser.add_argument("--output-dir", default="models/", help="Output directory")
    train_parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache extracted features (default: enabled)",
    )

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate the classifier")
    eval_parser.add_argument("--config", default="config.yaml", help="Config file path")
    eval_parser.add_argument(
        "--labels", default="samples/labels.csv", help="Labels CSV path"
    )
    eval_parser.add_argument("--model-dir", default="models/", help="Model directory")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict on a single clip")
    predict_parser.add_argument("--input", required=True, help="Input video file")
    predict_parser.add_argument("--model-dir", default="models/", help="Model directory")
    predict_parser.add_argument("--config", default="config.yaml", help="Config file path")

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch", help="Batch predict on a directory of clips"
    )
    batch_parser.add_argument("--input", required=True, help="Input directory of .mp4 files")
    batch_parser.add_argument(
        "--output", default="predictions.jsonl", help="Output JSONL path"
    )
    batch_parser.add_argument("--model-dir", default="models/", help="Model directory")
    batch_parser.add_argument("--config", default="config.yaml", help="Config file path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Dispatch to command handler
    handlers = {
        "train": cmd_train,
        "eval": cmd_eval,
        "predict": cmd_predict,
        "batch": cmd_batch,
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
