"""
Fetches video clips from Google's Veo-3 (Gemini API).

Usage:
    uv run python tools/fetch_veo3.py --prompts prompts.txt --output samples/ --count 15

Prompts should request scenes with visible text to trigger TEXT_INCONSISTENCY:
  - "A storefront with neon signs at night in a busy city"
  - "A person reading a newspaper in a park, text clearly visible"
  - "A classroom whiteboard with equations being written"

And scenes likely to produce flicker:
  - "A candle flickering in a dark room with reflections on glass"
  - "Ocean waves crashing on rocks at sunset"

Environment:
    GOOGLE_API_KEY must be set (or use gcloud auth for Vertex AI).
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Poll interval and timeout for video generation
POLL_INTERVAL_SECONDS = 20
POLL_TIMEOUT_SECONDS = 600


def load_prompts(prompts_path: str) -> list[str]:
    """Load prompts from a text file (one per line, # comments ignored)."""
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


def generate_video(client, types_module, prompt: str) -> object:
    """Submit a video generation job and return the operation.

    Args:
        client: Google GenAI client instance
        types_module: google.genai.types module
        prompt: Text prompt for video generation

    Returns:
        Operation object to poll for completion
    """
    logger.info(f"Submitting generation job: {prompt[:60]}...")
    operation = client.models.generate_videos(
        model="veo-3.0-generate-preview",
        prompt=prompt,
        config=types_module.GenerateVideosConfig(
            aspect_ratio="16:9",
            resolution="720p",
        ),
    )
    logger.info("Job submitted, polling for completion...")
    return operation


def poll_until_done(client, operation) -> object:
    """Poll for video generation completion.

    Args:
        client: Google GenAI client instance
        operation: Operation object from generate_videos

    Returns:
        Completed operation

    Raises:
        TimeoutError: If generation exceeds timeout
        RuntimeError: If generation fails
    """
    elapsed = 0
    while elapsed < POLL_TIMEOUT_SECONDS:
        if operation.done:
            if operation.error:
                raise RuntimeError(f"Video generation failed: {operation.error}")
            logger.info("Video generation completed.")
            return operation

        logger.info(f"Generating... (elapsed: {elapsed}s)")
        time.sleep(POLL_INTERVAL_SECONDS)
        elapsed += POLL_INTERVAL_SECONDS
        operation = client.operations.get(operation)

    raise TimeoutError(
        f"Video generation did not complete within {POLL_TIMEOUT_SECONDS}s"
    )


def save_video(client, operation, output_path: str) -> None:
    """Download and save a completed video to disk.

    Args:
        client: Google GenAI client instance
        operation: Completed operation with video result
        output_path: Path to save the MP4 file
    """
    generated_video = operation.result.generated_videos[0]
    client.files.download(file=generated_video.video)
    generated_video.video.save(output_path)
    logger.info(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch video clips from Google's Veo-3 API"
    )
    parser.add_argument(
        "--prompts", required=True, help="Path to prompts file (one per line)"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for MP4 files"
    )
    parser.add_argument(
        "--count", type=int, default=15, help="Max number of clips to generate"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting index for output filenames (default: 1)",
    )

    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable is not set.")
        logger.error("Export it with: export GOOGLE_API_KEY='...'")
        logger.error("Or use gcloud auth for Vertex AI authentication.")
        sys.exit(1)

    # Import Google GenAI SDK (deferred to avoid import error when key is missing)
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        logger.error(
            "google-genai package not installed. "
            "Install with: uv sync --group dataset"
        )
        sys.exit(1)

    # Load prompts
    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        logger.error(f"Prompts file not found: {args.prompts}")
        sys.exit(1)

    prompts = load_prompts(str(prompts_path))
    if not prompts:
        logger.error("No prompts found in file.")
        sys.exit(1)

    # Limit to --count
    prompts = prompts[: args.count]
    logger.info(f"Loaded {len(prompts)} prompts (max {args.count})")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize client
    client = genai.Client(api_key=api_key)

    # Track prompt → filename mapping
    mapping = []
    failed = []

    for i, prompt in enumerate(prompts):
        clip_index = args.start_index + i
        filename = f"veo3_{clip_index:03d}.mp4"
        output_path = output_dir / filename

        try:
            operation = generate_video(client, genai_types, prompt)
            operation = poll_until_done(client, operation)
            save_video(client, operation, str(output_path))

            mapping.append(
                {
                    "index": clip_index,
                    "filename": filename,
                    "prompt": prompt,
                    "status": "success",
                }
            )
        except Exception as e:
            logger.error(f"Failed for prompt '{prompt[:50]}...': {e}")
            failed.append(
                {
                    "index": clip_index,
                    "prompt": prompt,
                    "error": str(e),
                }
            )

    # Save mapping log
    mapping_path = output_dir / "veo3_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(
            {"generated": mapping, "failed": failed},
            f,
            indent=2,
        )

    logger.info(f"Mapping saved to {mapping_path}")
    logger.info(
        f"Done: {len(mapping)} succeeded, {len(failed)} failed "
        f"out of {len(prompts)} prompts"
    )

    if failed:
        logger.warning("Failed prompts:")
        for item in failed:
            logger.warning(f"  - {item['prompt'][:60]}...: {item['error']}")


if __name__ == "__main__":
    main()
