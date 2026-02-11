"""
Fetches video clips from OpenAI's Sora API.

Usage:
    uv run python tools/fetch_sora.py --prompts prompts.txt --output samples/ --count 15

Prompts should request scenes with visible text to trigger TEXT_INCONSISTENCY:
  - "A storefront with neon signs at night in a busy city"
  - "A person reading a newspaper in a park, text clearly visible"
  - "A classroom whiteboard with equations being written"

And scenes likely to produce flicker:
  - "A candle flickering in a dark room with reflections on glass"
  - "Ocean waves crashing on rocks at sunset"

Environment:
    OPENAI_API_KEY must be set.
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
POLL_INTERVAL_SECONDS = 15
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


def generate_video(client, prompt: str, duration: int = 8) -> str:
    """Submit a video generation job and return the video ID.

    Args:
        client: OpenAI client instance
        prompt: Text prompt for video generation
        duration: Clip duration in seconds (4, 8, or 12)

    Returns:
        Video job ID
    """
    logger.info(f"Submitting generation job: {prompt[:60]}...")
    video = client.videos.create(
        model="sora-2",
        prompt=prompt,
        duration=duration,
        resolution="1280x720",
    )
    logger.info(f"Job created: {video.id}")
    return video.id


def poll_until_done(client, video_id: str) -> dict:
    """Poll for video generation completion.

    Args:
        client: OpenAI client instance
        video_id: Video job ID

    Returns:
        Completed video object

    Raises:
        TimeoutError: If generation exceeds timeout
        RuntimeError: If generation fails
    """
    elapsed = 0
    while elapsed < POLL_TIMEOUT_SECONDS:
        video = client.videos.retrieve(video_id)
        status = video.status

        if status == "completed":
            logger.info(f"Video {video_id} completed.")
            return video
        elif status == "failed":
            error_msg = getattr(video, "error", "Unknown error")
            raise RuntimeError(f"Video generation failed: {error_msg}")

        logger.info(f"Video {video_id} status: {status} (elapsed: {elapsed}s)")
        time.sleep(POLL_INTERVAL_SECONDS)
        elapsed += POLL_INTERVAL_SECONDS

    raise TimeoutError(
        f"Video {video_id} did not complete within {POLL_TIMEOUT_SECONDS}s"
    )


def download_video(client, video, output_path: str) -> None:
    """Download a completed video to disk.

    Args:
        client: OpenAI client instance
        video: Completed video object
        output_path: Path to save the MP4 file
    """
    logger.info(f"Downloading video to {output_path}")

    # The SDK provides download via the video URL or content
    video_content = client.videos.download(video.id)

    with open(output_path, "wb") as f:
        for chunk in video_content.iter_bytes():
            f.write(chunk)

    logger.info(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch video clips from OpenAI's Sora API"
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
        "--duration",
        type=int,
        default=8,
        choices=[4, 8, 12],
        help="Clip duration in seconds (default: 8)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting index for output filenames (default: 1)",
    )

    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set.")
        logger.error("Export it with: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    # Import OpenAI SDK (deferred to avoid import error when key is missing)
    try:
        from openai import OpenAI
    except ImportError:
        logger.error(
            "openai package not installed. "
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
    client = OpenAI(api_key=api_key)

    # Track prompt → filename mapping
    mapping = []
    failed = []

    for i, prompt in enumerate(prompts):
        clip_index = args.start_index + i
        filename = f"sora_{clip_index:03d}.mp4"
        output_path = output_dir / filename

        try:
            video_id = generate_video(client, prompt, duration=args.duration)
            video = poll_until_done(client, video_id)
            download_video(client, video, str(output_path))

            mapping.append(
                {
                    "index": clip_index,
                    "filename": filename,
                    "prompt": prompt,
                    "video_id": video_id,
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
    mapping_path = output_dir / "sora_mapping.json"
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
