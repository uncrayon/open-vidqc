"""
Fetches video clips from OpenAI's Sora API.

Usage:
    uv run python tools/fetch_sora.py --prompts prompts.txt --output samples/ --count 15

Environment:
    OPENAI_API_KEY must be set.

Note: Implementation deferred to Milestone 6.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Fetch clips from Sora API")
    parser.add_argument("--prompts", required=True, help="Prompts file path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--count", type=int, default=15, help="Number of clips")

    _ = parser.parse_args()  # Parse to validate args, but don't use yet

    print("fetch_sora.py: Not yet implemented (Milestone 6)")
    print("Set OPENAI_API_KEY and implement Sora API integration when ready.")
    sys.exit(1)


if __name__ == "__main__":
    main()
