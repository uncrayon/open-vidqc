"""
Fetches video clips from Google's Veo-3 (Vertex AI / Gemini API).

Usage:
    uv run python tools/fetch_veo3.py --prompts prompts.txt --output samples/ --count 15

Environment:
    GOOGLE_API_KEY must be set (or use gcloud auth for Vertex AI).

Note: Implementation deferred to Milestone 6.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Fetch clips from Veo-3 API")
    parser.add_argument("--prompts", required=True, help="Prompts file path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--count", type=int, default=15, help="Number of clips")

    _ = parser.parse_args()  # Parse to validate args, but don't use yet

    print("fetch_veo3.py: Not yet implemented (Milestone 6)")
    print("Set GOOGLE_API_KEY and implement Veo-3 API integration when ready.")
    sys.exit(1)


if __name__ == "__main__":
    main()
