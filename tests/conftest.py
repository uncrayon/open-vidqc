"""Shared test fixtures and helpers for vidqc tests."""

from PIL import ImageFont

# Cross-platform font search order
_FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",  # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Ubuntu/Debian
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Fedora/RHEL
]


def get_test_font(size: int = 48) -> ImageFont.FreeTypeFont:
    """Load a TrueType font that works across macOS and Linux CI.

    Tries system fonts in order, falls back to Pillow's built-in default
    with an explicit size (requires Pillow >= 10.1).

    Args:
        size: Font size in points.

    Returns:
        A PIL ImageFont suitable for rendering readable text in tests.
    """
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    # Pillow >= 10.1 supports size arg on load_default
    return ImageFont.load_default(size=size)
