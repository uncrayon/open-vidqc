"""
Generates synthetic video clips with controllable artifacts.

Usage:
    uv run python tools/make_dataset.py --output samples/ --count 35

Generates six types:
  1. TEXT_INCONSISTENCY: Renders text on frames, applies per-frame random
     mutations (character swaps, jitter, thickness changes, partial erasure).
  2. TEMPORAL_FLICKER: Renders a stable scene, injects random brightness/hue
     shifts and texture swaps on random frame subsets.
  3. CLEAN_TEXT: Renders stable text and scenes with no perturbation.
  4. CLEAN_NOTEXT: Renders scenes without any text.
  5. CLEAN_SCENECUT: Renders two distinct stable scenes with a hard cut.
  6. CLEAN_BLUR: Renders text that progressively blurs (defocus transition).
"""

import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/System/Library/Fonts/Helvetica.ttc"
ROTATION_CHOICES = [90, 180, 270]
ANCHOR_POSITIONS = {
    "top_left": (0.15, 0.15),
    "top_center": (0.50, 0.15),
    "top_right": (0.85, 0.15),
    "center_left": (0.15, 0.50),
    "center": (0.50, 0.50),
    "center_right": (0.85, 0.50),
    "bottom_left": (0.15, 0.85),
    "bottom_center": (0.50, 0.85),
    "bottom_right": (0.85, 0.85),
}


def add_gaussian_noise(frame: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Add Gaussian noise using signed math and clipping (no uint8 wraparound)."""
    noise = np.random.normal(0.0, sigma, frame.shape).astype(np.float32)
    noisy = frame.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _load_font(font_size: int):
    """Load font with fallback."""
    try:
        return ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        return ImageFont.load_default()


def _get_anchor_center(width: int, height: int, anchor_name: str) -> tuple[int, int]:
    """Return anchor center position in pixel coordinates."""
    x_frac, y_frac = ANCHOR_POSITIONS.get(anchor_name, ANCHOR_POSITIONS["center"])
    x = int(width * x_frac)
    y = int(height * y_frac)
    return x, y


def _build_rotated_text_layer(
    text: str, font_size: int, color: tuple[int, int, int], rotation: int
) -> Image.Image:
    """Render text to transparent RGBA layer and rotate in 90-degree steps."""
    if rotation not in {0, 90, 180, 270}:
        raise ValueError(f"Rotation must be one of 0, 90, 180, 270 (got {rotation})")

    font = _load_font(font_size)
    sample_text = text if text else " "

    # Compute text bounds first, then draw on padded transparent layer.
    tmp_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    tmp_draw = ImageDraw.Draw(tmp_img)
    try:
        left, top, right, bottom = tmp_draw.textbbox((0, 0), sample_text, font=font)
        text_w = max(1, right - left)
        text_h = max(1, bottom - top)
        offset_x = -left
        offset_y = -top
    except AttributeError:
        text_w, text_h = tmp_draw.textsize(sample_text, font=font)
        text_w = max(1, text_w)
        text_h = max(1, text_h)
        offset_x = 0
        offset_y = 0

    pad = 4
    layer = Image.new("RGBA", (text_w + 2 * pad, text_h + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.text(
        (pad + offset_x, pad + offset_y),
        sample_text,
        font=font,
        fill=(color[0], color[1], color[2], 255),
    )

    if rotation == 0:
        return layer

    resample = getattr(Image, "Resampling", Image).BICUBIC
    return layer.rotate(rotation, expand=True, resample=resample)


def _clamp_text_position(
    x: int, y: int, text_w: int, text_h: int, frame_w: int, frame_h: int
) -> tuple[int, int]:
    """Clamp text position so the text layer stays fully in-frame."""
    max_x = max(0, frame_w - text_w)
    max_y = max(0, frame_h - text_h)
    return max(0, min(x, max_x)), max(0, min(y, max_y))


def _format_text_notes(rotation: int, anchor_name: str) -> str:
    """Format rotation/anchor metadata for labels notes column."""
    return f"rot={rotation};anchor={anchor_name}"


def add_text_to_frame(
    frame: np.ndarray,
    text: str,
    anchor_name: str,
    font_size: int = 40,
    color: Tuple[int, int, int] = (255, 255, 255),
    rotation: int = 0,
    jitter: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Add text to a frame using anchor-based placement and discrete rotation."""
    frame_h, frame_w = frame.shape[:2]
    text_layer = _build_rotated_text_layer(text, font_size, color, rotation)

    center_x, center_y = _get_anchor_center(frame_w, frame_h, anchor_name)
    x = int(center_x - (text_layer.width / 2) + jitter[0])
    y = int(center_y - (text_layer.height / 2) + jitter[1])
    x, y = _clamp_text_position(x, y, text_layer.width, text_layer.height, frame_w, frame_h)

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    pil_img.alpha_composite(text_layer, dest=(x, y))
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def mutate_text(text: str) -> str:
    """Randomly mutate text: swap characters, drop characters."""
    if not text or random.random() > 0.7:
        return text

    text_list = list(text)
    mutation_type = random.choice(["swap", "drop", "substitute"])

    if mutation_type == "swap" and len(text_list) > 1:
        idx = random.randint(0, len(text_list) - 2)
        text_list[idx], text_list[idx + 1] = text_list[idx + 1], text_list[idx]
    elif mutation_type == "drop" and len(text_list) > 1:
        idx = random.randint(0, len(text_list) - 1)
        text_list.pop(idx)
    elif mutation_type == "substitute" and len(text_list) > 0:
        idx = random.randint(0, len(text_list) - 1)
        text_list[idx] = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    return "".join(text_list)


def generate_text_inconsistency(
    output_path: Path, width: int = 1280, height: int = 720, fps: int = 30, duration: int = 5
) -> dict[str, str]:
    """Generate a clip with text inconsistency artifacts."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    base_text = "OPEN 24 HOURS"
    anchor_name = random.choice(list(ANCHOR_POSITIONS.keys()))
    rotation = random.choice(ROTATION_CHOICES)

    num_frames = fps * duration

    for _ in range(num_frames):
        # Create base frame with gradient
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            intensity = int(50 + (y / height) * 100)
            frame[y, :] = [intensity, intensity // 2, intensity // 3]

        # Add Gaussian noise
        frame = add_gaussian_noise(frame, sigma=2.0)

        # Mutate text with probability 0.3
        if random.random() < 0.3:
            current_text = mutate_text(base_text)
            # Add jitter
            jitter_x = random.randint(-3, 3)
            jitter_y = random.randint(-3, 3)
            # Vary font size slightly
            font_size = random.randint(38, 42)
            jitter = (jitter_x, jitter_y)
        else:
            current_text = base_text
            font_size = 40
            jitter = (0, 0)

        frame = add_text_to_frame(
            frame,
            current_text,
            anchor_name,
            font_size=font_size,
            rotation=rotation,
            jitter=jitter,
        )
        out.write(frame)

    out.release()
    return {"rotation": str(rotation), "anchor": anchor_name}


def generate_temporal_flicker(
    output_path: Path, width: int = 1280, height: int = 720, fps: int = 30, duration: int = 5
) -> None:
    """Generate a clip with temporal flicker artifacts."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    num_frames = fps * duration

    # Create base scene with geometric shapes
    base_frame = np.zeros((height, width, 3), dtype=np.uint8)
    base_frame[:] = [100, 120, 140]  # Gray-blue background
    cv2.circle(base_frame, (width // 3, height // 3), 100, (200, 200, 50), -1)
    cv2.rectangle(
        base_frame,
        (2 * width // 3 - 80, height // 2),
        (2 * width // 3 + 80, height // 2 + 160),
        (50, 200, 200),
        -1,
    )

    for _ in range(num_frames):
        frame = base_frame.copy()

        # Add flicker with probability 0.25
        if random.random() < 0.25:
            # Brightness shift
            brightness_factor = random.uniform(0.7, 1.3)
            frame = np.clip(frame * brightness_factor, 0, 255).astype(np.uint8)

            # Hue rotation (shift colors)
            if random.random() < 0.5:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hue_shift = random.randint(-15, 15)
                hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + hue_shift) % 180
                hsv[:, :, 0] = hsv[:, :, 0].astype(np.uint8)
                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Add minimal noise
        frame = add_gaussian_noise(frame, sigma=2.0)

        out.write(frame)

    out.release()


def generate_clean_text(
    output_path: Path, width: int = 1280, height: int = 720, fps: int = 30, duration: int = 5
) -> dict[str, str]:
    """Generate a clean clip with stable text."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    text = "WELCOME"
    anchor_name = random.choice(list(ANCHOR_POSITIONS.keys()))
    rotation = random.choice(ROTATION_CHOICES)

    num_frames = fps * duration

    for _ in range(num_frames):
        # Create stable frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = [80, 100, 120]  # Stable blue-gray background

        # Add minimal noise
        frame = add_gaussian_noise(frame, sigma=2.0)

        frame = add_text_to_frame(frame, text, anchor_name, rotation=rotation)
        out.write(frame)

    out.release()
    return {"rotation": str(rotation), "anchor": anchor_name}


def generate_clean_notext(
    output_path: Path, width: int = 1280, height: int = 720, fps: int = 30, duration: int = 5
) -> None:
    """Generate a clean clip with no text."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    num_frames = fps * duration

    for _ in range(num_frames):
        # Create gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            intensity = int(60 + (y / height) * 80)
            frame[y, :] = [intensity, intensity + 20, intensity + 40]

        # Add geometric shapes
        cv2.circle(frame, (width // 2, height // 2), 80, (150, 180, 200), -1)

        # Add minimal noise
        frame = add_gaussian_noise(frame, sigma=2.0)

        out.write(frame)

    out.release()


def generate_clean_scenecut(
    output_path: Path, width: int = 1280, height: int = 720, fps: int = 30, duration: int = 5
) -> None:
    """Generate a clean clip with a hard scene cut."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    num_frames = fps * duration
    cut_frame = num_frames // 2

    for i in range(num_frames):
        if i < cut_frame:
            # Scene A: green background with white circle
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = [50, 150, 50]  # Green
            cv2.circle(frame, (width // 2, height // 2), 100, (255, 255, 255), -1)
        else:
            # Scene B: gray background with red rectangle
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = [100, 100, 100]  # Gray
            cv2.rectangle(
                frame,
                (width // 2 - 100, height // 2 - 80),
                (width // 2 + 100, height // 2 + 80),
                (50, 50, 200),
                -1,
            )

        # Add minimal noise
        frame = add_gaussian_noise(frame, sigma=2.0)

        out.write(frame)

    out.release()


def generate_clean_blur(
    output_path: Path, width: int = 1280, height: int = 720, fps: int = 30, duration: int = 5
) -> dict[str, str]:
    """Generate a clip with text that progressively blurs."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    text = "HELLO WORLD"
    anchor_name = random.choice(list(ANCHOR_POSITIONS.keys()))
    rotation = random.choice(ROTATION_CHOICES)

    num_frames = fps * duration
    blur_start = num_frames // 2

    for i in range(num_frames):
        # Create stable frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = [70, 90, 110]  # Stable background

        # Add text
        frame = add_text_to_frame(frame, text, anchor_name, rotation=rotation)

        # Apply progressive blur after midpoint
        if i >= blur_start:
            kernel_size = 1 + 2 * ((i - blur_start) // 5)  # 1, 3, 5, 7, 9, ...
            kernel_size = min(kernel_size, 21)  # Cap at 21
            if kernel_size > 1:
                frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

        # Add minimal noise
        frame = add_gaussian_noise(frame, sigma=2.0)

        out.write(frame)

    out.release()
    return {"rotation": str(rotation), "anchor": anchor_name}


def generate_dataset(output_dir: Path, count: int, seed: int = 42) -> List[dict]:
    """Generate synthetic dataset with all clip types."""
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    labels = []
    clip_id = 0
    path_prefix = output_dir.name

    # Calculate clips per type (balanced distribution)
    clips_per_type = {
        "text_inconsistency": max(1, count // 6),
        "temporal_flicker": max(1, count // 6),
        "clean_text": max(1, count // 6),
        "clean_notext": max(1, count // 6),
        "clean_scenecut": max(1, count // 6),
        "clean_blur": max(1, count // 6),
    }

    # Generate TEXT_INCONSISTENCY clips
    for _ in range(clips_per_type["text_inconsistency"]):
        clip_id += 1
        filename = f"text_{clip_id:03d}.mp4"
        path = output_dir / filename
        print(f"Generating {filename} (TEXT_INCONSISTENCY)...")
        text_meta = generate_text_inconsistency(path)
        labels.append(
            {
                "clip_id": f"text_{clip_id:03d}",
                "path": f"{path_prefix}/{filename}",
                "artifact": "true",
                "artifact_category": "TEXT_INCONSISTENCY",
                "source": "synthetic",
                "notes": _format_text_notes(
                    int(text_meta["rotation"]), text_meta["anchor"]
                ),
            }
        )

    # Generate TEMPORAL_FLICKER clips
    for _ in range(clips_per_type["temporal_flicker"]):
        clip_id += 1
        filename = f"flicker_{clip_id:03d}.mp4"
        path = output_dir / filename
        print(f"Generating {filename} (TEMPORAL_FLICKER)...")
        generate_temporal_flicker(path)
        labels.append(
            {
                "clip_id": f"flicker_{clip_id:03d}",
                "path": f"{path_prefix}/{filename}",
                "artifact": "true",
                "artifact_category": "TEMPORAL_FLICKER",
                "source": "synthetic",
                "notes": "",
            }
        )

    # Generate CLEAN_TEXT clips
    for _ in range(clips_per_type["clean_text"]):
        clip_id += 1
        filename = f"clean_text_{clip_id:03d}.mp4"
        path = output_dir / filename
        print(f"Generating {filename} (CLEAN_TEXT)...")
        text_meta = generate_clean_text(path)
        labels.append(
            {
                "clip_id": f"clean_text_{clip_id:03d}",
                "path": f"{path_prefix}/{filename}",
                "artifact": "false",
                "artifact_category": "NONE",
                "source": "synthetic",
                "notes": _format_text_notes(
                    int(text_meta["rotation"]), text_meta["anchor"]
                ),
            }
        )

    # Generate CLEAN_NOTEXT clips
    for _ in range(clips_per_type["clean_notext"]):
        clip_id += 1
        filename = f"clean_notext_{clip_id:03d}.mp4"
        path = output_dir / filename
        print(f"Generating {filename} (CLEAN_NOTEXT)...")
        generate_clean_notext(path)
        labels.append(
            {
                "clip_id": f"clean_notext_{clip_id:03d}",
                "path": f"{path_prefix}/{filename}",
                "artifact": "false",
                "artifact_category": "NONE",
                "source": "synthetic",
                "notes": "",
            }
        )

    # Generate CLEAN_SCENECUT clips
    for _ in range(clips_per_type["clean_scenecut"]):
        clip_id += 1
        filename = f"clean_scenecut_{clip_id:03d}.mp4"
        path = output_dir / filename
        print(f"Generating {filename} (CLEAN_SCENECUT)...")
        generate_clean_scenecut(path)
        labels.append(
            {
                "clip_id": f"clean_scenecut_{clip_id:03d}",
                "path": f"{path_prefix}/{filename}",
                "artifact": "false",
                "artifact_category": "NONE",
                "source": "synthetic",
                "notes": "",
            }
        )

    # Generate CLEAN_BLUR clips
    for _ in range(clips_per_type["clean_blur"]):
        clip_id += 1
        filename = f"clean_blur_{clip_id:03d}.mp4"
        path = output_dir / filename
        print(f"Generating {filename} (CLEAN_BLUR)...")
        text_meta = generate_clean_blur(path)
        labels.append(
            {
                "clip_id": f"clean_blur_{clip_id:03d}",
                "path": f"{path_prefix}/{filename}",
                "artifact": "false",
                "artifact_category": "NONE",
                "source": "synthetic",
                "notes": _format_text_notes(
                    int(text_meta["rotation"]), text_meta["anchor"]
                ),
            }
        )

    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic video clips with controllable artifacts"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory for generated clips"
    )
    parser.add_argument(
        "--count", type=int, default=30, help="Total number of clips to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation",
    )

    args = parser.parse_args()

    print(f"Generating {args.count} synthetic clips (seed={args.seed})...")
    labels = generate_dataset(args.output, args.count, seed=args.seed)

    # Write labels.csv
    labels_path = args.output / "labels.csv" # I'm not sure this is "clean code" lol
    print(f"\nWriting {labels_path}...")
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip_id",
                "path",
                "artifact",
                "artifact_category",
                "source",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(labels)

    print(f"\n✓ Generated {len(labels)} clips")
    print(f"✓ Wrote {labels_path}")
    print("\nVerify clips are playable with: ffprobe samples/*.mp4")


if __name__ == "__main__":
    main()
