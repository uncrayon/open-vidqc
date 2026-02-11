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


def add_text_to_frame(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: int = 40,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Add text to a frame using PIL."""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except Exception:
        # Fallback to default font
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


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
) -> None:
    """Generate a clip with text inconsistency artifacts."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    base_text = "OPEN 24 HOURS"
    base_position = (width // 4, height // 2)

    num_frames = fps * duration

    for i in range(num_frames):
        # Create base frame with gradient
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            intensity = int(50 + (y / height) * 100)
            frame[y, :] = [intensity, intensity // 2, intensity // 3]

        # Add Gaussian noise
        noise = np.random.normal(0, 2, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)

        # Mutate text with probability 0.3
        if random.random() < 0.3:
            current_text = mutate_text(base_text)
            # Add jitter
            jitter_x = random.randint(-2, 2)
            jitter_y = random.randint(-2, 2)
            position = (base_position[0] + jitter_x, base_position[1] + jitter_y)
            # Vary font size slightly
            font_size = random.randint(38, 42)
        else:
            current_text = base_text
            position = base_position
            font_size = 40

        frame = add_text_to_frame(frame, current_text, position, font_size)
        out.write(frame)

    out.release()


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
        base_frame, (2 * width // 3 - 80, height // 2),
        (2 * width // 3 + 80, height // 2 + 160), (50, 200, 200), -1
    )

    for i in range(num_frames):
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
        noise = np.random.normal(0, 2, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)

        out.write(frame)

    out.release()


def generate_clean_text(
    output_path: Path, width: int = 1280, height: int = 720, fps: int = 30, duration: int = 5
) -> None:
    """Generate a clean clip with stable text."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    text = "WELCOME"
    position = (width // 3, height // 2)

    num_frames = fps * duration

    for i in range(num_frames):
        # Create stable frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = [80, 100, 120]  # Stable blue-gray background

        # Add minimal noise
        noise = np.random.normal(0, 2, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)

        frame = add_text_to_frame(frame, text, position)
        out.write(frame)

    out.release()


def generate_clean_notext(
    output_path: Path, width: int = 1280, height: int = 720, fps: int = 30, duration: int = 5
) -> None:
    """Generate a clean clip with no text."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    num_frames = fps * duration

    for i in range(num_frames):
        # Create gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            intensity = int(60 + (y / height) * 80)
            frame[y, :] = [intensity, intensity + 20, intensity + 40]

        # Add geometric shapes
        cv2.circle(frame, (width // 2, height // 2), 80, (150, 180, 200), -1)

        # Add minimal noise
        noise = np.random.normal(0, 2, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)

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
                frame, (width // 2 - 100, height // 2 - 80),
                (width // 2 + 100, height // 2 + 80), (50, 50, 200), -1
            )

        # Add minimal noise
        noise = np.random.normal(0, 2, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)

        out.write(frame)

    out.release()


def generate_clean_blur(
    output_path: Path, width: int = 1280, height: int = 720, fps: int = 30, duration: int = 5
) -> None:
    """Generate a clip with text that progressively blurs."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    text = "HELLO WORLD"
    position = (width // 4, height // 2)

    num_frames = fps * duration
    blur_start = num_frames // 2

    for i in range(num_frames):
        # Create stable frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = [70, 90, 110]  # Stable background

        # Add text
        frame = add_text_to_frame(frame, text, position)

        # Apply progressive blur after midpoint
        if i >= blur_start:
            kernel_size = 1 + 2 * ((i - blur_start) // 5)  # 1, 3, 5, 7, 9, ...
            kernel_size = min(kernel_size, 21)  # Cap at 21
            if kernel_size > 1:
                frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

        # Add minimal noise
        noise = np.random.normal(0, 2, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)

        out.write(frame)

    out.release()


def generate_dataset(output_dir: Path, count: int) -> List[dict]:
    """Generate synthetic dataset with all clip types."""
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    clip_id = 0

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
    for i in range(clips_per_type["text_inconsistency"]):
        clip_id += 1
        filename = f"text_{clip_id:03d}.mp4"
        path = output_dir / filename
        print(f"Generating {filename} (TEXT_INCONSISTENCY)...")
        generate_text_inconsistency(path)
        labels.append({
            "clip_id": f"text_{clip_id:03d}",
            "path": f"samples/{filename}",
            "artifact": "true",
            "artifact_category": "TEXT_INCONSISTENCY",
            "source": "synthetic",
            "notes": ""
        })

    # Generate TEMPORAL_FLICKER clips
    for i in range(clips_per_type["temporal_flicker"]):
        clip_id += 1
        filename = f"flicker_{clip_id:03d}.mp4"
        path = output_dir / filename
        print(f"Generating {filename} (TEMPORAL_FLICKER)...")
        generate_temporal_flicker(path)
        labels.append({
            "clip_id": f"flicker_{clip_id:03d}",
            "path": f"samples/{filename}",
            "artifact": "true",
            "artifact_category": "TEMPORAL_FLICKER",
            "source": "synthetic",
            "notes": ""
        })

    # Generate CLEAN_TEXT clips
    for i in range(clips_per_type["clean_text"]):
        clip_id += 1
        filename = f"clean_text_{clip_id:03d}.mp4"
        path = output_dir / filename
        print(f"Generating {filename} (CLEAN_TEXT)...")
        generate_clean_text(path)
        labels.append({
            "clip_id": f"clean_text_{clip_id:03d}",
            "path": f"samples/{filename}",
            "artifact": "false",
            "artifact_category": "NONE",
            "source": "synthetic",
            "notes": ""
        })

    # Generate CLEAN_NOTEXT clips
    for i in range(clips_per_type["clean_notext"]):
        clip_id += 1
        filename = f"clean_notext_{clip_id:03d}.mp4"
        path = output_dir / filename
        print(f"Generating {filename} (CLEAN_NOTEXT)...")
        generate_clean_notext(path)
        labels.append({
            "clip_id": f"clean_notext_{clip_id:03d}",
            "path": f"samples/{filename}",
            "artifact": "false",
            "artifact_category": "NONE",
            "source": "synthetic",
            "notes": ""
        })

    # Generate CLEAN_SCENECUT clips
    for i in range(clips_per_type["clean_scenecut"]):
        clip_id += 1
        filename = f"clean_scenecut_{clip_id:03d}.mp4"
        path = output_dir / filename
        print(f"Generating {filename} (CLEAN_SCENECUT)...")
        generate_clean_scenecut(path)
        labels.append({
            "clip_id": f"clean_scenecut_{clip_id:03d}",
            "path": f"samples/{filename}",
            "artifact": "false",
            "artifact_category": "NONE",
            "source": "synthetic",
            "notes": ""
        })

    # Generate CLEAN_BLUR clips
    for i in range(clips_per_type["clean_blur"]):
        clip_id += 1
        filename = f"clean_blur_{clip_id:03d}.mp4"
        path = output_dir / filename
        print(f"Generating {filename} (CLEAN_BLUR)...")
        generate_clean_blur(path)
        labels.append({
            "clip_id": f"clean_blur_{clip_id:03d}",
            "path": f"samples/{filename}",
            "artifact": "false",
            "artifact_category": "NONE",
            "source": "synthetic",
            "notes": ""
        })

    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic video clips with controllable artifacts"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for generated clips"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="Total number of clips to generate"
    )

    args = parser.parse_args()

    print(f"Generating {args.count} synthetic clips...")
    labels = generate_dataset(args.output, args.count)

    # Write labels.csv
    labels_path = Path("labels.csv")
    print(f"\nWriting {labels_path}...")
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["clip_id", "path", "artifact", "artifact_category", "source", "notes"]
        )
        writer.writeheader()
        writer.writerows(labels)

    print(f"\n✓ Generated {len(labels)} clips")
    print(f"✓ Wrote {labels_path}")
    print("\nVerify clips are playable with: ffprobe samples/*.mp4")


if __name__ == "__main__":
    main()
