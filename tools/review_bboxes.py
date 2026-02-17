"""Generate an interactive HTML report for bbox review on a single clip.

Usage:
    uv run python -m tools.review_bboxes --input ./samples/text_006.mp4
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path

import cv2

# Allow running as a path script (`python tools/review_bboxes.py`) by adding
# repo root to sys.path only in that execution mode.
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _encode_frame_to_data_url(frame_rgb) -> str:
    """Encode RGB frame to JPEG data URL."""
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Failed to encode frame as JPEG")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _sanitize_bbox(bbox) -> list[int] | None:
    """Normalize bbox to [x1, y1, x2, y2] integer coordinates."""
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        return [int(v) for v in bbox]
    except Exception:
        return None


def _append_box(box_map: dict[int, list[list[int]]], frame_idx: int, bbox) -> None:
    norm = _sanitize_bbox(bbox)
    if norm is None:
        return
    box_map.setdefault(frame_idx, []).append(norm)


def _map_evidence_boxes_to_frames(evidence: dict | None) -> dict[int, list[list[int]]]:
    """Map evidence bounding boxes to sampled frame indices.

    Notes format is parsed first to recover pair-level mapping:
      - TEXT evidence: typically 2 boxes per "Frames a-b" note
      - TEMPORAL evidence: typically 1 box per "Frames a-b" note

    Falls back to direct frame alignment, then to broadcasting all boxes to
    evidence frames if exact pairing is unavailable.
    """
    if evidence is None:
        return {}

    box_map: dict[int, list[list[int]]] = {}
    frames = [int(i) for i in evidence.get("frames", [])]
    boxes = evidence.get("bounding_boxes", []) or []
    notes = evidence.get("notes", "") or ""

    pairs = [(int(a), int(b)) for a, b in re.findall(r"Frames\s+(\d+)-(\d+):", notes)]
    if pairs:
        if len(boxes) >= len(pairs) * 2:
            for i, (frame_a, frame_b) in enumerate(pairs):
                _append_box(box_map, frame_a, boxes[2 * i])
                _append_box(box_map, frame_b, boxes[(2 * i) + 1])
            return box_map
        if len(boxes) >= len(pairs):
            for i, (frame_a, frame_b) in enumerate(pairs):
                _append_box(box_map, frame_a, boxes[i])
                _append_box(box_map, frame_b, boxes[i])
            return box_map

    if len(frames) == len(boxes):
        for frame_idx, bbox in zip(frames, boxes):
            _append_box(box_map, frame_idx, bbox)
        return box_map

    for frame_idx in frames:
        for bbox in boxes:
            _append_box(box_map, frame_idx, bbox)
    return box_map


def _build_html(data: dict) -> str:
    """Create self-contained HTML reviewer."""
    payload = json.dumps(data)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>vidqc bbox review</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --evidence: #e11d48;
      --ocr: #2563eb;
      --accent: #0f766e;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .wrap {{
      display: grid;
      grid-template-columns: 1fr 360px;
      gap: 16px;
      padding: 16px;
      max-width: 1400px;
      margin: 0 auto;
    }}
    .panel {{
      background: var(--panel);
      border-radius: 12px;
      padding: 12px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }}
    .toolbar {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 10px;
    }}
    button {{
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: #fff;
      padding: 8px 10px;
      cursor: pointer;
    }}
    button.ghost {{
      background: #cbd5e1;
      color: #0f172a;
    }}
    label {{
      font-size: 13px;
      color: var(--muted);
    }}
    #viewerCanvas {{
      width: 100%;
      height: auto;
      border-radius: 10px;
      background: #000;
    }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: pre-wrap;
      font-size: 12px;
      color: #0b1020;
    }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 10px;
      font-size: 12px;
      margin-right: 8px;
      margin-bottom: 8px;
      background: #e2e8f0;
      color: #0f172a;
    }}
    .badge.ev {{
      background: #fee2e2;
      color: #9f1239;
    }}
    .legend {{
      display: flex;
      gap: 12px;
      font-size: 12px;
      margin-top: 8px;
      color: var(--muted);
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 2px;
      display: inline-block;
      margin-right: 6px;
      vertical-align: middle;
    }}
    @media (max-width: 1024px) {{
      .wrap {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <div class="toolbar">
        <button id="prevBtn" class="ghost">Prev</button>
        <button id="nextBtn">Next</button>
        <label>Frame
          <input id="frameSlider" type="range" min="0" max="0" step="1" />
        </label>
        <label><input id="showEvidence" type="checkbox" checked /> Evidence boxes</label>
        <label><input id="showOCR" type="checkbox" /> OCR boxes</label>
      </div>
      <canvas id="viewerCanvas"></canvas>
      <div class="legend">
        <span><span class="swatch" style="background: var(--evidence);"></span>Evidence</span>
        <span><span class="swatch" style="background: var(--ocr);"></span>OCR</span>
      </div>
      <div id="frameMeta" class="mono"></div>
    </div>
    <div class="panel">
      <h3 style="margin-top: 0;">Prediction summary</h3>
      <div id="summaryBadges"></div>
      <div id="predictionJson" class="mono"></div>
    </div>
  </div>

  <script>
    const DATA = {payload};
    const frames = DATA.frames;
    const prediction = DATA.prediction;
    let idx = 0;

    const slider = document.getElementById('frameSlider');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const showEvidence = document.getElementById('showEvidence');
    const showOCR = document.getElementById('showOCR');
    const canvas = document.getElementById('viewerCanvas');
    const ctx = canvas.getContext('2d');
    const frameMeta = document.getElementById('frameMeta');
    const summaryBadges = document.getElementById('summaryBadges');
    const predictionJson = document.getElementById('predictionJson');

    slider.max = String(Math.max(frames.length - 1, 0));
    predictionJson.textContent = JSON.stringify(prediction, null, 2);

    function setSummary() {{
      const p = prediction.class_probabilities || {{}};
      const evidenceFrames = prediction.evidence ? prediction.evidence.frames.length : 0;
      const sourceCounts = {{}};
      for (const frame of frames) {{
        for (const region of frame.ocr_regions) {{
          const source = region.source || 'generic';
          sourceCounts[source] = (sourceCounts[source] || 0) + 1;
        }}
      }}
      const sourceBits = Object.keys(sourceCounts)
        .sort()
        .map((k) => `${{k}}=${{sourceCounts[k]}}`)
        .join(', ');
      summaryBadges.innerHTML = `
        <span class="badge">${{prediction.clip_id}}</span>
        <span class="badge">${{prediction.artifact_category}}</span>
        <span class="badge">conf=${{Number(prediction.confidence).toFixed(3)}}</span>
        <span class="badge">P(TEXT)=${{Number(p.TEXT_INCONSISTENCY || 0).toFixed(3)}}</span>
        <span class="badge">P(FLICKER)=${{Number(p.TEMPORAL_FLICKER || 0).toFixed(3)}}</span>
        <span class="badge ev">evidence_frames=${{evidenceFrames}}</span>
        <span class="badge">ocr_sources=${{sourceBits || 'none'}}</span>
      `;
    }}

    function drawBoxes(boxes, color, labels) {{
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.font = '12px ui-monospace, monospace';
      for (let i = 0; i < boxes.length; i++) {{
        const [x1, y1, x2, y2] = boxes[i];
        const w = Math.max(1, x2 - x1);
        const h = Math.max(1, y2 - y1);
        ctx.strokeRect(x1, y1, w, h);
        if (labels && labels[i]) {{
          const label = labels[i];
          const textW = ctx.measureText(label).width + 8;
          ctx.fillStyle = color;
          ctx.fillRect(x1, Math.max(0, y1 - 16), textW, 16);
          ctx.fillStyle = '#fff';
          ctx.fillText(label, x1 + 4, Math.max(11, y1 - 4));
        }}
      }}
    }}

    function renderFrame() {{
      if (frames.length === 0) return;
      const frame = frames[idx];
      const img = new Image();
      img.onload = () => {{
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        if (showEvidence.checked && frame.evidence_boxes.length > 0) {{
          const labels = frame.evidence_boxes.map(() => 'evidence');
          drawBoxes(frame.evidence_boxes, '#e11d48', labels);
        }}
        if (showOCR.checked && frame.ocr_regions.length > 0) {{
          const boxes = frame.ocr_regions.map(r => r.bbox);
          const labels = frame.ocr_regions.map(
            r => `${{r.text}} (${{r.confidence.toFixed(2)}}, ${{r.source || 'generic'}})`
          );
          drawBoxes(boxes, '#2563eb', labels);
        }}

        frameMeta.textContent = [
          `frame=${{frame.index}} / ${{frames.length - 1}}`,
          `timestamp=${{frame.timestamp.toFixed(3)}}s`,
          `is_evidence_frame=${{frame.is_evidence_frame}}`,
          `ocr_regions=${{frame.ocr_regions.length}}`,
          `evidence_boxes=${{frame.evidence_boxes.length}}`,
          `ocr_skipped=${{frame.ocr_skipped}}`,
          `reused_from=${{frame.reused_from === null ? 'n/a' : frame.reused_from}}`
        ].join('\\n');
      }};
      img.src = frame.image;
      slider.value = String(idx);
    }}

    function step(delta) {{
      idx = Math.max(0, Math.min(frames.length - 1, idx + delta));
      renderFrame();
    }}

    prevBtn.addEventListener('click', () => step(-1));
    nextBtn.addEventListener('click', () => step(1));
    slider.addEventListener('input', (e) => {{
      idx = Number(e.target.value);
      renderFrame();
    }});
    showEvidence.addEventListener('change', renderFrame);
    showOCR.addEventListener('change', renderFrame);
    window.addEventListener('keydown', (e) => {{
      if (e.key === 'ArrowLeft') step(-1);
      if (e.key === 'ArrowRight') step(1);
      if (e.key === 'o' || e.key === 'O') {{
        showOCR.checked = !showOCR.checked;
        renderFrame();
      }}
      if (e.key === 'e' || e.key === 'E') {{
        showEvidence.checked = !showEvidence.checked;
        renderFrame();
      }}
    }});

    setSummary();
    renderFrame();
  </script>
</body>
</html>
"""


def build_review_report(
    input_video: str, output_path: str, model_dir: str = "models", config_path: str = "config.yaml"
) -> Path:
    """Generate interactive bbox review report."""
    from vidqc.config import load_config
    from vidqc.features.text_inconsistency import _run_ocr_with_skip
    from vidqc.predict import predict_clip
    from vidqc.utils.video import extract_frames

    config = load_config(config_path)
    model_path = str(Path(model_dir) / "model.json")

    prediction = predict_clip(input_video, model_path, config).model_dump()
    evidence = prediction.get("evidence")
    evidence_map = _map_evidence_boxes_to_frames(evidence)
    evidence_frames = set(evidence.get("frames", []) if evidence else [])

    frame_list = extract_frames(input_video, config)
    frames_rgb = [f.data for f in frame_list]
    timestamps = [f.timestamp for f in frame_list]
    ocr_results = _run_ocr_with_skip(frames_rgb, timestamps, config)
    ocr_by_index = {r.frame_index: r for r in ocr_results}

    review_frames = []
    for frame in frame_list:
        ocr_result = ocr_by_index.get(frame.index)
        ocr_regions = []
        if ocr_result is not None:
            ocr_regions = [
                {
                    "bbox": [int(v) for v in region.bbox],
                    "text": region.text,
                    "confidence": float(region.confidence),
                    "source": region.source,
                }
                for region in ocr_result.regions
            ]
        review_frames.append(
            {
                "index": int(frame.index),
                "timestamp": float(frame.timestamp),
                "image": _encode_frame_to_data_url(frame.data),
                "ocr_regions": ocr_regions,
                "evidence_boxes": evidence_map.get(frame.index, []),
                "is_evidence_frame": frame.index in evidence_frames,
                "ocr_skipped": bool(ocr_result.was_skipped) if ocr_result is not None else False,
                "reused_from": (
                    int(ocr_result.reused_from_index)
                    if ocr_result is not None and ocr_result.reused_from_index is not None
                    else None
                ),
            }
        )

    data = {
        "prediction": prediction,
        "video_path": input_video,
        "frames": review_frames,
    }

    html = _build_html(data)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create interactive HTML report to review bbox detections."
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path (default: reports/bbox_review_<clip>.html)",
    )
    parser.add_argument("--model-dir", default="models/", help="Model directory")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output is None:
        args.output = str(Path("reports") / f"bbox_review_{input_path.stem}.html")

    output = build_review_report(
        input_video=args.input,
        output_path=args.output,
        model_dir=args.model_dir,
        config_path=args.config,
    )
    print(f"Review report generated: {output}")


if __name__ == "__main__":
    main()
