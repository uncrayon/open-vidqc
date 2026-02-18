"""Tests for OCR bbox size diagnostics in text feature extraction."""

from vidqc.features.text_inconsistency import (
    OCRFrameResult,
    OCRRegion,
    _format_bbox_stats_for_log,
    _summarize_ocr_bbox_sizes,
)


def test_summarize_ocr_bbox_sizes_empty():
    stats = _summarize_ocr_bbox_sizes([])

    assert stats["total_regions"] == 0
    assert stats["min_width"] == 0
    assert stats["min_height"] == 0
    assert stats["max_width"] == 0
    assert stats["max_height"] == 0
    assert stats["tiny_lt7_count"] == 0
    assert stats["tiny_lt3_count"] == 0


def test_summarize_ocr_bbox_sizes_counts():
    ocr_results = [
        OCRFrameResult(
            frame_index=0,
            timestamp=0.0,
            regions=[
                OCRRegion(bbox=[0, 0, 10, 20], text="A", confidence=0.9, frame_index=0),
                OCRRegion(bbox=[3, 3, 6, 8], text="B", confidence=0.8, frame_index=0),
            ],
            was_skipped=False,
            reused_from_index=None,
        ),
        OCRFrameResult(
            frame_index=1,
            timestamp=0.2,
            regions=[
                OCRRegion(bbox=[2, 2, 3, 3], text="C", confidence=0.7, frame_index=1),
            ],
            was_skipped=False,
            reused_from_index=None,
        ),
    ]

    stats = _summarize_ocr_bbox_sizes(ocr_results)

    assert stats["total_regions"] == 3
    assert stats["min_width"] == 1
    assert stats["min_height"] == 1
    assert stats["max_width"] == 10
    assert stats["max_height"] == 20
    assert stats["tiny_lt7_count"] == 2
    assert stats["tiny_lt3_count"] == 1

    text = _format_bbox_stats_for_log(stats)
    assert "regions=3" in text
    assert "min=1x1" in text
    assert "max=10x20" in text
    assert "tiny<7=2" in text
    assert "tiny<3=1" in text
