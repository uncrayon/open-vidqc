"""Tests for bbox review tool evidence mapping."""

from tools.review_bboxes import _map_evidence_boxes_to_frames


def test_map_text_style_pairs_two_boxes_per_pair():
    evidence = {
        "frames": [3, 4, 10, 11],
        "bounding_boxes": [
            [1, 1, 10, 10],
            [2, 2, 11, 11],
            [3, 3, 12, 12],
            [4, 4, 13, 13],
        ],
        "notes": "Frames 3-4: x | Frames 10-11: y",
    }
    box_map = _map_evidence_boxes_to_frames(evidence)
    assert box_map[3] == [[1, 1, 10, 10]]
    assert box_map[4] == [[2, 2, 11, 11]]
    assert box_map[10] == [[3, 3, 12, 12]]
    assert box_map[11] == [[4, 4, 13, 13]]


def test_map_temporal_style_one_box_per_pair():
    evidence = {
        "frames": [1, 2, 8, 9],
        "bounding_boxes": [[5, 5, 20, 20], [7, 7, 21, 21]],
        "notes": "Frames 1-2: a | Frames 8-9: b",
    }
    box_map = _map_evidence_boxes_to_frames(evidence)
    assert box_map[1] == [[5, 5, 20, 20]]
    assert box_map[2] == [[5, 5, 20, 20]]
    assert box_map[8] == [[7, 7, 21, 21]]
    assert box_map[9] == [[7, 7, 21, 21]]


def test_map_fallback_broadcast_when_unpaired():
    evidence = {
        "frames": [2, 5],
        "bounding_boxes": [[1, 1, 4, 4], [2, 2, 5, 5], [3, 3, 6, 6]],
        "notes": "",
    }
    box_map = _map_evidence_boxes_to_frames(evidence)
    assert len(box_map[2]) == 3
    assert len(box_map[5]) == 3
