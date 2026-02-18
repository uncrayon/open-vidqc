"""Tests for OCR/XGBoost device resolution helpers."""

from vidqc.utils.acceleration import resolve_ocr_gpu, resolve_xgboost_device


def test_resolve_ocr_gpu_explicit_values():
    use_gpu, reason = resolve_ocr_gpu(True)
    assert use_gpu is True
    assert "requested=True" in reason

    use_gpu, reason = resolve_ocr_gpu("false")
    assert use_gpu is False
    assert "requested=false" in reason


def test_resolve_ocr_gpu_auto_with_runtime_override():
    use_gpu, reason = resolve_ocr_gpu("auto", runtime_cuda_available=True)
    assert use_gpu is True
    assert "runtime_cuda_override=True" in reason

    use_gpu, reason = resolve_ocr_gpu("auto", runtime_cuda_available=False)
    assert use_gpu is False
    assert "runtime_cuda_override=False" in reason


def test_resolve_xgboost_device_auto_without_runtime_cuda():
    resolved, reason = resolve_xgboost_device(
        "auto",
        runtime_cuda_available=False,
        xgboost_cuda_build=True,
    )
    assert resolved == "cpu"
    assert "requested=auto" in reason


def test_resolve_xgboost_device_auto_without_cuda_build():
    resolved, reason = resolve_xgboost_device(
        "auto",
        runtime_cuda_available=True,
        xgboost_cuda_build=False,
    )
    assert resolved == "cpu"
    assert "requested=auto" in reason


def test_resolve_xgboost_device_auto_with_cuda_ready():
    resolved, reason = resolve_xgboost_device(
        "auto",
        runtime_cuda_available=True,
        xgboost_cuda_build=True,
    )
    assert resolved == "cuda"
    assert "requested=auto" in reason


def test_resolve_xgboost_device_explicit_cuda_falls_back():
    resolved, reason = resolve_xgboost_device(
        "cuda",
        runtime_cuda_available=False,
        xgboost_cuda_build=True,
    )
    assert resolved == "cpu"
    assert "fallback=cpu" in reason


def test_resolve_xgboost_device_explicit_cpu():
    resolved, reason = resolve_xgboost_device(
        "cpu",
        runtime_cuda_available=True,
        xgboost_cuda_build=True,
    )
    assert resolved == "cpu"
    assert reason == "requested=cpu"
