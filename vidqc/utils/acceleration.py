"""GPU/CPU acceleration resolution helpers."""

from typing import Optional

_ALLOWED_XGBOOST_DEVICES = {"auto", "cpu", "cuda"}


def detect_runtime_cuda_available() -> tuple[bool, str]:
    """Detect whether CUDA is available at runtime via torch."""
    try:
        import torch
    except ImportError:
        return False, "torch_not_installed"

    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - defensive guard
        return False, f"torch_cuda_check_failed:{exc.__class__.__name__}"

    return available, f"torch.cuda.is_available={available}"


def detect_xgboost_cuda_build() -> tuple[bool, str]:
    """Detect whether installed XGBoost was built with CUDA support."""
    try:
        import xgboost as xgb
    except ImportError:
        return False, "xgboost_not_installed"

    try:
        build_info = xgb.build_info()
    except Exception as exc:  # pragma: no cover - defensive guard
        return False, f"xgboost_build_info_failed:{exc.__class__.__name__}"

    use_cuda = bool(build_info.get("USE_CUDA", False))
    return use_cuda, f"xgboost.build_info.USE_CUDA={use_cuda}"


def resolve_ocr_gpu(
    raw_value: object, runtime_cuda_available: Optional[bool] = None
) -> tuple[bool, str]:
    """Resolve OCR GPU config into a concrete bool with a reason."""
    if isinstance(raw_value, bool):
        return raw_value, f"requested={raw_value}"

    value = str(raw_value).lower().strip()
    if value == "true":
        return True, "requested=true"
    if value == "false":
        return False, "requested=false"

    if runtime_cuda_available is None:
        runtime_cuda_available, runtime_reason = detect_runtime_cuda_available()
    else:
        runtime_reason = f"runtime_cuda_override={runtime_cuda_available}"

    return bool(runtime_cuda_available), (
        f"requested={value or 'auto'}; {runtime_reason}"
    )


def resolve_xgboost_device(
    raw_value: object,
    runtime_cuda_available: Optional[bool] = None,
    xgboost_cuda_build: Optional[bool] = None,
) -> tuple[str, str]:
    """Resolve XGBoost device config to 'cpu' or 'cuda' with safe fallback."""
    requested = str(raw_value if raw_value is not None else "auto").lower().strip()
    if requested not in _ALLOWED_XGBOOST_DEVICES:
        requested = "auto"
        unknown_reason = "requested_invalid_fallback_to_auto"
    else:
        unknown_reason = ""

    if requested == "cpu":
        return "cpu", "requested=cpu"

    if runtime_cuda_available is None:
        runtime_cuda_available, runtime_reason = detect_runtime_cuda_available()
    else:
        runtime_reason = f"runtime_cuda_override={runtime_cuda_available}"

    if xgboost_cuda_build is None:
        xgboost_cuda_build, build_reason = detect_xgboost_cuda_build()
    else:
        build_reason = f"xgboost_cuda_build_override={xgboost_cuda_build}"

    runtime_ok = bool(runtime_cuda_available)
    build_ok = bool(xgboost_cuda_build)

    prefix = f"{unknown_reason}; " if unknown_reason else ""

    if runtime_ok and build_ok:
        return "cuda", (
            f"{prefix}requested={requested}; {runtime_reason}; {build_reason}"
        )

    if requested == "cuda":
        return "cpu", (
            f"{prefix}requested=cuda; fallback=cpu; "
            f"{runtime_reason}; {build_reason}"
        )

    return "cpu", f"{prefix}requested=auto; {runtime_reason}; {build_reason}"
