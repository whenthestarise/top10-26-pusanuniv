"""Python 3.15용 로컬 LLM 실행: CUDA, Apple MPS, CPU를 지원합니다."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def select_device(requested: str = "auto") -> torch.device:
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"지원하지 않는 실행 장치: {requested}")
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA를 사용할 수 없습니다. --device auto 또는 cpu를 사용하세요."
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS를 사용할 수 없습니다. --device cpu를 사용하세요.")
    return torch.device("cuda:0" if requested == "cuda" else requested)


def load_model(model_id: str, device: torch.device):
    """CPU는 FP32, GPU는 FP16으로 로딩합니다. Mac에는 CUDA 양자화를 적용하지 않습니다."""
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map={"": str(device)},
        attn_implementation="sdpa",
        # Transformers의 공식 옵션으로 기존 pread 가중치 읽기 방식을 유지합니다.
        disable_mmap=True,
    )
    model.eval()
    return model


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_allocated_gib(device: torch.device) -> float | None:
    """최대 메모리 통계를 제공하지 않는 장치는 JSON에 null을 기록합니다."""
    if device.type == "cuda":
        return round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)
    return None


def current_allocated_gib(device: torch.device) -> float | None:
    if device.type == "cuda":
        return round(torch.cuda.memory_allocated(device) / 1024**3, 3)
    if device.type == "mps":
        return round(torch.mps.current_allocated_memory() / 1024**3, 3)
    return None


def print_memory_info(device: torch.device) -> None:
    allocated = current_allocated_gib(device)
    if allocated is not None:
        print(f"GPU 현재 할당 메모리: {allocated:.3f} GiB")


def print_runtime_info(device: torch.device) -> None:
    print(f"실행 장치: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(f"전체 VRAM: {total_bytes / 1024**3:.2f} GiB")
        print(f"사용 가능한 VRAM: {free_bytes / 1024**3:.2f} GiB")
    elif device.type == "mps":
        print("GPU: Apple Silicon (MPS), FP16")
    else:
        print("CPU 실행, FP32 (GPU보다 느릴 수 있습니다.)")
