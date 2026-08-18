"""AutoDL/local deployment adaptation helpers for a single 2-GPU host.

This module contains environment inspection only. It does not implement any
assignment algorithm or alter the official Modal execution path.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from dataclasses import dataclass

import torch


EXPECTED_PYTHON = (3, 12)
EXPECTED_TORCH_PREFIX = "2.10."
EXPECTED_CUDA_PREFIX = "12.9"
EXPECTED_VLLM = "0.19.1"
MINIMUM_CUDA_129_DRIVER = (575, 51, 3)


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    total_memory_gib: float
    capability: tuple[int, int]
    bf16_supported: bool


@dataclass(frozen=True)
class LocalEnvironment:
    python_version: str
    torch_version: str
    cuda_runtime: str | None
    cuda_visible_devices: str
    driver_version: str | None
    cuda_available: bool
    gpus: tuple[GPUInfo, ...]
    nccl_available: bool
    nccl_version: str | None
    vllm_version: str | None


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    versions = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return ", ".join(sorted(versions)) or None


def collect_local_environment() -> LocalEnvironment:
    """Collect the versions and GPU properties required by the local setup."""
    cuda_available = torch.cuda.is_available()
    gpus: list[GPUInfo] = []
    if cuda_available:
        original_device = torch.cuda.current_device()
        try:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                torch.cuda.set_device(index)
                gpus.append(
                    GPUInfo(
                        index=index,
                        name=properties.name,
                        total_memory_gib=properties.total_memory / 2**30,
                        capability=(properties.major, properties.minor),
                        bf16_supported=torch.cuda.is_bf16_supported(),
                    )
                )
        finally:
            torch.cuda.set_device(original_device)

    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_version = None

    nccl_version = None
    if cuda_available and torch.distributed.is_nccl_available():
        try:
            raw_nccl_version = torch.cuda.nccl.version()
            if isinstance(raw_nccl_version, tuple):
                nccl_version = ".".join(str(part) for part in raw_nccl_version)
            else:
                nccl_version = str(raw_nccl_version)
        except (AttributeError, RuntimeError):
            nccl_version = "available (version query failed)"

    return LocalEnvironment(
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_runtime=torch.version.cuda,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        driver_version=_driver_version(),
        cuda_available=cuda_available,
        gpus=tuple(gpus),
        nccl_available=torch.distributed.is_nccl_available(),
        nccl_version=nccl_version,
        vllm_version=vllm_version,
    )


def print_local_environment(environment: LocalEnvironment) -> None:
    """Print a compact, human-readable local environment report."""
    print("=== CS336 AutoDL/local environment check ===")
    print(f"Python version: {environment.python_version}")
    print(f"torch version: {environment.torch_version}")
    print(f"CUDA runtime: {environment.cuda_runtime or '<unavailable>'}")
    print(f"NVIDIA driver: {environment.driver_version or '<unavailable>'}")
    print(f"CUDA_VISIBLE_DEVICES: {environment.cuda_visible_devices}")
    print(f"CUDA available: {environment.cuda_available}")
    print(f"Visible GPU count: {len(environment.gpus)}")
    for gpu in environment.gpus:
        print(
            f"GPU {gpu.index}: {gpu.name}; memory={gpu.total_memory_gib:.2f} GiB; "
            f"capability={gpu.capability[0]}.{gpu.capability[1]}; "
            f"BF16 support={gpu.bf16_supported}"
        )
    print(
        f"NCCL availability: {environment.nccl_available}; "
        f"version={environment.nccl_version or '<unavailable>'}"
    )
    print(f"vLLM version: {environment.vllm_version or '<not installed>'}")


def validate_local_environment(
    environment: LocalEnvironment,
    *,
    require_rtx_5090: bool = False,
) -> list[str]:
    """Return local deployment mismatches; an empty list means ready."""
    issues: list[str] = []
    if sys.version_info[:2] != EXPECTED_PYTHON:
        issues.append(f"Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]} is required.")
    if not environment.torch_version.startswith(EXPECTED_TORCH_PREFIX):
        issues.append(
            f"Expected torch {EXPECTED_TORCH_PREFIX}x, got {environment.torch_version}."
        )
    if not (environment.cuda_runtime or "").startswith(EXPECTED_CUDA_PREFIX):
        issues.append(
            f"Expected CUDA runtime {EXPECTED_CUDA_PREFIX}.x, got {environment.cuda_runtime}."
        )
    if environment.driver_version is None:
        issues.append("Could not query the NVIDIA driver version with nvidia-smi.")
    else:
        driver_parts = environment.driver_version.split(",", maxsplit=1)[0].split(".")
        try:
            driver_tuple = tuple(int(part) for part in driver_parts[:3])
            driver_tuple += (0,) * (3 - len(driver_tuple))
        except ValueError:
            issues.append(
                f"Could not parse NVIDIA driver version {environment.driver_version!r}."
            )
        else:
            if driver_tuple < MINIMUM_CUDA_129_DRIVER:
                issues.append(
                    "CUDA 12.9 requires NVIDIA Linux driver 575.51.03 or newer."
                )
    if environment.vllm_version != EXPECTED_VLLM:
        issues.append(f"Expected vLLM {EXPECTED_VLLM}, got {environment.vllm_version}.")
    if len(environment.gpus) != 2:
        issues.append(
            f"Exactly 2 visible GPUs are required, got {len(environment.gpus)}."
        )
    if environment.cuda_visible_devices != "0,1":
        issues.append(
            "CUDA_VISIBLE_DEVICES must be exactly '0,1' so policy and vLLM mappings are deterministic."
        )
    if any(not gpu.bf16_supported for gpu in environment.gpus):
        issues.append("Every visible GPU must support BF16.")
    if not environment.nccl_available:
        issues.append("PyTorch NCCL backend is unavailable.")
    if require_rtx_5090 and any("RTX 5090" not in gpu.name for gpu in environment.gpus):
        issues.append("Both visible GPUs must be RTX 5090 devices.")
    return issues


def visible_memory_used_mib() -> tuple[float, ...]:
    """Return total driver-visible memory use for each logical CUDA device."""
    values = []
    for index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        values.append((total_bytes - free_bytes) / 2**20)
    return tuple(values)
