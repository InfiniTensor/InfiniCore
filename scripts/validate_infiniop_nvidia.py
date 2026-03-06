#!/usr/bin/env python3
"""Validate infiniop C API outputs against PyTorch on NVIDIA CUDA.

Ops covered:
- erf
- erfc
- erfinv
- matrix_power
- pixel_shuffle

Usage:
    python scripts/validate_infiniop_nvidia.py
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch


INFINI_DEVICE_NVIDIA = 1
INFINI_STATUS_BAD_TENSOR_STRIDES = 12
INFINI_DTYPE_F16 = 12
INFINI_DTYPE_F32 = 13
INFINI_DTYPE_BF16 = 19

DTYPE_CASES: Tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16, torch.float32)
DTYPE_NAME = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}
DTYPE_TO_INFINI = {
    torch.float16: INFINI_DTYPE_F16,
    torch.bfloat16: INFINI_DTYPE_BF16,
    torch.float32: INFINI_DTYPE_F32,
}

TOL_ERF_ERFC_MATRIX: Dict[torch.dtype, Tuple[float, float]] = {
    torch.float16: (1e-3, 1e-2),
    torch.bfloat16: (1e-2, 5e-2),
    torch.float32: (1e-5, 1e-4),
}
TOL_ERFINV: Dict[torch.dtype, Tuple[float, float]] = {
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1e-2, 5e-2),
    torch.float32: (1e-5, 1e-4),
}
TOL_PIXEL_SHUFFLE: Dict[torch.dtype, Tuple[float, float]] = {
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1e-2, 5e-2),
    torch.float32: (1e-5, 1e-4),
}


def _status_ok(status: int, ctx: str) -> None:
    if status != 0:
        raise RuntimeError(f"{ctx} failed with status={status}")


def _storage_size_for_strided(shape: Sequence[int], stride: Sequence[int]) -> int:
    if len(shape) != len(stride):
        raise ValueError(f"shape/stride rank mismatch: {shape} vs {stride}")
    max_offset = 0
    for dim, st in zip(shape, stride):
        if st < 0:
            raise ValueError(f"negative stride not supported: {stride}")
        if dim <= 0:
            raise ValueError(f"invalid shape dim in {shape}")
        max_offset += (dim - 1) * st
    return max_offset + 1


def _make_strided_view_from_contiguous(src: torch.Tensor, stride: Sequence[int]) -> torch.Tensor:
    storage_size = _storage_size_for_strided(src.shape, stride)
    base = torch.zeros(storage_size, dtype=src.dtype, device=src.device)
    view = torch.as_strided(base, size=tuple(src.shape), stride=tuple(stride))
    view.copy_(src)
    return view


def _make_empty_output(shape: Sequence[int], dtype: torch.dtype, stride: Optional[Sequence[int]]) -> torch.Tensor:
    if stride is None:
        return torch.empty(tuple(shape), dtype=dtype, device="cuda")
    storage_size = _storage_size_for_strided(shape, stride)
    base = torch.empty(storage_size, dtype=dtype, device="cuda")
    return torch.as_strided(base, size=tuple(shape), stride=tuple(stride))


def _random_tensor(
    shape: Sequence[int],
    dtype: torch.dtype,
    stride: Optional[Sequence[int]] = None,
    low: float = -1.0,
    high: float = 1.0,
    clamp: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    src = torch.empty(tuple(shape), dtype=torch.float32, device="cuda").uniform_(low, high)
    if clamp is not None:
        src = src.clamp(min=clamp[0], max=clamp[1])
    src = src.to(dtype=dtype)
    if stride is None:
        return src.contiguous()
    return _make_strided_view_from_contiguous(src, stride)


def _to_compare_dtype(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.bfloat16:
        return x.to(dtype=torch.float32)
    return x


def _compare(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> Tuple[bool, float, float]:
    a = _to_compare_dtype(actual)
    b = _to_compare_dtype(expected)
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    if ok:
        return True, 0.0, 0.0
    diff = (a - b).abs()
    max_abs = float(diff.max().item())
    denom = b.abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max().item())
    return False, max_abs, max_rel


class _InfiniLib:
    def __init__(self, librt: ctypes.CDLL, libop: ctypes.CDLL):
        self._librt = librt
        self._libop = libop

    def __getattr__(self, name: str):
        if hasattr(self._libop, name):
            return getattr(self._libop, name)
        if hasattr(self._librt, name):
            return getattr(self._librt, name)
        raise AttributeError(name)


def _platform_lib_names() -> Tuple[str, str]:
    system = platform.system()
    if system == "Windows":
        return "infiniop.dll", "infinirt.dll"
    if system == "Darwin":
        return "libinfiniop.dylib", "libinfinirt.dylib"
    return "libinfiniop.so", "libinfinirt.so"


def _parse_paths_from_text(text: str, marker: str) -> List[Path]:
    pattern = rf"([~\w\-./\\: ]+{re.escape(marker)})"
    out: List[Path] = []
    for raw in re.findall(pattern, text):
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.exists():
            out.append(p)
    return out


def _search_name(root: Path, filename: str) -> List[Path]:
    if not root.exists():
        return []
    paths: List[Path] = []
    for p in root.rglob(filename):
        if p.is_file():
            paths.append(p.resolve())
    return paths


def _discover_libraries() -> Tuple[Path, Path]:
    op_name, rt_name = _platform_lib_names()

    op_candidates: List[Path] = []
    rt_candidates: List[Path] = []

    def add_if_exists(dst: List[Path], maybe: Optional[str]) -> None:
        if not maybe:
            return
        p = Path(maybe).expanduser().resolve()
        if p.exists() and p.is_file():
            dst.append(p)

    add_if_exists(op_candidates, os.getenv("INFINIOP_LIB"))
    add_if_exists(rt_candidates, os.getenv("INFINIRT_LIB"))

    infini_root = Path(os.getenv("INFINI_ROOT", str(Path.home() / ".infini"))).expanduser()
    add_if_exists(op_candidates, str(infini_root / "lib" / op_name))
    add_if_exists(rt_candidates, str(infini_root / "lib" / rt_name))

    xmake_cmds = [
        ["xmake", "show", "-t", "target", "infiniop"],
        ["xmake", "show", "-t", "target"],
    ]
    for cmd in xmake_cmds:
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        except FileNotFoundError:
            break
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        op_candidates.extend(_parse_paths_from_text(text, op_name))
        rt_candidates.extend(_parse_paths_from_text(text, rt_name))

    search_roots = [
        Path.cwd() / "build",
        Path.cwd() / "xmake-build",
        Path.cwd() / "out",
        Path.cwd(),
        infini_root / "lib",
    ]
    for root in search_roots:
        op_candidates.extend(_search_name(root, op_name))
        rt_candidates.extend(_search_name(root, rt_name))

    # Dedupe while preserving order
    def uniq(paths: List[Path]) -> List[Path]:
        seen = set()
        out = []
        for p in paths:
            s = str(p)
            if s in seen:
                continue
            seen.add(s)
            out.append(p)
        return out

    op_candidates = uniq(op_candidates)
    rt_candidates = uniq(rt_candidates)

    # Try exact directory pairing first.
    rt_by_dir = {p.parent: p for p in rt_candidates}
    for op in op_candidates:
        if op.parent in rt_by_dir:
            return op, rt_by_dir[op.parent]

    # Fallback: if one side found, infer sibling in same dir.
    for op in op_candidates:
        sibling = op.parent / rt_name
        if sibling.exists():
            return op, sibling.resolve()
    for rt in rt_candidates:
        sibling = rt.parent / op_name
        if sibling.exists():
            return sibling.resolve(), rt

    raise FileNotFoundError(
        "Could not locate infiniop shared libraries. "
        f"Need both {op_name} and {rt_name}. "
        "Set INFINIOP_LIB/INFINIRT_LIB or build/install first."
    )


def _load_api() -> _InfiniLib:
    op_path, rt_path = _discover_libraries()
    rtld_global = getattr(ctypes, "RTLD_GLOBAL", 0)
    librt = ctypes.CDLL(str(rt_path), mode=rtld_global)
    libop = ctypes.CDLL(str(op_path), mode=rtld_global)
    api = _InfiniLib(librt, libop)

    c_void_p_p = ctypes.POINTER(ctypes.c_void_p)

    api.infiniopCreateHandle.argtypes = [c_void_p_p]
    api.infiniopCreateHandle.restype = ctypes.c_int
    api.infiniopDestroyHandle.argtypes = [ctypes.c_void_p]
    api.infiniopDestroyHandle.restype = ctypes.c_int

    api.infinirtSetDevice.argtypes = [ctypes.c_int, ctypes.c_int]
    api.infinirtSetDevice.restype = ctypes.c_int

    api.infiniopCreateTensorDescriptor.argtypes = [
        c_void_p_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_ssize_t),
        ctypes.c_int,
    ]
    api.infiniopCreateTensorDescriptor.restype = ctypes.c_int
    api.infiniopDestroyTensorDescriptor.argtypes = [ctypes.c_void_p]
    api.infiniopDestroyTensorDescriptor.restype = ctypes.c_int

    return api


def _create_handle(api: _InfiniLib) -> ctypes.c_void_p:
    _status_ok(api.infinirtSetDevice(INFINI_DEVICE_NVIDIA, int(torch.cuda.current_device())), "infinirtSetDevice")
    handle = ctypes.c_void_p()
    _status_ok(api.infiniopCreateHandle(ctypes.byref(handle)), "infiniopCreateHandle")
    return handle


def _create_tensor_desc(api: _InfiniLib, t: torch.Tensor) -> ctypes.c_void_p:
    if t.dtype not in DTYPE_TO_INFINI:
        raise TypeError(f"Unsupported dtype for infiniop: {t.dtype}")
    if t.device.type != "cuda":
        raise TypeError(f"Tensor must be CUDA, got {t.device}")

    ndim = t.dim()
    shape_arr = (ctypes.c_size_t * ndim)(*map(int, t.shape))
    stride_arr = (ctypes.c_ssize_t * ndim)(*map(int, t.stride()))

    desc = ctypes.c_void_p()
    _status_ok(
        api.infiniopCreateTensorDescriptor(
            ctypes.byref(desc),
            ctypes.c_size_t(ndim),
            shape_arr,
            stride_arr,
            ctypes.c_int(DTYPE_TO_INFINI[t.dtype]),
        ),
        "infiniopCreateTensorDescriptor",
    )
    return desc


def _create_tensor_desc_from_spec(
    api: _InfiniLib,
    shape: Sequence[int],
    stride: Sequence[int],
    dtype: int,
) -> ctypes.c_void_p:
    if len(shape) != len(stride):
        raise ValueError(f"shape/stride rank mismatch: {shape} vs {stride}")

    ndim = len(shape)
    shape_arr = (ctypes.c_size_t * ndim)(*map(int, shape))
    stride_arr = (ctypes.c_ssize_t * ndim)(*map(int, stride))
    desc = ctypes.c_void_p()
    _status_ok(
        api.infiniopCreateTensorDescriptor(
            ctypes.byref(desc),
            ctypes.c_size_t(ndim),
            shape_arr,
            stride_arr,
            ctypes.c_int(dtype),
        ),
        f"infiniopCreateTensorDescriptor shape={tuple(shape)} stride={tuple(stride)}",
    )
    return desc


def _expect_descriptor_reject(
    create_call: Callable[[ctypes.POINTER(ctypes.c_void_p)], int],
    destroy_fn: Callable,
    op_name: str,
    case_name: str,
) -> Optional[str]:
    op_desc = ctypes.c_void_p()
    status = create_call(ctypes.byref(op_desc))
    if status == 0:
        if op_desc:
            destroy_status = destroy_fn(op_desc)
            if destroy_status != 0:
                return (
                    f"{op_name}/{case_name}: unexpected success and destroy status={destroy_status}; "
                    f"expected status={INFINI_STATUS_BAD_TENSOR_STRIDES}"
                )
        return (
            f"{op_name}/{case_name}: unexpected success; "
            f"expected status={INFINI_STATUS_BAD_TENSOR_STRIDES}"
        )
    if status != INFINI_STATUS_BAD_TENSOR_STRIDES:
        return (
            f"{op_name}/{case_name}: rejected with wrong status={status}; "
            f"expected status={INFINI_STATUS_BAD_TENSOR_STRIDES}"
        )
    return None


def _run_negative_descriptor_tests(
    api: _InfiniLib,
    handle: ctypes.c_void_p,
) -> List[str]:
    failures: List[str] = []
    c_void_p_p = ctypes.POINTER(ctypes.c_void_p)

    if hasattr(api, "infinirtSetDevice"):
        dev = int(torch.cuda.current_device())
        status = api.infinirtSetDevice(INFINI_DEVICE_NVIDIA, dev)
        if status != 0:
            failures.append(f"infinirtSetDevice(NVIDIA,{dev}) failed with status={status}")
            return failures

    try:
        api.infiniopCreatePixelShuffleDescriptor.argtypes = [
            ctypes.c_void_p,
            c_void_p_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        api.infiniopCreatePixelShuffleDescriptor.restype = ctypes.c_int
        api.infiniopDestroyPixelShuffleDescriptor.argtypes = [ctypes.c_void_p]
        api.infiniopDestroyPixelShuffleDescriptor.restype = ctypes.c_int

        api.infiniopCreateMatrixPowerDescriptor.argtypes = [
            ctypes.c_void_p,
            c_void_p_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        api.infiniopCreateMatrixPowerDescriptor.restype = ctypes.c_int
        api.infiniopDestroyMatrixPowerDescriptor.argtypes = [ctypes.c_void_p]
        api.infiniopDestroyMatrixPowerDescriptor.restype = ctypes.c_int
    except AttributeError as exc:
        failures.append(f"negative tests missing symbol: {exc}")
        return failures

    pixel_x_desc = ctypes.c_void_p()
    pixel_y_desc = ctypes.c_void_p()
    matrix_x_desc = ctypes.c_void_p()
    matrix_y_desc = ctypes.c_void_p()
    matrix_valid_desc = ctypes.c_void_p()

    try:
        pixel_in_shape = (1, 4, 2, 2)
        pixel_in_stride = (0, 0, 2, 1)
        pixel_factor = 2
        pixel_out_shape = _pixel_shuffle_output_shape(pixel_in_shape, pixel_factor)
        pixel_out_stride = (16, 16, 4, 1)

        pixel_x_desc = _create_tensor_desc_from_spec(api, pixel_in_shape, pixel_in_stride, INFINI_DTYPE_F32)
        pixel_y_desc = _create_tensor_desc_from_spec(api, pixel_out_shape, pixel_out_stride, INFINI_DTYPE_F32)

        err = _expect_descriptor_reject(
            lambda out_desc: api.infiniopCreatePixelShuffleDescriptor(
                handle, out_desc, pixel_y_desc, pixel_x_desc, int(pixel_factor)
            ),
            api.infiniopDestroyPixelShuffleDescriptor,
            "pixel_shuffle",
            "broadcasted input channel stride",
        )
        if err is not None:
            failures.append(err)

        matrix_shape = (2, 2)
        matrix_valid_stride = (2, 1)
        matrix_invalid_strides = [(0, 1), (2, 0)]

        matrix_valid_desc = _create_tensor_desc_from_spec(api, matrix_shape, matrix_valid_stride, INFINI_DTYPE_F32)
        for invalid_stride in matrix_invalid_strides:
            matrix_x_desc = _create_tensor_desc_from_spec(api, matrix_shape, invalid_stride, INFINI_DTYPE_F32)
            err = _expect_descriptor_reject(
                lambda out_desc: api.infiniopCreateMatrixPowerDescriptor(
                    handle, out_desc, matrix_valid_desc, matrix_x_desc, int(3)
                ),
                api.infiniopDestroyMatrixPowerDescriptor,
                "matrix_power",
                f"x_stride={invalid_stride}",
            )
            if err is not None:
                failures.append(err)
            if matrix_x_desc:
                destroy_status = api.infiniopDestroyTensorDescriptor(matrix_x_desc)
                if destroy_status != 0:
                    failures.append(
                        f"matrix_power/x_stride={invalid_stride}: destroy x descriptor status={destroy_status}"
                    )
                matrix_x_desc = ctypes.c_void_p()

        for invalid_stride in matrix_invalid_strides:
            matrix_y_desc = _create_tensor_desc_from_spec(api, matrix_shape, invalid_stride, INFINI_DTYPE_F32)
            err = _expect_descriptor_reject(
                lambda out_desc: api.infiniopCreateMatrixPowerDescriptor(
                    handle, out_desc, matrix_y_desc, matrix_valid_desc, int(3)
                ),
                api.infiniopDestroyMatrixPowerDescriptor,
                "matrix_power",
                f"y_stride={invalid_stride}",
            )
            if err is not None:
                failures.append(err)
            if matrix_y_desc:
                destroy_status = api.infiniopDestroyTensorDescriptor(matrix_y_desc)
                if destroy_status != 0:
                    failures.append(
                        f"matrix_power/y_stride={invalid_stride}: destroy y descriptor status={destroy_status}"
                    )
                matrix_y_desc = ctypes.c_void_p()

    except Exception as exc:
        failures.append(f"negative tests error: {exc}")
    finally:
        if pixel_x_desc:
            status = api.infiniopDestroyTensorDescriptor(pixel_x_desc)
            if status != 0:
                failures.append(f"pixel_shuffle: destroy x descriptor status={status}")
        if pixel_y_desc:
            status = api.infiniopDestroyTensorDescriptor(pixel_y_desc)
            if status != 0:
                failures.append(f"pixel_shuffle: destroy y descriptor status={status}")
        if matrix_x_desc:
            status = api.infiniopDestroyTensorDescriptor(matrix_x_desc)
            if status != 0:
                failures.append(f"matrix_power: destroy x descriptor status={status}")
        if matrix_y_desc:
            status = api.infiniopDestroyTensorDescriptor(matrix_y_desc)
            if status != 0:
                failures.append(f"matrix_power: destroy y descriptor status={status}")
        if matrix_valid_desc:
            status = api.infiniopDestroyTensorDescriptor(matrix_valid_desc)
            if status != 0:
                failures.append(f"matrix_power: destroy valid descriptor status={status}")

    return failures


def _run_unary_case(
    api: _InfiniLib,
    handle: ctypes.c_void_p,
    create_fn: Callable,
    get_ws_fn: Callable,
    run_fn: Callable,
    destroy_fn: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
) -> None:
    x_desc = ctypes.c_void_p()
    y_desc = ctypes.c_void_p()
    op_desc = ctypes.c_void_p()
    workspace = None
    try:
        x_desc = _create_tensor_desc(api, x)
        y_desc = _create_tensor_desc(api, y)
        _status_ok(create_fn(handle, ctypes.byref(op_desc), y_desc, x_desc), "create descriptor")

        ws_size = ctypes.c_size_t(0)
        _status_ok(get_ws_fn(op_desc, ctypes.byref(ws_size)), "get workspace size")

        ws_ptr = ctypes.c_void_p()
        if ws_size.value > 0:
            workspace = torch.empty(ws_size.value, dtype=torch.uint8, device="cuda")
            ws_ptr = ctypes.c_void_p(workspace.data_ptr())

        stream_ptr = ctypes.c_void_p(int(torch.cuda.current_stream().cuda_stream))
        _status_ok(
            run_fn(
                op_desc,
                ws_ptr,
                ctypes.c_size_t(ws_size.value),
                ctypes.c_void_p(y.data_ptr()),
                ctypes.c_void_p(x.data_ptr()),
                stream_ptr,
            ),
            "execute operator",
        )
        torch.cuda.synchronize()
    finally:
        if op_desc:
            api_call = destroy_fn(op_desc)
            if api_call != 0:
                print(f"WARN: destroy op descriptor status={api_call}", file=sys.stderr)
        if x_desc:
            api_call = api.infiniopDestroyTensorDescriptor(x_desc)
            if api_call != 0:
                print(f"WARN: destroy x descriptor status={api_call}", file=sys.stderr)
        if y_desc:
            api_call = api.infiniopDestroyTensorDescriptor(y_desc)
            if api_call != 0:
                print(f"WARN: destroy y descriptor status={api_call}", file=sys.stderr)


def _run_matrix_power_case(
    api: _InfiniLib,
    handle: ctypes.c_void_p,
    x: torch.Tensor,
    y: torch.Tensor,
    n: int,
) -> None:
    x_desc = ctypes.c_void_p()
    y_desc = ctypes.c_void_p()
    op_desc = ctypes.c_void_p()
    workspace = None
    try:
        x_desc = _create_tensor_desc(api, x)
        y_desc = _create_tensor_desc(api, y)
        _status_ok(
            api.infiniopCreateMatrixPowerDescriptor(handle, ctypes.byref(op_desc), y_desc, x_desc, int(n)),
            "create matrix_power descriptor",
        )

        ws_size = ctypes.c_size_t(0)
        _status_ok(api.infiniopGetMatrixPowerWorkspaceSize(op_desc, ctypes.byref(ws_size)), "get matrix_power workspace")

        ws_ptr = ctypes.c_void_p()
        if ws_size.value > 0:
            workspace = torch.empty(ws_size.value, dtype=torch.uint8, device="cuda")
            ws_ptr = ctypes.c_void_p(workspace.data_ptr())

        stream_ptr = ctypes.c_void_p(int(torch.cuda.current_stream().cuda_stream))
        _status_ok(
            api.infiniopMatrixPower(
                op_desc,
                ws_ptr,
                ctypes.c_size_t(ws_size.value),
                ctypes.c_void_p(y.data_ptr()),
                ctypes.c_void_p(x.data_ptr()),
                stream_ptr,
            ),
            "execute matrix_power",
        )
        torch.cuda.synchronize()
    finally:
        if op_desc:
            api_call = api.infiniopDestroyMatrixPowerDescriptor(op_desc)
            if api_call != 0:
                print(f"WARN: destroy matrix_power descriptor status={api_call}", file=sys.stderr)
        if x_desc:
            api_call = api.infiniopDestroyTensorDescriptor(x_desc)
            if api_call != 0:
                print(f"WARN: destroy x descriptor status={api_call}", file=sys.stderr)
        if y_desc:
            api_call = api.infiniopDestroyTensorDescriptor(y_desc)
            if api_call != 0:
                print(f"WARN: destroy y descriptor status={api_call}", file=sys.stderr)


def _run_pixel_shuffle_case(
    api: _InfiniLib,
    handle: ctypes.c_void_p,
    x: torch.Tensor,
    y: torch.Tensor,
    upscale_factor: int,
) -> None:
    x_desc = ctypes.c_void_p()
    y_desc = ctypes.c_void_p()
    op_desc = ctypes.c_void_p()
    workspace = None
    try:
        x_desc = _create_tensor_desc(api, x)
        y_desc = _create_tensor_desc(api, y)
        _status_ok(
            api.infiniopCreatePixelShuffleDescriptor(handle, ctypes.byref(op_desc), y_desc, x_desc, int(upscale_factor)),
            "create pixel_shuffle descriptor",
        )

        ws_size = ctypes.c_size_t(0)
        _status_ok(api.infiniopGetPixelShuffleWorkspaceSize(op_desc, ctypes.byref(ws_size)), "get pixel_shuffle workspace")

        ws_ptr = ctypes.c_void_p()
        if ws_size.value > 0:
            workspace = torch.empty(ws_size.value, dtype=torch.uint8, device="cuda")
            ws_ptr = ctypes.c_void_p(workspace.data_ptr())

        stream_ptr = ctypes.c_void_p(int(torch.cuda.current_stream().cuda_stream))
        _status_ok(
            api.infiniopPixelShuffle(
                op_desc,
                ws_ptr,
                ctypes.c_size_t(ws_size.value),
                ctypes.c_void_p(y.data_ptr()),
                ctypes.c_void_p(x.data_ptr()),
                stream_ptr,
            ),
            "execute pixel_shuffle",
        )
        torch.cuda.synchronize()
    finally:
        if op_desc:
            api_call = api.infiniopDestroyPixelShuffleDescriptor(op_desc)
            if api_call != 0:
                print(f"WARN: destroy pixel_shuffle descriptor status={api_call}", file=sys.stderr)
        if x_desc:
            api_call = api.infiniopDestroyTensorDescriptor(x_desc)
            if api_call != 0:
                print(f"WARN: destroy x descriptor status={api_call}", file=sys.stderr)
        if y_desc:
            api_call = api.infiniopDestroyTensorDescriptor(y_desc)
            if api_call != 0:
                print(f"WARN: destroy y descriptor status={api_call}", file=sys.stderr)


def _run_unary_op(
    api: _InfiniLib,
    handle: ctypes.c_void_p,
    op_name: str,
    torch_fn: Callable[[torch.Tensor], torch.Tensor],
    tol_map: Dict[torch.dtype, Tuple[float, float]],
    input_cases: Sequence[Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]],
    clamp: Optional[Tuple[float, float]] = None,
) -> Tuple[int, int, List[str]]:
    failures: List[str] = []
    total = 0

    c_void_p_p = ctypes.POINTER(ctypes.c_void_p)
    try:
        create_fn = getattr(api, f"infiniopCreate{op_name}Descriptor")
        get_ws_fn = getattr(api, f"infiniopGet{op_name}WorkspaceSize")
        run_fn = getattr(api, f"infiniop{op_name}")
        destroy_fn = getattr(api, f"infiniopDestroy{op_name}Descriptor")
    except AttributeError as exc:
        return 0, 0, [f"missing symbol: {exc}"]

    create_fn.argtypes = [ctypes.c_void_p, c_void_p_p, ctypes.c_void_p, ctypes.c_void_p]
    create_fn.restype = ctypes.c_int
    get_ws_fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    get_ws_fn.restype = ctypes.c_int
    run_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    run_fn.restype = ctypes.c_int
    destroy_fn.argtypes = [ctypes.c_void_p]
    destroy_fn.restype = ctypes.c_int

    for dtype in DTYPE_CASES:
        atol, rtol = tol_map[dtype]
        for shape, in_stride in input_cases:
            total += 1
            x = _random_tensor(shape, dtype=dtype, stride=in_stride, low=-0.95, high=0.95, clamp=clamp)
            y = _make_empty_output(shape, dtype=dtype, stride=None)
            try:
                _run_unary_case(api, handle, create_fn, get_ws_fn, run_fn, destroy_fn, x, y)
                expected = torch_fn(x)
                ok, max_abs, max_rel = _compare(y, expected, atol=atol, rtol=rtol)
                if not ok:
                    failures.append(
                        f"dtype={DTYPE_NAME[dtype]} shape={shape} in_stride={in_stride} "
                        f"max_abs={max_abs:.4e} max_rel={max_rel:.4e} tol(atol={atol},rtol={rtol})"
                    )
            except Exception as exc:
                failures.append(
                    f"dtype={DTYPE_NAME[dtype]} shape={shape} in_stride={in_stride} error={exc}"
                )

    return total, total - len(failures), failures


def _run_matrix_power(
    api: _InfiniLib,
    handle: ctypes.c_void_p,
) -> Tuple[int, int, List[str]]:
    c_void_p_p = ctypes.POINTER(ctypes.c_void_p)
    try:
        api.infiniopCreateMatrixPowerDescriptor.argtypes = [
            ctypes.c_void_p,
            c_void_p_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        api.infiniopCreateMatrixPowerDescriptor.restype = ctypes.c_int
        api.infiniopGetMatrixPowerWorkspaceSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        api.infiniopGetMatrixPowerWorkspaceSize.restype = ctypes.c_int
        api.infiniopMatrixPower.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        api.infiniopMatrixPower.restype = ctypes.c_int
        api.infiniopDestroyMatrixPowerDescriptor.argtypes = [ctypes.c_void_p]
        api.infiniopDestroyMatrixPowerDescriptor.restype = ctypes.c_int
    except AttributeError as exc:
        return 0, 0, [f"missing symbol: {exc}"]

    cases = [
        {"shape": (3, 3), "n": 2, "in_stride": None, "out_stride": None},
        {"shape": (6, 6), "n": 0, "in_stride": None, "out_stride": None},
        # Official-style strided input
        {"shape": (4, 4), "n": 3, "in_stride": (256, 64), "out_stride": None},
        # Strided output validation
        {"shape": (4, 4), "n": 3, "in_stride": None, "out_stride": (256, 64)},
    ]

    failures: List[str] = []
    total = 0

    for dtype in DTYPE_CASES:
        atol, rtol = TOL_ERF_ERFC_MATRIX[dtype]
        for case in cases:
            shape = case["shape"]
            n = case["n"]
            in_stride = case["in_stride"]
            out_stride = case["out_stride"]
            total += 1

            x = _random_tensor(shape, dtype=dtype, stride=in_stride, low=-0.8, high=0.8)
            y = _make_empty_output(shape, dtype=dtype, stride=out_stride)
            try:
                _run_matrix_power_case(api, handle, x, y, n)
                expected = torch.matrix_power(x, n)
                ok, max_abs, max_rel = _compare(y, expected, atol=atol, rtol=rtol)
                if not ok:
                    failures.append(
                        f"dtype={DTYPE_NAME[dtype]} shape={shape} n={n} in_stride={in_stride} out_stride={out_stride} "
                        f"max_abs={max_abs:.4e} max_rel={max_rel:.4e} tol(atol={atol},rtol={rtol})"
                    )
            except Exception as exc:
                failures.append(
                    f"dtype={DTYPE_NAME[dtype]} shape={shape} n={n} in_stride={in_stride} out_stride={out_stride} error={exc}"
                )

    return total, total - len(failures), failures


def _pixel_shuffle_output_shape(shape: Sequence[int], factor: int) -> Tuple[int, ...]:
    n, c, h, w = shape
    oc = c // (factor * factor)
    return (n, oc, h * factor, w * factor)


def _run_pixel_shuffle(
    api: _InfiniLib,
    handle: ctypes.c_void_p,
) -> Tuple[int, int, List[str]]:
    c_void_p_p = ctypes.POINTER(ctypes.c_void_p)
    try:
        api.infiniopCreatePixelShuffleDescriptor.argtypes = [
            ctypes.c_void_p,
            c_void_p_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        api.infiniopCreatePixelShuffleDescriptor.restype = ctypes.c_int
        api.infiniopGetPixelShuffleWorkspaceSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        api.infiniopGetPixelShuffleWorkspaceSize.restype = ctypes.c_int
        api.infiniopPixelShuffle.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        api.infiniopPixelShuffle.restype = ctypes.c_int
        api.infiniopDestroyPixelShuffleDescriptor.argtypes = [ctypes.c_void_p]
        api.infiniopDestroyPixelShuffleDescriptor.restype = ctypes.c_int
    except AttributeError as exc:
        return 0, 0, [f"missing symbol: {exc}"]

    cases = [
        {"shape": (1, 4, 8, 8), "factor": 2, "in_stride": None, "out_stride": None},
        # Official-style strided input
        {"shape": (2, 9, 4, 4), "factor": 3, "in_stride": (288, 144, 36, 9), "out_stride": None},
        # Strided output validation
        {"shape": (2, 9, 4, 4), "factor": 3, "in_stride": None, "out_stride": (500, 500, 20, 1)},
    ]

    failures: List[str] = []
    total = 0

    for dtype in DTYPE_CASES:
        atol, rtol = TOL_PIXEL_SHUFFLE[dtype]
        for case in cases:
            shape = case["shape"]
            factor = case["factor"]
            in_stride = case["in_stride"]
            out_stride = case["out_stride"]
            out_shape = _pixel_shuffle_output_shape(shape, factor)
            total += 1

            x = _random_tensor(shape, dtype=dtype, stride=in_stride, low=-1.0, high=1.0)
            y = _make_empty_output(out_shape, dtype=dtype, stride=out_stride)
            try:
                _run_pixel_shuffle_case(api, handle, x, y, factor)
                expected = torch.nn.functional.pixel_shuffle(x, factor)
                ok, max_abs, max_rel = _compare(y, expected, atol=atol, rtol=rtol)
                if not ok:
                    failures.append(
                        f"dtype={DTYPE_NAME[dtype]} shape={shape} factor={factor} in_stride={in_stride} out_stride={out_stride} "
                        f"max_abs={max_abs:.4e} max_rel={max_rel:.4e} tol(atol={atol},rtol={rtol})"
                    )
            except Exception as exc:
                failures.append(
                    f"dtype={DTYPE_NAME[dtype]} shape={shape} factor={factor} in_stride={in_stride} out_stride={out_stride} error={exc}"
                )

    return total, total - len(failures), failures


def main() -> int:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if not torch.cuda.is_available():
        print("erf: FAIL (CUDA not available)")
        print("erfc: FAIL (CUDA not available)")
        print("erfinv: FAIL (CUDA not available)")
        print("matrix_power: FAIL (CUDA not available)")
        print("pixel_shuffle: FAIL (CUDA not available)")
        return 1

    try:
        api = _load_api()
    except Exception as exc:
        print(f"erf: FAIL (library load error: {exc})")
        print(f"erfc: FAIL (library load error: {exc})")
        print(f"erfinv: FAIL (library load error: {exc})")
        print(f"matrix_power: FAIL (library load error: {exc})")
        print(f"pixel_shuffle: FAIL (library load error: {exc})")
        return 1

    handle = ctypes.c_void_p()
    summary: List[Tuple[str, int, int, List[str]]] = []
    negative_failures: List[str] = []

    try:
        handle = _create_handle(api)

        unary_cases = [
            ((13, 4), None),
            ((13, 4), (10, 1)),
            ((8, 16), None),
            ((8, 16), (40, 1)),
            ((2, 3, 4), None),
        ]

        summary.append(
            ("erf",) + _run_unary_op(api, handle, "Erf", torch.erf, TOL_ERF_ERFC_MATRIX, unary_cases)
        )
        summary.append(
            ("erfc",) + _run_unary_op(api, handle, "Erfc", torch.erfc, TOL_ERF_ERFC_MATRIX, unary_cases)
        )
        summary.append(
            (
                "erfinv",
            )
            + _run_unary_op(
                api,
                handle,
                "Erfinv",
                torch.erfinv,
                TOL_ERFINV,
                unary_cases,
                clamp=(-0.999, 0.999),
            )
        )
        summary.append(("matrix_power",) + _run_matrix_power(api, handle))
        summary.append(("pixel_shuffle",) + _run_pixel_shuffle(api, handle))
        negative_failures = _run_negative_descriptor_tests(api, handle)

    except Exception as exc:
        print(f"fatal: FAIL ({exc})")
        return 1
    finally:
        if handle:
            status = api.infiniopDestroyHandle(handle)
            if status != 0:
                print(f"WARN: infiniopDestroyHandle status={status}", file=sys.stderr)

    any_fail = False
    for op_name, total, passed, failures in summary:
        if not failures:
            print(f"{op_name}: PASS ({passed}/{total})")
            continue

        any_fail = True
        print(f"{op_name}: FAIL ({passed}/{total})")
        for msg in failures:
            print(f"  {msg}")

    if negative_failures:
        any_fail = True
        print("negative tests: FAIL")
        for msg in negative_failures:
            print(f"  {msg}")
    else:
        print("negative tests: PASS")

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
