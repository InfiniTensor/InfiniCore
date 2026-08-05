import ctypes
import json
import os
from ctypes import c_uint64
from pathlib import Path

import torch

from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    TestWorkspace,
    check_error,
    get_args,
    get_test_devices,
    infiniopOperatorDescriptor_t,
    test_operator,
    to_torch_dtype,
)


_TEST_CASES = [
    ((2, 64), (2, 4), (2, 128)),
    ((2, 3, 16), (2, 3, 1), (2, 3, 32)),
    ((1, 3584), (1, 224), (1, 7168)),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

_CHECKPOINT_CASES = [
    (
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.gate_proj.weight_scale",
        0,
        2,
    ),
    (
        "language_model.model.layers.1.mlp.experts.0.down_proj.weight",
        "language_model.model.layers.1.mlp.experts.0.down_proj.weight_scale",
        17,
        2,
    ),
    (
        "language_model.model.layers.0.mlp.down_proj.weight",
        "language_model.model.layers.0.mlp.down_proj.weight_scale",
        31,
        2,
    ),
]


def reference_dequantize(packed: torch.Tensor, scales: torch.Tensor, out_dtype):
    """Independent OCP/Quark raw-layout decoder used as the test oracle."""
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    low = packed & 0x0F
    high = packed >> 4
    codes = torch.stack((low, high), dim=-1).flatten(-2)
    values = magnitudes[(codes & 0x07).to(torch.int64)]
    values = torch.where((codes & 0x08) != 0, -values, values)
    exponents = scales.to(torch.int32).sub(127).repeat_interleave(32, dim=-1)
    return torch.ldexp(values, exponents).to(out_dtype)


def run_dequantize(handle, device, packed_data, scales_data, dtype, sync=None):
    packed = TestTensor.from_torch(packed_data, InfiniDtype.U8, device)
    scales = TestTensor.from_torch(scales_data, InfiniDtype.U8, device)
    out_shape = list(packed_data.shape)
    out_shape[-1] *= 2
    out = TestTensor(out_shape, None, dtype, device, mode="zeros")
    expected = reference_dequantize(
        packed.torch_tensor(), scales.torch_tensor(), to_torch_dtype(dtype)
    )

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateMxfp4DequantizeDescriptor(
            handle,
            ctypes.byref(descriptor),
            out.descriptor,
            packed.descriptor,
            scales.descriptor,
        )
    )
    for tensor in (out, packed, scales):
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetMxfp4DequantizeWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)
    check_error(
        LIBINFINIOP.infiniopMxfp4Dequantize(
            descriptor,
            workspace.data(),
            workspace_size.value,
            out.data(),
            packed.data(),
            scales.data(),
            None,
        )
    )
    if sync is not None:
        sync()

    torch.testing.assert_close(out.actual_tensor(), expected, atol=0, rtol=0)
    check_error(LIBINFINIOP.infiniopDestroyMxfp4DequantizeDescriptor(descriptor))


def test(
    handle,
    device,
    packed_shape,
    scales_shape,
    out_shape,
    dtype,
    sync=None,
):
    print(
        f"Testing MXFP4 dequantize on {InfiniDeviceNames[device]} "
        f"with packed_shape={packed_shape}, scales_shape={scales_shape}, "
        f"out_shape={out_shape}, out_dtype={InfiniDtypeNames[dtype]}"
    )

    probe = TestTensor((1,), None, InfiniDtype.U8, device, mode="zeros")
    torch_device = probe.actual_tensor().device
    packed_data = torch.randint(0, 256, packed_shape, dtype=torch.uint8, device=torch_device)
    # Keep all expected values finite in every requested output dtype.
    scales_data = torch.randint(120, 136, scales_shape, dtype=torch.uint8, device=torch_device)
    run_dequantize(handle, device, packed_data, scales_data, dtype, sync)


def test_checkpoint(
    handle,
    device,
    checkpoint_dir,
    weight_name,
    scale_name,
    row_start,
    row_count,
    dtype,
    sync=None,
):
    from safetensors import safe_open

    checkpoint_dir = Path(checkpoint_dir)
    index = json.loads(
        (checkpoint_dir / "model.safetensors.index.json").read_text()
    )
    weight_map = index["weight_map"]
    weight_shard = weight_map[weight_name]
    scale_shard = weight_map[scale_name]
    if weight_shard != scale_shard:
        raise RuntimeError("MXFP4 weight and scale must be in the same shard")

    with safe_open(checkpoint_dir / weight_shard, framework="pt", device="cpu") as f:
        packed_data = f.get_slice(weight_name)[
            row_start : row_start + row_count, :
        ]
        scales_data = f.get_slice(scale_name)[
            row_start : row_start + row_count, :
        ]

    print(
        f"Testing checkpoint MXFP4 dequantize on {InfiniDeviceNames[device]} "
        f"with tensor={weight_name}, rows={row_start}:{row_start + row_count}, "
        f"packed_shape={tuple(packed_data.shape)}, "
        f"scale_shape={tuple(scales_data.shape)}, "
        f"out_dtype={InfiniDtypeNames[dtype]}"
    )
    run_dequantize(handle, device, packed_data, scales_data, dtype, sync)


if __name__ == "__main__":
    args = get_args()
    checkpoint_dir = os.environ.get("INFINICORE_MXFP4_CHECKPOINT")
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
        if checkpoint_dir:
            checkpoint_cases = [
                (checkpoint_dir, *case) for case in _CHECKPOINT_CASES
            ]
            test_operator(
                device, test_checkpoint, checkpoint_cases, _TENSOR_DTYPES
            )
    print("\033[92mTest passed!\033[0m")
