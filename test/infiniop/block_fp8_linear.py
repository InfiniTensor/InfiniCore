import ctypes
from ctypes import POINTER, c_int32, c_size_t, c_void_p

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceEnum,
    InfiniDeviceNames,
    InfiniDtype,
    TestTensor,
    TestWorkspace,
    check_error,
    get_args,
    get_test_devices,
    infiniopHandle_t,
    infiniopOperatorDescriptor_t,
    infiniopTensorDescriptor_t,
    test_operator,
)


LIBINFINIOP.infiniopCreateBlockFP8LinearDescriptor.restype = c_int32
LIBINFINIOP.infiniopCreateBlockFP8LinearDescriptor.argtypes = [
    infiniopHandle_t,
    POINTER(infiniopOperatorDescriptor_t),
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
]
LIBINFINIOP.infiniopGetBlockFP8LinearWorkspaceSize.restype = c_int32
LIBINFINIOP.infiniopGetBlockFP8LinearWorkspaceSize.argtypes = [
    infiniopOperatorDescriptor_t,
    POINTER(c_size_t),
]
LIBINFINIOP.infiniopBlockFP8Linear.restype = c_int32
LIBINFINIOP.infiniopBlockFP8Linear.argtypes = [
    infiniopOperatorDescriptor_t,
    c_void_p,
    c_size_t,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
]
LIBINFINIOP.infiniopDestroyBlockFP8LinearDescriptor.restype = c_int32
LIBINFINIOP.infiniopDestroyBlockFP8LinearDescriptor.argtypes = [
    infiniopOperatorDescriptor_t
]


_TEST_CASES = [
    (1, 128, 128),
    (16, 256, 128),
    (64, 256, 256),
]
_TENSOR_DTYPES = [None]
_BLOCK_SIZE = 128


def _reference(input_source, weight_source, weight_scale):
    m_count, k_count = input_source.shape
    n_count = weight_source.shape[0]
    input_blocks = k_count // _BLOCK_SIZE
    input_fp32 = input_source.float().reshape(
        m_count, input_blocks, _BLOCK_SIZE
    )
    activation_scale = input_fp32.abs().amax(dim=-1).div(448.0).clamp_min(1e-10)
    activation_fp8 = (
        input_fp32.div(activation_scale.unsqueeze(-1))
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
    )
    dequant_activation = (
        activation_fp8.float() * activation_scale.unsqueeze(-1)
    ).reshape(m_count, k_count)
    expanded_weight_scale = weight_scale.repeat_interleave(
        _BLOCK_SIZE, dim=0
    ).repeat_interleave(_BLOCK_SIZE, dim=1)[:n_count, :k_count]
    dequant_weight = weight_source.float() * expanded_weight_scale
    return (dequant_activation @ dequant_weight.transpose(0, 1)).to(
        torch.bfloat16
    )


def test(handle, device, m_count, n_count, k_count, _dtype, sync):
    if device != InfiniDeviceEnum.NVIDIA:
        print(f"Skipping BlockFP8Linear on {InfiniDeviceNames[device]}")
        return
    if torch.cuda.get_device_capability()[0] < 12:
        print("Skipping BlockFP8Linear because it requires SM120")
        return

    print(
        f"Testing BlockFP8Linear on NVIDIA with "
        f"M={m_count}, N={n_count}, K={k_count}"
    )
    generator = torch.Generator(device="cpu").manual_seed(
        2000 + m_count + n_count + k_count
    )
    input_source = (
        torch.randn((m_count, k_count), generator=generator) * 0.25
    ).to(torch.bfloat16)
    weight_source = (
        torch.randn((n_count, k_count), generator=generator) * 0.25
    ).to(torch.float8_e4m3fn)
    scale_shape = (
        (n_count + _BLOCK_SIZE - 1) // _BLOCK_SIZE,
        (k_count + _BLOCK_SIZE - 1) // _BLOCK_SIZE,
    )
    weight_scale_source = (
        torch.rand(scale_shape, generator=generator, dtype=torch.float32) * 0.5
        + 0.5
    )
    expected = _reference(input_source, weight_source, weight_scale_source)

    input_tensor = TestTensor.from_torch(
        input_source, InfiniDtype.BF16, device
    )
    weight = TestTensor.from_torch(weight_source, InfiniDtype.F8, device)
    weight_scale = TestTensor.from_torch(
        weight_scale_source, InfiniDtype.F32, device
    )
    output = TestTensor(
        (m_count, n_count), None, InfiniDtype.BF16, device, mode="zeros"
    )

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateBlockFP8LinearDescriptor(
            handle,
            ctypes.byref(descriptor),
            output.descriptor,
            input_tensor.descriptor,
            weight.descriptor,
            weight_scale.descriptor,
        )
    )
    for tensor in (output, input_tensor, weight, weight_scale):
        tensor.destroy_desc()

    workspace_size = c_size_t(0)
    check_error(
        LIBINFINIOP.infiniopGetBlockFP8LinearWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)
    if m_count == 1:
        status = LIBINFINIOP.infiniopBlockFP8Linear(
            descriptor,
            workspace.data(),
            workspace_size.value - 1,
            output.data(),
            input_tensor.data(),
            weight.data(),
            weight_scale.data(),
            None,
        )
        assert status != 0

    check_error(
        LIBINFINIOP.infiniopBlockFP8Linear(
            descriptor,
            workspace.data(),
            workspace_size.value,
            output.data(),
            input_tensor.data(),
            weight.data(),
            weight_scale.data(),
            None,
        )
    )
    if sync is not None:
        sync()

    actual = output.actual_tensor().cpu()
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=0.25, rtol=0.08
    )
    check_error(LIBINFINIOP.infiniopDestroyBlockFP8LinearDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    for test_device in get_test_devices(args):
        test_operator(test_device, test, _TEST_CASES, _TENSOR_DTYPES)
    print("\033[92mBlockFP8Linear test passed!\033[0m")
