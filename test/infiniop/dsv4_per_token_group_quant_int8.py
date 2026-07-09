import ctypes
from ctypes import c_int32, c_size_t

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
)


def _workspace(desc, getter, device):
    size = c_size_t(0)
    check_error(getter(desc, ctypes.byref(size)))
    return TestWorkspace(size.value, device), size.value


def _ref_group_quant(x, group_size):
    shape = x.shape
    grouped = x.float().reshape(-1, shape[-1] // group_size, group_size)
    absmax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = absmax / 127.0
    q = torch.clamp(grouped / scale, -128.0, 127.0).to(torch.int8)
    return q.reshape(shape), scale.reshape(*shape[:-1], shape[-1] // group_size)


_CASES = [
    ((4, 512), 128),
    ((64, 4096), 128),
    ((8, 256, 4096), 128),
]


def test_op(handle, device, shape, group_size, dtype=InfiniDtype.BF16, sync=None):
    print(
        f"Testing DSV4 per_token_group_quant_int8 on {InfiniDeviceNames[device]} "
        f"shape:{shape} group_size:{group_size} dtype:{InfiniDtypeNames[dtype]}"
    )
    x = TestTensor(shape, None, dtype, device)
    q = TestTensor(shape, None, InfiniDtype.I8, device, mode="zeros")
    scale_shape = shape[:-1] + (shape[-1] // group_size,)
    scale = TestTensor(scale_shape, None, InfiniDtype.F32, device, mode="zeros")

    ref_q, ref_s = _ref_group_quant(x.torch_tensor(), group_size)
    if sync:
        sync()

    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4PerTokenGroupQuantInt8Descriptor(
            handle,
            ctypes.byref(desc),
            q.descriptor,
            scale.descriptor,
            x.descriptor,
            c_int32(group_size),
        )
    )
    for tensor in [x, q, scale]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc,
        LIBINFINIOP.infiniopGetDsv4PerTokenGroupQuantInt8WorkspaceSize,
        device,
    )
    check_error(
        LIBINFINIOP.infiniopDsv4PerTokenGroupQuantInt8(
            desc,
            workspace.data(),
            workspace_size,
            q.data(),
            scale.data(),
            x.data(),
            None,
        )
    )
    assert torch.equal(q.actual_tensor(), ref_q)
    assert torch.allclose(scale.actual_tensor(), ref_s, atol=1e-5, rtol=1e-5)
    check_error(LIBINFINIOP.infiniopDestroyDsv4PerTokenGroupQuantInt8Descriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.BF16])
    print("\033[92mDSV4 per_token_group_quant_int8 Test passed!\033[0m")
