import ctypes
from ctypes import c_float, c_size_t

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
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


_CASES = [((8, 4),), ((64, 4),), ((2, 16, 4),)]
_EPS = 1e-6


def test_op(handle, device, shape, dtype=InfiniDtype.BF16, sync=None):
    print(f"Testing DSV4 mhc_pre on {InfiniDeviceNames[device]} shape:{shape}")
    input_mix = TestTensor(shape, None, dtype, device, scale=4.0, bias=-2.0)
    output = TestTensor(shape, None, dtype, device, mode="zeros")
    mhc_scale = TestTensor((1,), None, InfiniDtype.F32, device, mode="ones")
    mhc_base = TestTensor(
        (shape[-1],), None, InfiniDtype.F32, device, mode="random", scale=2.0, bias=-1.0
    )

    ref = (
        torch.sigmoid(
            input_mix.torch_tensor().float() * mhc_scale.torch_tensor()[0].float()
            + mhc_base.torch_tensor().float()
        )
        + _EPS
    )
    ref = ref.to(output.torch_tensor().dtype)

    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4MhcPreDescriptor(
            handle,
            ctypes.byref(desc),
            output.descriptor,
            input_mix.descriptor,
            mhc_scale.descriptor,
            mhc_base.descriptor,
            c_float(_EPS),
        )
    )
    for tensor in [input_mix, output, mhc_scale, mhc_base]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc,
        LIBINFINIOP.infiniopGetDsv4MhcPreWorkspaceSize,
        device,
    )
    check_error(
        LIBINFINIOP.infiniopDsv4MhcPre(
            desc,
            workspace.data(),
            workspace_size,
            output.data(),
            input_mix.data(),
            mhc_scale.data(),
            mhc_base.data(),
            None,
        )
    )
    assert torch.allclose(output.actual_tensor(), ref, atol=2e-3, rtol=2e-3)
    check_error(LIBINFINIOP.infiniopDestroyDsv4MhcPreDescriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.BF16])
    print("\033[92mDSV4 mhc_pre Test passed!\033[0m")
