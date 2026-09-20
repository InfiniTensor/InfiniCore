import ctypes
import struct
from ctypes import POINTER, c_int32, c_int64, c_size_t, c_void_p

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


LIBINFINIOP.infiniopCreateLinearGgufDescriptor.restype = c_int32
LIBINFINIOP.infiniopCreateLinearGgufDescriptor.argtypes = [
    infiniopHandle_t,
    POINTER(infiniopOperatorDescriptor_t),
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    c_int64,
]
LIBINFINIOP.infiniopGetLinearGgufWorkspaceSize.restype = c_int32
LIBINFINIOP.infiniopGetLinearGgufWorkspaceSize.argtypes = [
    infiniopOperatorDescriptor_t,
    POINTER(c_size_t),
]
LIBINFINIOP.infiniopLinearGguf.restype = c_int32
LIBINFINIOP.infiniopLinearGguf.argtypes = [
    infiniopOperatorDescriptor_t,
    c_void_p,
    c_size_t,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
]
LIBINFINIOP.infiniopDestroyLinearGgufDescriptor.restype = c_int32
LIBINFINIOP.infiniopDestroyLinearGgufDescriptor.argtypes = [
    infiniopOperatorDescriptor_t
]


Q8_0 = 8
Q4_K = 12
Q5_K = 13
Q6_K = 14
BLOCK_ELEMS = {Q8_0: 32, Q4_K: 256, Q5_K: 256, Q6_K: 256}
BLOCK_BYTES = {Q8_0: 34, Q4_K: 144, Q5_K: 176, Q6_K: 210}
_TEST_CASES = [
    (ggml_type, m, 8, 256)
    for ggml_type in (Q8_0, Q4_K, Q5_K, Q6_K)
    for m in (1, 8, 17)
]
_TENSOR_DTYPES = [None]


def _signed(value):
    return value - 256 if value >= 128 else value


def _half(data, offset=0):
    return struct.unpack_from("<e", data, offset)[0]


def _scale_min(scales, index):
    if index < 4:
        return scales[index] & 63, scales[index + 4] & 63
    scale = (scales[index + 4] & 15) | ((scales[index - 4] >> 6) << 4)
    minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4)
    return scale, minimum


def _decode_block(ggml_type, block):
    if ggml_type == Q8_0:
        scale = _half(block)
        return [_signed(value) * scale for value in block[2:34]]

    if ggml_type in (Q4_K, Q5_K):
        scale = _half(block)
        minimum = _half(block, 2)
        scales = block[4:16]
        high = block[16:48] if ggml_type == Q5_K else None
        quants = block[48:] if ggml_type == Q5_K else block[16:]
        output = [0.0] * 256
        q_offset = 0
        high_mask_1, high_mask_2 = 1, 2
        for base, scale_index in zip(range(0, 256, 64), range(0, 8, 2)):
            scale_1, min_1 = _scale_min(scales, scale_index)
            scale_2, min_2 = _scale_min(scales, scale_index + 1)
            for lane in range(32):
                low = quants[q_offset + lane]
                q1 = low & 15
                q2 = low >> 4
                if high is not None:
                    q1 += 16 if high[lane] & high_mask_1 else 0
                    q2 += 16 if high[lane] & high_mask_2 else 0
                output[base + lane] = scale * scale_1 * q1 - minimum * min_1
                output[base + 32 + lane] = (
                    scale * scale_2 * q2 - minimum * min_2
                )
            q_offset += 32
            high_mask_1 <<= 2
            high_mask_2 <<= 2
        return output

    scale = _half(block, 208)
    low = block[:128]
    high = block[128:192]
    scales = [_signed(value) for value in block[192:208]]
    output = [0.0] * 256
    for half_index in range(2):
        low_offset = half_index * 64
        high_offset = half_index * 32
        scale_offset = half_index * 8
        out_offset = half_index * 128
        for lane in range(32):
            group = lane // 16
            q1 = (low[low_offset + lane] & 15) | (
                ((high[high_offset + lane] >> 0) & 3) << 4
            )
            q2 = (low[low_offset + lane + 32] & 15) | (
                ((high[high_offset + lane] >> 2) & 3) << 4
            )
            q3 = (low[low_offset + lane] >> 4) | (
                ((high[high_offset + lane] >> 4) & 3) << 4
            )
            q4 = (low[low_offset + lane + 32] >> 4) | (
                ((high[high_offset + lane] >> 6) & 3) << 4
            )
            output[out_offset + lane] = (
                scale * scales[scale_offset + group] * (q1 - 32)
            )
            output[out_offset + lane + 32] = (
                scale * scales[scale_offset + group + 2] * (q2 - 32)
            )
            output[out_offset + lane + 64] = (
                scale * scales[scale_offset + group + 4] * (q3 - 32)
            )
            output[out_offset + lane + 96] = (
                scale * scales[scale_offset + group + 6] * (q4 - 32)
            )
    return output


def _make_block(ggml_type, seed):
    if ggml_type == Q8_0:
        quant = [((seed * 13 + index * 17) % 255) - 127 for index in range(32)]
        return bytearray(struct.pack("<e", 0.0078125)) + bytearray(
            value & 255 for value in quant
        )
    if ggml_type in (Q4_K, Q5_K):
        header = bytearray(struct.pack("<ee", 0.015625, 0.0078125))
        scales = bytearray((seed * 11 + index * 7 + 1) & 255 for index in range(12))
        quants = bytearray((seed * 19 + index * 23) & 255 for index in range(128))
        if ggml_type == Q4_K:
            return header + scales + quants
        high = bytearray((seed * 29 + index * 31) & 255 for index in range(32))
        return header + scales + high + quants
    low = bytearray((seed * 17 + index * 13) & 255 for index in range(128))
    high = bytearray((seed * 23 + index * 11) & 255 for index in range(64))
    scales = bytearray((((seed + index * 3) % 15) - 7) & 255 for index in range(16))
    return low + high + scales + bytearray(struct.pack("<e", 0.015625))


def _make_weight(ggml_type, n_count, k_count):
    blocks_per_row = k_count // BLOCK_ELEMS[ggml_type]
    packed_rows = []
    dense_rows = []
    for row in range(n_count):
        packed_row = bytearray()
        dense_row = []
        for block_index in range(blocks_per_row):
            block = _make_block(ggml_type, row * blocks_per_row + block_index + 1)
            assert len(block) == BLOCK_BYTES[ggml_type]
            packed_row.extend(block)
            dense_row.extend(_decode_block(ggml_type, block))
        packed_rows.append(list(packed_row))
        dense_rows.append(dense_row)
    return torch.tensor(packed_rows, dtype=torch.uint8), torch.tensor(
        dense_rows, dtype=torch.float32
    )


def _assert_invalid_descriptors(handle, output, activation, weight, ggml_type):
    invalid_type_desc = infiniopOperatorDescriptor_t()
    status = LIBINFINIOP.infiniopCreateLinearGgufDescriptor(
        handle,
        ctypes.byref(invalid_type_desc),
        output.descriptor,
        activation.descriptor,
        weight.descriptor,
        999,
    )
    assert status != 0

    bad_weight = TestTensor(
        (weight.shape[0], weight.shape[1] - 1),
        None,
        InfiniDtype.U8,
        InfiniDeviceEnum.NVIDIA,
        mode="zeros",
    )
    invalid_shape_desc = infiniopOperatorDescriptor_t()
    status = LIBINFINIOP.infiniopCreateLinearGgufDescriptor(
        handle,
        ctypes.byref(invalid_shape_desc),
        output.descriptor,
        activation.descriptor,
        bad_weight.descriptor,
        ggml_type,
    )
    assert status != 0
    bad_weight.destroy_desc()


def test(handle, device, ggml_type, m_count, n_count, k_count, _dtype, sync):
    if device != InfiniDeviceEnum.NVIDIA:
        print(f"Skipping LinearGguf on {InfiniDeviceNames[device]}")
        return
    print(
        f"Testing LinearGguf on NVIDIA with type={ggml_type}, "
        f"M={m_count}, N={n_count}, K={k_count}"
    )
    generator = torch.Generator(device="cpu").manual_seed(
        1000 + ggml_type * 100 + m_count
    )
    activation_source = (
        torch.randn((m_count, k_count), generator=generator) * 0.125
    ).to(torch.bfloat16)
    packed_source, dense_weight = _make_weight(ggml_type, n_count, k_count)
    expected = (activation_source.float() @ dense_weight.transpose(0, 1)).to(
        torch.bfloat16
    )

    activation = TestTensor.from_torch(
        activation_source, InfiniDtype.BF16, device
    )
    weight = TestTensor.from_torch(packed_source, InfiniDtype.U8, device)
    output = TestTensor(
        (m_count, n_count), None, InfiniDtype.BF16, device, mode="zeros"
    )

    if ggml_type == Q8_0 and m_count == 1:
        _assert_invalid_descriptors(
            handle, output, activation, weight, ggml_type
        )

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateLinearGgufDescriptor(
            handle,
            ctypes.byref(descriptor),
            output.descriptor,
            activation.descriptor,
            weight.descriptor,
            ggml_type,
        )
    )
    for tensor in (output, activation, weight):
        tensor.destroy_desc()

    workspace_size = c_size_t(0)
    check_error(
        LIBINFINIOP.infiniopGetLinearGgufWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)
    check_error(
        LIBINFINIOP.infiniopLinearGguf(
            descriptor,
            workspace.data(),
            workspace_size.value,
            output.data(),
            activation.data(),
            weight.data(),
            None,
        )
    )
    if sync is not None:
        sync()

    actual = output.actual_tensor().cpu()
    assert torch.allclose(actual.float(), expected.float(), atol=0.125, rtol=0.03)
    check_error(LIBINFINIOP.infiniopDestroyLinearGgufDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    for test_device in get_test_devices(args):
        test_operator(test_device, test, _TEST_CASES, _TENSOR_DTYPES)
    print("\033[92mLinearGguf test passed!\033[0m")
