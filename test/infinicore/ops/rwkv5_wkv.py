import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from framework import (
    InfiniDeviceNames,
    TensorInitializer,
    convert_infinicore_to_torch,
    get_args,
    get_test_devices,
    infinicore_tensor_from_torch,
    torch_device_map,
)
from framework.datatypes import to_torch_dtype
from framework.utils.tensor_utils import synchronize_device

import infinicore

_TEST_CASES = [
    {"shape": (1, 1, 2, 3), "seed": 11},
    {"shape": (2, 3, 3, 4), "seed": 23},
]
_DTYPES = [infinicore.float16, infinicore.float32, infinicore.bfloat16]
_TOLERANCES = {
    infinicore.float16: {"atol": 3e-3, "rtol": 3e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 3e-2, "rtol": 8e-2},
}


def _reference(receptance, key, value, time_decay, time_faaaa, state):
    input_dtype = receptance.dtype
    batch, seq_len, hidden_size = receptance.shape
    num_heads, head_size = time_faaaa.shape

    r = receptance.float().view(batch, seq_len, num_heads, head_size).transpose(1, 2)
    k = key.float().view(batch, seq_len, num_heads, head_size).transpose(1, 2)
    v = value.float().view(batch, seq_len, num_heads, head_size).transpose(1, 2)
    decay = torch.exp(-torch.exp(time_decay.float())).view(1, num_heads, head_size, 1)
    first = time_faaaa.float().view(1, num_heads, head_size, 1)
    next_state = state.float().clone()
    out = torch.empty(
        (batch, seq_len, num_heads, head_size),
        dtype=torch.float32,
        device=receptance.device,
    )

    for t in range(seq_len):
        att = k[:, :, t, :, None] * v[:, :, t, None, :]
        out[:, t] = torch.matmul(r[:, :, t, None, :], first * att + next_state).squeeze(
            2
        )
        next_state = att + decay * next_state

    return out.reshape(batch, seq_len, hidden_size).to(input_dtype), next_state


def _make_tensor(shape, dtype, device, seed, scale=1.0, bias=0.0):
    torch.manual_seed(seed)
    return TensorInitializer.create_tensor(shape, dtype, device, scale=scale, bias=bias)


def _run_case(device, dtype, shape, seed):
    batch, seq_len, num_heads, head_size = shape
    hidden_size = num_heads * head_size
    tensor_shape = (batch, seq_len, hidden_size)
    param_shape = (num_heads, head_size)
    state_shape = (batch, num_heads, head_size, head_size)

    torch_device = torch_device_map[device]
    infinicore.set_device(infinicore.device(torch_device, 0))

    receptance = _make_tensor(tensor_shape, dtype, device, seed, scale=1.2, bias=-0.6)
    key = _make_tensor(tensor_shape, dtype, device, seed + 1, scale=1.0, bias=-0.5)
    value = _make_tensor(tensor_shape, dtype, device, seed + 2, scale=1.0, bias=-0.5)
    time_decay = _make_tensor(
        param_shape, dtype, device, seed + 3, scale=2.0, bias=-1.0
    )
    time_faaaa = _make_tensor(
        param_shape, dtype, device, seed + 4, scale=1.0, bias=-0.5
    )
    state = _make_tensor(
        state_shape, infinicore.float32, device, seed + 5, scale=0.4, bias=-0.2
    )

    expected_out, expected_state = _reference(
        receptance, key, value, time_decay, time_faaaa, state
    )

    state_for_infini = state.clone()
    out = infinicore.rwkv5_wkv(
        infinicore_tensor_from_torch(receptance),
        infinicore_tensor_from_torch(key),
        infinicore_tensor_from_torch(value),
        infinicore_tensor_from_torch(time_decay),
        infinicore_tensor_from_torch(time_faaaa),
        infinicore_tensor_from_torch(state_for_infini),
    )
    synchronize_device(torch_device)

    actual_out = convert_infinicore_to_torch(out)
    tol = _TOLERANCES[dtype]
    torch.testing.assert_close(
        actual_out, expected_out, atol=tol["atol"], rtol=tol["rtol"]
    )
    torch.testing.assert_close(state_for_infini, expected_state, atol=3e-5, rtol=2e-4)

    explicit_out = torch.empty_like(receptance)
    state_for_explicit = state.clone()
    infinicore.rwkv5_wkv_(
        infinicore_tensor_from_torch(explicit_out),
        infinicore_tensor_from_torch(receptance),
        infinicore_tensor_from_torch(key),
        infinicore_tensor_from_torch(value),
        infinicore_tensor_from_torch(time_decay),
        infinicore_tensor_from_torch(time_faaaa),
        infinicore_tensor_from_torch(state_for_explicit),
    )
    synchronize_device(torch_device)
    torch.testing.assert_close(
        explicit_out, expected_out, atol=tol["atol"], rtol=tol["rtol"]
    )
    torch.testing.assert_close(state_for_explicit, expected_state, atol=3e-5, rtol=2e-4)


def main():
    args = get_args()
    devices = [device for device in get_test_devices(args) if device != 0]
    failed = []

    for device in devices:
        print(f"\n{'=' * 60}")
        print(f"Testing Rwkv5Wkv on {InfiniDeviceNames[device]}")
        print(f"{'=' * 60}")
        for case in _TEST_CASES:
            for dtype in _DTYPES:
                if dtype == infinicore.bfloat16 and torch_device_map[device] == "musa":
                    continue
                desc = f"shape={case['shape']}, dtype={to_torch_dtype(dtype)}"
                try:
                    print(desc)
                    _run_case(device, dtype, case["shape"], case["seed"])
                    print("\033[92m✓\033[0m Passed")
                except Exception as exc:
                    failed.append((InfiniDeviceNames[device], desc, exc))
                    print(f"\033[91m✗\033[0m {exc}")
                    if args.debug or args.verbose:
                        raise

    if failed:
        print("\n\033[91mRwkv5Wkv failures:\033[0m")
        for device_name, desc, exc in failed:
            print(f" - {device_name} {desc}: {exc}")
        sys.exit(1)

    print("\n\033[92mAll Rwkv5Wkv tests passed!\033[0m")


if __name__ == "__main__":
    main()
