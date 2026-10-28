"""Numerical/cache-isolation regression for the C++ GDN prefill bridge.

Run after building InfiniCore with ATen and Iluvatar support. The helper only
exposes the existing public C++ adaptor; all model work remains in the engine.
"""

import ctypes
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import torch
import infinicore as ic
from infinicore.nn.functional import chunk_gated_delta_rule


def upload(t):
    return ic.from_torch(t.cpu()).to(ic.device("cuda", 0))


def read(t):
    cpu = t.to(ic.device("cpu"))
    raw = ctypes.string_at(cpu.data_ptr(), t.numel() * 2)
    return (
        torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy())
        .view(torch.bfloat16)
        .reshape(tuple(t.shape))
    )


def relative_rms(actual, expected):
    error = (actual.float() - expected.float()).square().mean().sqrt()
    return (error / expected.float().square().mean().sqrt().clamp_min(1e-8)).item()


def main():
    torch.set_num_threads(4)
    torch.manual_seed(31)
    ic.set_device(ic.device("cuda", 0))
    root = Path(__file__).resolve().parents[3]
    infini_root = Path(os.environ.get("INFINI_ROOT", str(Path.home() / ".infini")))
    with tempfile.TemporaryDirectory(prefix="infinicore-gdn-test-") as directory:
        library = Path(directory) / "bridge.so"
        subprocess.run(
            [
                os.environ.get("CXX", "clang++"),
                "-std=c++17",
                "-shared",
                "-fPIC",
                "-D_GLIBCXX_USE_CXX11_ABI=0",
                "-I" + str(root / "include"),
                "-I" + str(infini_root / "include"),
                str(Path(__file__).with_name("iluvatar_gdn_bridge.cc")),
                "-L" + str(infini_root / "lib"),
                "-linfinicore_cpp_api",
                "-o",
                str(library),
            ],
            check=True,
        )
        fn = ctypes.CDLL(str(library)).test_iluvatar_chunk
        fn.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int] * 3
        fn.restype = ctypes.c_int
        for lengths in [
            [129],
            [357],
            [1045],
            [129, 357, 1045],
            [357] * 16,
            [1349] * 6,
            [1349, 1, 1],
            [1, 129, 1, 357],
        ]:
            batch, tokens = len(lengths), sum(lengths)
            offsets = torch.tensor([0] + list(np.cumsum(lengths)), dtype=torch.int32)
            cu = offsets.cuda()
            q = (torch.randn(1, tokens, 16, 128, device="cuda") * 0.2).bfloat16()
            k = torch.randn_like(q) * 0.2
            v = (torch.randn(1, tokens, 32, 128, device="cuda") * 0.2).bfloat16()
            g = (-torch.rand(1, tokens, 32, device="cuda") * 0.2).bfloat16()
            beta = torch.rand(1, tokens, 32, device="cuda").bfloat16()
            pool = batch * 2 + 2
            state = (torch.randn(pool, 32, 128, 128, device="cuda") * 0.02).bfloat16()
            initial = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
            final = initial.flip(
                0
            ).contiguous()  # Overlapping read/write slots, including swaps.
            output = torch.empty_like(v)
            for iteration in range(3):
                before = state.clone()
                # Independent read and write regions avoid races in the native
                # reference when the candidate deliberately swaps live slots.
                reference_pool = torch.zeros_like(state)
                reference_pool[:batch] = before[initial.long()]
                reference = upload(reference_pool)
                expected = chunk_gated_delta_rule(
                    upload(q),
                    upload(k),
                    upload(v),
                    upload(g),
                    upload(beta),
                    reference,
                    cu_seqlens=upload(cu),
                    initial_state_indices=upload(
                        torch.arange(batch, dtype=torch.int32)
                    ),
                    final_state_indices=upload(
                        torch.arange(batch, batch * 2, dtype=torch.int32)
                    ),
                    use_qk_l2norm=True,
                )
                expected_output = read(expected)
                expected_state = read(reference)[batch : batch * 2]
                tensors = [q, k, v, g, beta, state, output, offsets, cu, initial, final]
                assert fn(*[t.data_ptr() for t in tensors], tokens, batch, pool) == 0
                actual = output.cpu()
                out_error = relative_rms(actual, expected_output)
                state_error = relative_rms(state[final.long()].cpu(), expected_state)
                assert out_error < 0.02 and state_error < 0.02, (
                    lengths,
                    out_error,
                    state_error,
                )
                untouched = torch.ones(pool, dtype=torch.bool, device="cuda")
                untouched[final.long()] = False
                assert torch.equal(state[untouched], before[untouched])
                print(
                    "PASS",
                    lengths,
                    iteration,
                    "output_nrms",
                    out_error,
                    "state_nrms",
                    state_error,
                    flush=True,
                )
            os.environ["INFINICORE_DISABLE_IXFORMER_GDN"] = "1"
            before = state.clone()
            assert fn(*[t.data_ptr() for t in tensors], tokens, batch, pool) == 2
            assert torch.equal(state, before)
            del os.environ["INFINICORE_DISABLE_IXFORMER_GDN"]
    print("All C++ vendor GDN/cache-isolation checks passed", flush=True)


if __name__ == "__main__":
    main()
