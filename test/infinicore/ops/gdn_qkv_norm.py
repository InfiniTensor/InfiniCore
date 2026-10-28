"""GDN strided QKV pack + float L2 normalization, including native graph replay."""

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
import infinicore as ic


def main():
    torch.manual_seed(197)
    torch.set_num_threads(4)
    ic.set_device(ic.device("cuda", 0))
    root = Path(__file__).resolve().parents[3]
    infini = Path(os.environ["INFINI_ROOT"])
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        lib = Path(tmp) / "bridge.so"
        subprocess.run(
            [
                "clang++",
                "-std=c++17",
                "-shared",
                "-fPIC",
                "-D_GLIBCXX_USE_CXX11_ABI=0",
                "-I" + str(root / "include"),
                "-I" + str(infini / "include"),
                str(Path(__file__).with_name("gdn_qkv_norm_bridge.cc")),
                "-L" + str(infini / "lib"),
                "-linfinicore_cpp_api",
                "-o",
                str(lib),
            ],
            check=True,
        )
        fn = ctypes.CDLL(str(lib)).test_gdn_pack
        fn.argtypes = [ctypes.c_void_p] * 6 + [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int,
            ctypes.c_int,
        ]
        fn.restype = ctypes.c_double
        for n in [1, 3, 32, 357, 1349, 8192]:
            for pad in [0, 16]:
                data = torch.randn(n, 8192 + pad, device="cuda", dtype=torch.bfloat16)
                q = data[:, :2048].view(1, n, 16, 128)
                k = data[:, 2048:4096].view(1, n, 16, 128)
                v = data[:, 4096:8192].view(1, n, 32, 128)
                if n > 1:
                    q[:, 0] = 0
                    k[:, 1] = 0
                outputs = [
                    torch.empty_like(q, memory_format=torch.contiguous_format),
                    torch.empty_like(k, memory_format=torch.contiguous_format),
                    torch.empty_like(v, memory_format=torch.contiguous_format),
                ]

                def legacy():
                    qa = q.contiguous().float()
                    ka = k.contiguous().float()
                    return [
                        (qa * torch.rsqrt(qa.square().sum(-1, True) + 1e-6)).bfloat16(),
                        (ka * torch.rsqrt(ka.square().sum(-1, True) + 1e-6)).bfloat16(),
                        v.contiguous(),
                    ]

                ref = legacy()
                c = (ctypes.c_int64 * 7)(
                    n,
                    q.stride(1),
                    q.stride(2),
                    k.stride(1),
                    k.stride(2),
                    v.stride(1),
                    v.stride(2),
                )
                for _ in range(3):
                    legacy()
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(10):
                    legacy()
                b.record()
                b.synchronize()
                old_ms = a.elapsed_time(b) / 10
                for mode in [0, 2]:
                    torch.cuda.synchronize()
                    ms = fn(*[t.data_ptr() for t in outputs + [q, k, v]], c, mode, 10)
                    assert ms >= 0
                    errs = [
                        float(
                            (
                                (o.float() - r.float()).square().mean()
                                / (r.float().square().mean() + 1e-30)
                            ).sqrt()
                        )
                        for o, r in zip(outputs, ref)
                    ]
                    assert max(errs) < 0.0002, (n, pad, mode, errs)
                    assert torch.equal(outputs[2], v)
                    if n > 1:
                        assert (
                            torch.count_nonzero(outputs[0][:, 0]) == 0
                            and torch.count_nonzero(outputs[1][:, 1]) == 0
                        )
                    row = dict(
                        tokens=n,
                        padding=pad,
                        graph=mode == 2,
                        nrms=max(errs),
                        old_ms=old_ms,
                        new_ms=ms,
                    )
                    results.append(row)
                    print("PASS", row, flush=True)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(results, indent=2))
    print("ALL_GDN_QKV_PACK_TESTS_PASSED", flush=True)


if __name__ == "__main__":
    main()
