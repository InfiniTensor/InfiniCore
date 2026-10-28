"""Qwen3.5 vision fused QKV extraction / 2-D RoPE / 72-to-128 padding.

Covers input/output/table/position strides, zero lanes, invalid positions,
output sentinels and native graph replay; model-level A/B is separate.
"""

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
    torch.manual_seed(185)
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
                str(Path(__file__).with_name("vision_qkv_rope_pad_bridge.cc")),
                "-L" + str(infini / "lib"),
                "-linfinicore_cpp_api",
                "-o",
                str(lib),
            ],
            check=True,
        )
        fn = ctypes.CDLL(str(lib)).test_vision_pack
        fn.argtypes = [ctypes.c_void_p] * 8 + [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int,
            ctypes.c_int,
        ]
        fn.restype = ctypes.c_double
        for n in [1, 3, 1200, 9600, 26400]:
            for pad in [0, 16]:
                # Pad the token row without changing the physical head width.
                packed = torch.randn(
                    n, 3 * 16 * 72 + pad, device="cuda", dtype=torch.bfloat16
                )
                qkv = packed[:, : 3 * 16 * 72].view(n, 3, 16, 72)
                angle = torch.arange(256, device="cuda")[:, None] / 10000 ** (
                    torch.arange(18, device="cuda")[None, :] / 18
                )
                cs = torch.empty(256, 72 + pad, device="cuda", dtype=torch.bfloat16)
                sn = torch.empty(256, 72 + 2 * pad, device="cuda", dtype=torch.bfloat16)
                cs[:, :72] = angle.cos().repeat(1, 4)
                sn[:, :72] = angle.sin().repeat(1, 4)
                positions = torch.randint(
                    -1, 257, (2, n, 2 if pad else 1), device="cuda", dtype=torch.int64
                )
                rows, cols = positions[0, :, 0], positions[1, :, 0]
                outputs = [
                    torch.full(
                        (n, 2048 + pad * (i + 1)),
                        7,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    for i in range(3)
                ]
                pos = torch.where(
                    torch.arange(36, device="cuda")[None, :] < 18,
                    rows[:, None],
                    cols[:, None],
                )
                pos = torch.where((pos >= 0) & (pos < 256), pos, 0)
                cv = cs[pos, torch.arange(36, device="cuda")[None, :]].float()[
                    :, None, :
                ]
                sv = sn[pos, torch.arange(36, device="cuda")[None, :]].float()[
                    :, None, :
                ]
                references = []
                for index in [0, 1]:
                    x0 = qkv[:, index, :, :36].float()
                    x1 = qkv[:, index, :, 36:].float()
                    references.append(
                        torch.nn.functional.pad(
                            torch.cat(
                                [x0 * cv - x1 * sv, x1 * cv + x0 * sv], -1
                            ).bfloat16(),
                            (0, 56),
                        )
                    )
                references.append(torch.nn.functional.pad(qkv[:, 2], (0, 56)))
                cfg = [
                    n,
                    *qkv.stride()[:3],
                    cs.stride(0),
                    sn.stride(0),
                    rows.stride(0),
                    cols.stride(0),
                    *[o.stride(0) for o in outputs],
                ]
                c = (ctypes.c_int64 * len(cfg))(*cfg)
                for mode in [0, 2]:
                    for out in outputs:
                        out.fill_(7)
                    torch.cuda.synchronize()
                    ms = fn(
                        *[t.data_ptr() for t in outputs + [qkv, cs, sn, rows, cols]],
                        c,
                        mode,
                        5,
                    )
                    assert ms >= 0
                    errs = []
                    for i, (out, ref) in enumerate(zip(outputs, references)):
                        actual = out[:, :2048].view(n, 16, 128)
                        err = float(
                            (
                                (actual.float() - ref.float()).square().mean()
                                / ref.float().square().mean()
                            ).sqrt()
                        )
                        assert err < 0.0002, (n, pad, mode, i, err)
                        assert torch.equal(
                            actual[:, :, 72:], torch.zeros_like(actual[:, :, 72:])
                        )
                        assert (out[:, 2048:] == 7).all()
                        errs.append(err)
                    row = dict(
                        tokens=n,
                        padding=pad,
                        graph=mode == 2,
                        max_nrms=max(errs),
                        milliseconds=ms,
                    )
                    results.append(row)
                    print("PASS", row, flush=True)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(results, indent=2))
    print("ALL_VISION_QKV_PACK_TESTS_PASSED", flush=True)


if __name__ == "__main__":
    main()
