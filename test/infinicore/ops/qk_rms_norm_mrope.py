"""Fused head256 Q/K RMSNorm + partial MRoPE: strides, axes, graph replay."""

import ctypes, json, os, subprocess, tempfile, sys
from pathlib import Path
import torch, infinicore as ic


def main():
    torch.manual_seed(173)
    torch.set_num_threads(4)
    ic.set_device(ic.device("cuda", 0))
    root = Path(__file__).resolve().parents[3]
    infini = Path(os.environ["INFINI_ROOT"])
    rows = []
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
                str(Path(__file__).with_name("qk_rms_norm_mrope_bridge.cc")),
                "-L" + str(infini / "lib"),
                "-linfinicore_cpp_api",
                "-o",
                str(lib),
            ],
            check=True,
        )
        fn = ctypes.CDLL(str(lib)).test_qk_norm_mrope
        fn.argtypes = [ctypes.c_void_p] * 9 + [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int,
            ctypes.c_int,
        ]
        fn.restype = ctypes.c_double
        for n, qh, kh, pad in [
            (1, 16, 4, 0),
            (3, 3, 1, 16),
            (5, 8, 2, 0),
            (32, 16, 4, 16),
            (1349, 16, 4, 0),
            (8192, 16, 4, 0),
        ]:
            stride = (qh * 2 + kh * 2) * 256
            packed = torch.randn(n, stride, device="cuda", dtype=torch.bfloat16)
            q = torch.as_strided(packed, (n, qh, 256), (stride, 512, 1))
            k = torch.as_strided(packed, (n, kh, 256), (stride, 256, 1), qh * 512)
            qw = (1 + torch.randn(256, device="cuda") * 0.03).bfloat16()
            kw = (1 + torch.randn(256, device="cuda") * 0.03).bfloat16()
            angle = torch.arange(2048, device="cuda")[:, None] / 10000 ** (
                torch.arange(32, device="cuda")[None, :] / 32
            )
            cs = torch.empty(2048, 32 + pad, device="cuda", dtype=torch.bfloat16)
            cs[:, :32] = angle.cos()
            sn = torch.empty(2048, 32 + 2 * pad, device="cuda", dtype=torch.bfloat16)
            sn[:, :32] = angle.sin()
            pos = torch.randint(
                0, 2048, (3, n, 2 if pad else 1), device="cuda", dtype=torch.int64
            )
            qo = torch.full((n, qh * 256 + pad), 7, device="cuda", dtype=torch.bfloat16)
            ko = torch.full((n, kh * 256 + pad), 7, device="cuda", dtype=torch.bfloat16)
            rq = qo.clone()
            rk = ko.clone()
            for inter in [0, 1]:
                for axes in [0, 1]:
                    cfg = [
                        n,
                        qh,
                        kh,
                        stride,
                        512,
                        stride,
                        256,
                        qo.stride(0),
                        256,
                        ko.stride(0),
                        256,
                        2048,
                        cs.stride(0),
                        sn.stride(0),
                        pos.stride(0) if axes else 0,
                        pos.stride(1),
                        11,
                        11,
                        10,
                        inter,
                    ]
                    c = (ctypes.c_int64 * len(cfg))(*cfg)

                    def call(a, b, mode, loops):
                        torch.cuda.synchronize()
                        return fn(
                            *[t.data_ptr() for t in [a, b, q, k, qw, kw, cs, sn, pos]],
                            c,
                            mode,
                            loops,
                        )

                    old = call(rq, rk, 1, 3)
                    new = call(qo, ko, 0, 3)
                    assert old >= 0 and new >= 0
                    nrms = max(
                        float(
                            (
                                (a[:, : h * 256].float() - b[:, : h * 256].float())
                                .square()
                                .mean()
                                / b[:, : h * 256].float().square().mean()
                            ).sqrt()
                        )
                        for a, b, h in [(qo, rq, qh), (ko, rk, kh)]
                    )
                    assert nrms < 0.001, (n, inter, axes, nrms)
                    if pad:
                        assert (qo[:, -pad:] == 7).all() and (ko[:, -pad:] == 7).all()
                    if n <= 32:
                        qo.fill_(7)
                        ko.fill_(7)
                        assert call(qo, ko, 2, 3) >= 0
                        assert torch.allclose(
                            qo.float(), rq.float(), atol=0.04, rtol=0.02
                        )
                        assert torch.allclose(
                            ko.float(), rk.float(), atol=0.04, rtol=0.02
                        )
                    row = dict(
                        tokens=n,
                        q_heads=qh,
                        k_heads=kh,
                        padding=pad,
                        interleaved=inter,
                        axes=axes,
                        nrms=nrms,
                        old_ms=old,
                        new_ms=new,
                    )
                    rows.append(row)
                    print("PASS", row, flush=True)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(rows, indent=2))
    print("ALL_FUSED_QK_MROPE_TESTS_PASSED", flush=True)


if __name__ == "__main__":
    main()
