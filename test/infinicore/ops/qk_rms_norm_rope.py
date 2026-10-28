"""Native fused QK norm/RoPE regression, including packed and padded strides."""

import ctypes, json, os, subprocess, tempfile
from pathlib import Path
import torch, infinicore as ic
from infinicore.nn.functional import rms_norm, rope


def main():
    torch.manual_seed(61)
    torch.set_num_threads(4)
    ic.set_device(ic.device("cuda", 0))
    root = Path(__file__).resolve().parents[3]
    infini = Path(os.environ["INFINI_ROOT"])
    rows = []
    with tempfile.TemporaryDirectory() as temp:
        lib = Path(temp) / "bridge.so"
        subprocess.run(
            [
                "clang++",
                "-std=c++17",
                "-shared",
                "-fPIC",
                "-D_GLIBCXX_USE_CXX11_ABI=0",
                "-I" + str(root / "include"),
                "-I" + str(infini / "include"),
                str(Path(__file__).with_name("qk_rms_norm_rope_bridge.cc")),
                "-L" + str(infini / "lib"),
                "-linfinicore_cpp_api",
                "-o",
                str(lib),
            ],
            check=True,
        )
        fn = ctypes.CDLL(str(lib)).test_qk_norm_rope
        fn.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 9
        fn.restype = ctypes.c_int
        for tokens, qh, kh, pad in [
            (1, 32, 8, 0),
            (3, 3, 1, 16),
            (5, 8, 2, 0),
            (8, 32, 8, 0),
            (32, 32, 8, 16),
            (53, 32, 8, 0),
            (1045, 32, 8, 0),
            (15675, 32, 8, 0),
        ]:
            stride = (qh + kh) * 128 + 128
            packed = torch.randn(tokens, stride, device="cuda", dtype=torch.bfloat16)
            qw = (1 + torch.randn(128, device="cuda") * 0.03).bfloat16()
            kw = (1 + torch.randn(128, device="cuda") * 0.03).bfloat16()
            positions = torch.randint(
                0, 2048, (tokens,), device="cuda", dtype=torch.int64
            )
            angle = torch.arange(2048, device="cuda")[:, None] / 10000 ** (
                torch.arange(64, device="cuda")[None, :] / 64
            )
            cos = angle.cos().bfloat16()
            sin = angle.sin().bfloat16()
            # Different table row strides and a non-unit positions stride.
            cos_store = torch.empty(
                (2048, 80 if pad else 64), device="cuda", dtype=torch.bfloat16
            )
            sin_store = torch.empty(
                (2048, 96 if pad else 64), device="cuda", dtype=torch.bfloat16
            )
            pos_store = torch.empty(
                (tokens, 2 if pad else 1), device="cuda", dtype=torch.int64
            )
            cos_store[:, :64] = cos
            sin_store[:, :64] = sin
            pos_store[:, 0] = positions
            qo = torch.full(
                (tokens, qh * 128 + pad), 7, device="cuda", dtype=torch.bfloat16
            )
            ko = torch.full(
                (tokens, kh * 128 + pad), 7, device="cuda", dtype=torch.bfloat16
            )
            ix = ic.from_torch(packed)
            q = ix.narrow(1, 0, qh * 128).view([tokens, qh, 128])
            k = ix.narrow(1, qh * 128, kh * 128).view([tokens, kh, 128])
            rq = rms_norm(q, [128], ic.from_torch(qw), 1e-6)
            rk = rms_norm(k, [128], ic.from_torch(kw), 1e-6)
            args = [ic.from_torch(x) for x in [positions, sin, cos]]
            rope(rq, *args, out=rq)
            rope(rk, *args, out=rk)
            ic.sync_device()
            torch.cuda.synchronize()
            for repeat in range(2):
                assert (
                    fn(
                        *[
                            x.data_ptr()
                            for x in [
                                qo,
                                ko,
                                packed,
                                qw,
                                kw,
                                cos_store,
                                sin_store,
                                pos_store,
                            ]
                        ],
                        tokens,
                        qh,
                        kh,
                        stride,
                        2048,
                        pad,
                        cos_store.stride(0),
                        sin_store.stride(0),
                        pos_store.stride(0),
                    )
                    == 0
                )

                def read(t):
                    t = t.to(ic.device("cpu"))
                    raw = ctypes.string_at(t.data_ptr(), t.numel() * 2)
                    import numpy as np

                    return (
                        torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy())
                        .view(torch.bfloat16)
                        .view(tuple(t.shape))
                        .float()
                    )

                errors = []
                for actual, ref in [
                    (qo[:, : qh * 128].view(tokens, qh, 128).float().cpu(), read(rq)),
                    (ko[:, : kh * 128].view(tokens, kh, 128).float().cpu(), read(rk)),
                ]:
                    errors.append(
                        float(
                            (
                                (actual - ref).square().mean() / ref.square().mean()
                            ).sqrt()
                        )
                    )
                assert max(errors) < 0.001, (tokens, errors)
                if pad:
                    assert (qo[:, -pad:] == 7).all() and (ko[:, -pad:] == 7).all()
                row = dict(
                    tokens=tokens,
                    q_heads=qh,
                    k_heads=kh,
                    padding=pad,
                    repeat=repeat,
                    nrms=max(errors),
                )
                rows.append(row)
                print("PASS", row, flush=True)
    print("ALL_FUSED_QK_NUMERICS_PASSED", flush=True)
    if len(__import__("sys").argv) > 1:
        Path(__import__("sys").argv[1]).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
