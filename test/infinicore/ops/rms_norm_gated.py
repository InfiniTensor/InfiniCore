"""BF16 head norm/gate fusion, packed row strides and output padding."""

import ctypes, json, os, subprocess, tempfile
from pathlib import Path
import numpy as np
import torch, infinicore as ic
from infinicore.nn.functional import rms_norm, swiglu


def main():
    torch.manual_seed(81)
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
                str(Path(__file__).with_name("rms_norm_gated_bridge.cc")),
                "-L" + str(infini / "lib"),
                "-linfinicore_cpp_api",
                "-o",
                str(lib),
            ],
            check=True,
        )
        fn = ctypes.CDLL(str(lib)).test_rms_norm_gated
        fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5
        fn.restype = ctypes.c_int
        for tokens, heads, pad in [
            (1, 32, 0),
            (3, 3, 16),
            (5, 8, 0),
            (16, 32, 0),
            (32, 32, 16),
            (53, 32, 0),
            (1349, 32, 0),
            (8192, 32, 0),
        ]:
            width = heads * 128
            xs = width + pad
            zs = width + 64
            ys = width + pad
            x = torch.randn(tokens, xs, device="cuda", dtype=torch.bfloat16)
            z = torch.randn(tokens, zs, device="cuda", dtype=torch.bfloat16)
            w = (1 + torch.randn(128, device="cuda") * 0.05).bfloat16()
            y = torch.full((tokens, ys), 7, device="cuda", dtype=torch.bfloat16)
            ix = ic.from_torch(x).narrow(1, 0, width).view([tokens, heads, 128])
            iz = ic.from_torch(z).narrow(1, 0, width).view([tokens, heads, 128])
            ref = swiglu(rms_norm(ix, [128], ic.from_torch(w), 1e-6), iz).to(
                ic.device("cpu")
            )
            raw = ctypes.string_at(ref.data_ptr(), ref.numel() * 2)
            ref = (
                torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy())
                .view(torch.bfloat16)
                .float()
                .view(tokens, width)
            )
            ic.sync_device()
            torch.cuda.synchronize()
            for repeat in range(2):
                assert (
                    fn(*[t.data_ptr() for t in [y, x, z, w]], tokens, heads, xs, zs, ys)
                    == 0
                )
                nrms = float(
                    (
                        (y[:, :width].float().cpu() - ref).square().mean()
                        / ref.square().mean()
                    ).sqrt()
                )
                assert nrms < 0.001, (tokens, nrms)
                if pad:
                    assert (y[:, width:] == 7).all()
                row = dict(
                    tokens=tokens, heads=heads, padding=pad, repeat=repeat, nrms=nrms
                )
                rows.append(row)
                print("PASS", row, flush=True)
    if len(__import__("sys").argv) > 1:
        Path(__import__("sys").argv[1]).write_text(json.dumps(rows, indent=2))
    print("ALL_GATED_NORM_NUMERICS_PASSED", flush=True)


if __name__ == "__main__":
    main()
