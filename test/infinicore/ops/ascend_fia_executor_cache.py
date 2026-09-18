"""Small Ascend-only cache A/B regression, no model weights or profiling.

Use the freshly built libinfinicore_cpp_api via LD_LIBRARY_PATH, then run:
  ASCEND_RT_VISIBLE_DEVICES=0 python ascend_fia_executor_cache.py --output-dir DIR
Runs two fresh processes; leaves outputs and logs in DIR for inspection.
"""
import argparse
import ctypes
import os
from pathlib import Path
import subprocess
import sys


def exercise(output):
    import torch
    import infinicore as ic
    from infinicore.ops.mha_varlen import mha_varlen
    from infinicore.lib import _infinicore

    torch.set_num_threads(1)
    device = ic.device("npu", 0)
    cpu = ic.device("cpu", 0)

    def upload(tensor):
        result = ic.from_torch(tensor).to(device)
        # H2D is async: retain the Torch host buffer until the copy is complete.
        _infinicore.sync_stream()
        return result

    def download(tensor):
        # A synchronous ACL memcpy does not fence work on Core's custom stream.
        _infinicore.set_device(device._underlying)
        _infinicore.sync_stream()
        host = tensor.to(cpu)
        data = bytearray(ctypes.string_at(host.data_ptr(), host.numel() * 2))
        return torch.frombuffer(data, dtype=torch.bfloat16).clone()

    def cumulative(lengths):
        return upload(torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0).int())

    # A/B differ only in data and addresses; C/D change actual sequence values
    # with all tensor shapes and max-seqlen hints held constant. E reuses B's key.
    cases = [(1, [32]*16, [48]*16), (11, [32]*16, [48]*16),
             (11, [32]*16, [47]*16), (11, [31, 33]+[32]*14, [48]*16),
             (11, [32]*16, [48]*16)]
    retained, results = [], []
    for seed, qlens, klens in cases:
        torch.manual_seed(seed)
        q = upload(torch.randn(512, 16, 128).to(torch.bfloat16))
        k = upload(torch.randn(48, 16, 16, 128).to(torch.bfloat16))
        v = upload(torch.randn(48, 16, 16, 128).to(torch.bfloat16))
        bt = upload(torch.randperm(48).reshape(16, 3).int())
        sq, sk = cumulative(qlens), cumulative(klens)
        out = ic.empty((512, 16, 128), dtype=ic.bfloat16, device=device)
        mha_varlen(q, k, v, sq, sk, bt, 64, 64, scale=128**-0.5, out=out)
        results.append(download(out))
        # Keep old buffers alive, so missed rebinding cannot hide behind address reuse.
        retained.append((q, k, v, bt, sq, sk, out))
    for tensors, expected in zip(retained, results):
        assert torch.equal(download(tensors[-1]), expected), "wrote to an old output address"
    assert not torch.equal(results[0], results[1]), "different inputs produced identical outputs"
    assert not torch.equal(results[1], results[2]), "KV lengths ignored"
    assert not torch.equal(results[1], results[3]), "Q lengths ignored"
    assert torch.equal(results[1], results[4]), "same inputs did not reproduce"
    torch.save(results, output)
    print("PASS: address changes, Q/K sequence values, old-output sentinels")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--child-output", type=Path)
    args = parser.parse_args()
    if args.child_output:
        exercise(args.child_output)
        return
    if not args.output_dir:
        parser.error("--output-dir is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("0", "1"):
        env = dict(os.environ, INFINICORE_ASCEND_FIA_EXECUTOR_CACHE=mode,
                   INFINICORE_ASCEND_FIA_EXECUTOR_CACHE_STATS="1")
        env.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
        result = subprocess.run(
            [sys.executable, __file__, "--child-output", str(args.output_dir / f"{mode}.pt")],
            env=env, text=True, capture_output=True, timeout=90,
        )
        log = result.stdout + result.stderr
        (args.output_dir / f"{mode}.log").write_text(log)
        print(f"cache={mode}: exit={result.returncode}\n{log}", flush=True)
        assert result.returncode == 0, f"cache={mode} failed"
        expected = "hits=2 misses=3 entries=3" if mode == "1" else "hits=0 misses=0 entries=0"
        assert expected in log, f"cache counts incorrect; expected {expected}"
    import torch
    baseline = torch.load(args.output_dir / "0.pt", weights_only=True)
    cached = torch.load(args.output_dir / "1.pt", weights_only=True)
    assert all(torch.equal(a, b) for a, b in zip(baseline, cached)), "A/B mismatch"
    print("PASS: all five cached/one-shot outputs are bitwise identical; clean child exits")


if __name__ == "__main__":
    main()
