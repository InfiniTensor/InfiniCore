import time

import torch
from infinicore.lib import _infinicore

import infinicore


def ic(tensor):
    return infinicore.from_torch(tensor)


def raw(tensor):
    return ic(tensor)._underlying


def sync():
    infinicore.sync_device()


def produce(cache, vendor_cache, compressed_kv, norm_weight, rope, slots):
    _infinicore.fp8_mla_rmsnorm_dual_cache_(
        raw(cache),
        raw(vendor_cache),
        raw(compressed_kv),
        raw(norm_weight),
        raw(rope),
        raw(slots),
        1e-5,
    )
    sync()


def bench(fn, warmup=5, iterations=20):
    for _ in range(warmup):
        fn()
    sync()
    begin = time.perf_counter()
    for _ in range(iterations):
        fn()
    sync()
    return (time.perf_counter() - begin) * 1000.0 / iterations


def validate_dual_cache():
    torch.manual_seed(7)
    num_tokens = 64
    compressed_kv = torch.randn(num_tokens, 512, device="cuda", dtype=torch.bfloat16)
    norm_weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    rope = torch.randn(num_tokens, 64, device="cuda", dtype=torch.bfloat16)
    slots = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    cache = torch.zeros(1, 64, 656, device="cuda", dtype=torch.uint8)
    vendor_cache = torch.zeros(1, 64, 576, device="cuda", dtype=torch.bfloat16)
    produce(cache, vendor_cache, compressed_kv, norm_weight, rope, slots)

    entries = cache.view(-1, 656)[slots]
    latent_fp8 = entries[:, :512].contiguous().view(torch.float8_e4m3fn)
    scales = entries[:, 512:528].contiguous().view(torch.float32)
    expected_latent = (
        (latent_fp8.float().view(num_tokens, 4, 128) * scales.view(num_tokens, 4, 1))
        .reshape(num_tokens, 512)
        .to(torch.bfloat16)
    )
    expected_rope = entries[:, 528:].contiguous().view(torch.bfloat16)
    vendor_entries = vendor_cache.view(-1, 576)[slots]
    torch.testing.assert_close(vendor_entries[:, :512], expected_latent, rtol=0, atol=0)
    torch.testing.assert_close(vendor_entries[:, 512:], expected_rope, rtol=0, atol=0)
    print("dual cache byte semantics ok")

    num_heads = 64
    query = torch.randn(1, num_heads, 576, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(num_tokens, device="cuda", dtype=torch.int32).view(
        1, 1, num_tokens
    )
    lens = torch.tensor([num_tokens], device="cuda", dtype=torch.int32)
    output_fp8 = torch.empty(1, num_heads, 512, device="cuda", dtype=torch.bfloat16)
    output_vendor = torch.empty_like(output_fp8)
    scale = float(576**-0.5)
    infinicore.sparse_flash_mla_(
        ic(output_fp8),
        ic(query),
        ic(cache.view(-1, 1, 656)),
        ic(indices),
        ic(lens),
        scale,
    )
    infinicore.sparse_flash_mla_(
        ic(output_vendor),
        ic(query),
        ic(vendor_cache.view(-1, 1, 576)),
        ic(indices),
        ic(lens),
        scale,
    )
    sync()
    diff = (output_fp8.float() - output_vendor.float()).abs()
    print(
        "sparse output diff:",
        f"max={diff.max().item():.6f}",
        f"mean={diff.mean().item():.6f}",
    )
    torch.testing.assert_close(output_vendor, output_fp8, rtol=0.05, atol=0.05)
    print("sparse FP8/vendor numerical comparison ok")


def benchmark_producer():
    torch.manual_seed(9)
    for num_tokens in (1, 4, 16):
        compressed_kv = torch.randn(
            num_tokens, 512, device="cuda", dtype=torch.bfloat16
        )
        norm_weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)
        rope = torch.randn(num_tokens, 64, device="cuda", dtype=torch.bfloat16)
        slots = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
        cache = torch.zeros(1, 64, 656, device="cuda", dtype=torch.uint8)
        vendor_cache = torch.zeros(1, 64, 576, device="cuda", dtype=torch.bfloat16)
        cache_raw = raw(cache)
        vendor_raw = raw(vendor_cache)
        compressed_raw = raw(compressed_kv)
        weight_raw = raw(norm_weight)
        rope_raw = raw(rope)
        slots_raw = raw(slots)

        def run_fp8():
            _infinicore.fp8_mla_rmsnorm_cache_(
                cache_raw,
                compressed_raw,
                weight_raw,
                rope_raw,
                slots_raw,
                1e-5,
            )

        def run_dual():
            _infinicore.fp8_mla_rmsnorm_dual_cache_(
                cache_raw,
                vendor_raw,
                compressed_raw,
                weight_raw,
                rope_raw,
                slots_raw,
                1e-5,
            )

        fp8_ms = bench(run_fp8, warmup=10, iterations=200)
        dual_ms = bench(run_dual, warmup=10, iterations=200)
        print(
            f"producer T={num_tokens}: fp8={fp8_ms:.4f}ms "
            f"dual={dual_ms:.4f}ms delta={dual_ms - fp8_ms:.4f}ms"
        )


def benchmark_sparse():
    torch.manual_seed(11)
    topk = 2048
    num_heads = 64
    compressed_kv = torch.randn(topk, 512, device="cuda", dtype=torch.bfloat16)
    norm_weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    rope = torch.randn(topk, 64, device="cuda", dtype=torch.bfloat16)
    slots = torch.arange(topk, device="cuda", dtype=torch.int64)
    cache = torch.zeros(32, 64, 656, device="cuda", dtype=torch.uint8)
    vendor_cache = torch.zeros(32, 64, 576, device="cuda", dtype=torch.bfloat16)
    produce(cache, vendor_cache, compressed_kv, norm_weight, rope, slots)
    scale = float(576**-0.5)

    for num_tokens in (1, 4, 16):
        query = torch.randn(
            num_tokens,
            num_heads,
            576,
            device="cuda",
            dtype=torch.bfloat16,
        )
        indices = (
            torch.arange(topk, device="cuda", dtype=torch.int32)
            .view(1, 1, topk)
            .expand(num_tokens, 1, topk)
            .contiguous()
        )
        lens = torch.full((num_tokens,), topk, device="cuda", dtype=torch.int32)
        output_fp8 = torch.empty(
            num_tokens,
            num_heads,
            512,
            device="cuda",
            dtype=torch.bfloat16,
        )
        output_vendor = torch.empty_like(output_fp8)

        def run_fp8():
            infinicore.sparse_flash_mla_(
                ic(output_fp8),
                ic(query),
                ic(cache.view(-1, 1, 656)),
                ic(indices),
                ic(lens),
                scale,
            )

        def run_vendor():
            infinicore.sparse_flash_mla_(
                ic(output_vendor),
                ic(query),
                ic(vendor_cache.view(-1, 1, 576)),
                ic(indices),
                ic(lens),
                scale,
            )

        fp8_ms = bench(run_fp8)
        vendor_ms = bench(run_vendor)
        print(
            f"T={num_tokens} K={topk}: fp8={fp8_ms:.3f}ms "
            f"vendor={vendor_ms:.3f}ms speedup={fp8_ms / vendor_ms:.2f}x"
        )


if __name__ == "__main__":
    validate_dual_cache()
    benchmark_producer()
    benchmark_sparse()
