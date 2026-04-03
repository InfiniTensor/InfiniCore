# InfLLM-v2 / FlashAttention Local-Window Patch Notes

## What this note covers
This repo includes a small set of non-upstream changes to the FlashAttention kernel launch templates vendored under:

`sglang/3rdparty/infllmv2_cuda_impl/csrc/flash_attn/src/`

The goal is to ensure that when the InfLLM-v2 attention wrapper provides `params.window_size_left` / `params.window_size_right`, the FlashAttention kernel is actually launched in **local-window mode** (`Is_local=true`) instead of being hard-disabled.

## Patch summary (behavioral intent)
- When `window_size_left >= 0` or `window_size_right >= 0` AND the request is **non-causal**, select the FlashAttention `Is_local` specialization path.
- When `Is_local` is chosen, `Is_causal` must be treated as `false` (local-window causal-local semantics).

Concretely, the template uses the predicate:

```cpp
// forward:
(params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal

// backward:
(params.window_size_left >= 0 || params.window_size_right >= 0) && !params.is_causal
```

## Files changed

### 1) Forward launch templates
`sglang/3rdparty/infllmv2_cuda_impl/csrc/flash_attn/src/flash_fwd_launch_template.h`

Changes:
- Replace the previous hard-disabled `Is_local` selection with `LOCAL_SWITCH(...)` so `Is_local` can become `true` when window sizes are provided and the call is non-causal.
- The `LOCAL_SWITCH(...)` predicate appears in multiple forward paths (standard forward + splitkv variants + stage1).

Key pattern (appears multiple times in this file):
```cpp
LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
    // Is_local selected by LOCAL_SWITCH.
    ...
    auto kernel = &flash_fwd_kernel<... Is_causal, Is_local && !Is_causal, ...>;
});
```

Why this matters:
- FlashAttention masking uses `Is_local` to apply causal-local / left-window masking semantics.
- `flash_fwd_kernel` internally references `params.window_size_left` for block-range selection when `Is_local` is enabled.

### 2) Backward launch templates
`sglang/3rdparty/infllmv2_cuda_impl/csrc/flash_attn/src/flash_bwd_launch_template.h`

Changes:
- Enable `Is_local` specialization selection with `LOCAL_SWITCH(...)` using the same window-size predicate, but based on `params.is_causal` for backward.

Key pattern:
```cpp
LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !params.is_causal, Is_local, [&] {
    // Is_local selected by LOCAL_SWITCH.
    ...
    auto kernel = &flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<..., Is_causal, Is_local && !Is_causal, ...>;
});
```

## Compatibility / safety notes
- `flash_attn/src/mask.h` enforces that the kernel cannot be both causal and local simultaneously (there is a static assert guarding `Is_causal && Is_local`).
- The launch-template logic above ensures local mode is only selected under `causal=false`, aligning with that constraint.

## Rationale (why we patch this)
Without this change, even if InfLLM-v2 passes `window_size_left/right`, the FlashAttention kernels remain in a non-local configuration, so left-window limiting does not take effect. Re-enabling `Is_local` selection allows prefill TTFT scaling to reflect the intended windowed/local computation path.

