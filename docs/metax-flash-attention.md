# MetaX Flash Attention ABI selection

MetaX wheels can export different C++ signatures for `mha_varlen_fwd` even
within MACA 3.x. A missing trailing `bool` causes an undefined-symbol error
when loading InfiniCore, before any attention kernel can execute.

When building MetaX with Flash Attention, the build now inspects the actual
linked extension using `nm -D -C --defined-only`. It selects the declaration
and call for three known signatures: the original arguments, an additional
optional tensor, or that tensor plus `return_max_logit`. The last argument is
passed as `false` when present. NVIDIA and other device adapters are unchanged.

Use `FLASH_ATTN_2_CUDA_SO=/absolute/path/to/flash_attn_2_cuda.so` to select an
exact wheel. Otherwise the resolver checks `--flash-attn` for a unique
extension, then `FLASH_ATTN_METAX_CUDA_SO_CONTAINER`, then the Python environment.
Invalid explicit paths and unknown/missing/ambiguous signatures fail during
build with a diagnostic. The same resolver is used for compilation and linking.
MACA-version handling for the other Flash Attention entry points remains as
before; this change only detects the varlen ABI.

Run the detection tests without a GPU:

```sh
xmake lua tests/xmake/test_metax_flash_abi.lua
```

With `FLASH_ATTN_2_CUDA_SO` set, the test also checks the installed extension.
The three recognized signatures and missing/unknown/ambiguous negative cases
are covered by fixtures. Older-wheel execution was not available for testing.

The current-bool variant was built and executed on a C500 slice with MACA
3.5.3.20, torch 2.8.0+metax3.5.3.9 and flash-attn 2.6.3+metax3.5.3.9torch2.8.
Prefill and Decode each passed six FP16/BF16 cases against an explicit FP32
attention reference. Cases include GQA, head dimensions 64/128, noncontiguous
physical page mappings, history KV and causal masks. End-to-end Qwen3-0.6B/4B
also ran with the matching InfiniLM cache/chunk/graph integration candidate.
These are correctness checks, not a new attention-kernel performance claim.
