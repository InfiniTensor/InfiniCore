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

The current-bool variant passed C500 FP16/BF16 Prefill and Decode reference
checks. Older-wheel execution was not available. Hardware conditions, model
results and archived backend comparisons are in
[PR #1558](https://github.com/InfiniTensor/InfiniCore/pull/1558). The fix restores
loading of affected wheels; it does not introduce a faster attention kernel.
