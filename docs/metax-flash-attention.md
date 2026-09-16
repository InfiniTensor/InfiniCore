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

## Practical benefit and backend-selection experiment

The direct before/after result of this fix is availability: with the affected
wheel, the old declaration produces an undefined-symbol error when importing
InfiniCore; the matching declaration allows import, Prefill and Decode to run.
There is no meaningful latency ratio against a backend that could not load.

A separate archived experiment shows why restoring access to the existing
vendor implementation matters. Both arms used the **same ABI-compatible
prototype library**, switching only `attn_backend` between `paged-attn` and
`flash-attn`. This measures backend selection, not the speed of ABI detection
or a new kernel supplied by this PR.

Conditions: C500 50% compute / 32,000 MiB slice, six CPU cores; MACA 3.5.3.20,
driver 3.8.30 and the torch/Flash Attention versions above. Qwen3-0.6B and
Qwen3-4B, BF16 model/KV, TP1/PP1, eager, 32 paged KV blocks of 256 tokens,
prefix reuse off, chunking off. Each request has 2,048 input and 16 generated
tokens, greedy with EOS ignored for timing. Two measurements after warmup
per backend/model. TTFT is their mean; ITL is the mean of each request's
median token interval. All eight measured outputs match the HF eager
reference and the corresponding other-backend outputs.

| Model | Native TTFT ms | Vendor TTFT ms | TTFT reduction | Native ITL ms | Vendor ITL ms | ITL reduction |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 472.52 | 78.80 | 83.32% | 17.13 | 6.88 | 59.85% |
| Qwen3-4B | 1,212.21 | 212.09 | 82.50% | 26.00 | 12.87 | 50.50% |

Per-request output rates, including Prefill, were 21.93 -> 87.92 token/s
and 9.99 -> 39.46 token/s respectively. These are finite single-request
windows, not sustained service throughput. Two samples on a shared slice
do not establish statistical significance or support all model families.
The tested backend still uses InfiniLM's scheduling and KV ownership;
selecting vendor attention does not replace the framework's cache manager.

[Sanitized per-request measurements](validation/metax-backend-performance.json)
include original artifact hashes and the loaded binary hashes, verifying
that both arms used the same compatibility build. This experiment preceded
the final generalized signature resolver; the final resolver separately
passed six Prefill and six Decode operator cases plus the integrated model
checks. NVIDIA and other device adapters are unchanged.

The portable ABI fixture command above and the installed-library check are
the focused reproduction entrypoints for this PR. The two-sample backend
table documents the practical motivation; it is not a claim that the final
resolver was itself performance-benchmarked against the failed import.

[Results screenshot](validation/test-results.png): browser capture of the
audited saved-results report, not a new GPU run. The linked CPU CI result
does not replace MetaX hardware validation or upstream required checks.
