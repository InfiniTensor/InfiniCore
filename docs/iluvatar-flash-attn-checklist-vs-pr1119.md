# Iluvatar Flash-Attention port — checklist vs [PR #1119](https://github.com/InfiniTensor/InfiniCore/pull/1119)

This document mirrors the **MetaX flash-attn integration** merged in PR #1119 and tracks **Iluvatar (`ENABLE_ILUVATAR_API`)** work against it. Checkbox state reflects the **current working tree** as compared to `origin/main` (see *How this list is toggled*).

**Reference:** [InfiniTensor/InfiniCore#1119 — issue/1117: metax support flash-attn](https://github.com/InfiniTensor/InfiniCore/pull/1119)

---

## How this list is toggled

Regenerate or verify locally:

```bash
cd /path/to/InfiniCore
git fetch origin
git diff origin/main --name-only
git diff origin/main
git status -sb
```

- **[x]** — present in `git diff` vs `HEAD` (or satisfied by unchanged upstream already on `main`).
- **[ ]** — not addressed in the current diff, out of sync with `include/`, or a regression in `git status`.

*Last aligned to workspace snapshot: 2026-05-14.*

---

## A. PR #1119 themes → Iluvatar parity (code + build)

| Status | Item |
| :---: | --- |
| [x] | **ATen / CUDA-style interop** — `aten_adaptor.hpp` / `aten_adaptor.cc`: `ENABLE_ILUVATAR_API` joins NVIDIA/MetaX/QY for CUDA ATen headers, `get_cuda_stream()`, and `Device::Type::ILUVATAR` → `at::kCUDA` in `to_at_device()`. |
| [x] | **Flash symbol layout** — `flash_attention_adaptor.hpp`: omit `namespace flash { … }` when `ENABLE_ILUVATAR_API` (pip `flash_attn_2_cuda` global symbols), same pattern as MetaX. |
| [x] | **Varlen ABI** — `flash_attention_adaptor.hpp` + `mha_varlen_flashattn.cc`: Iluvatar-specific extra `mha_varlen_fwd` parameters (`deterministic`, `sm_margin`, `max_seqlen_k_new`) and call-site literals; analogous to MetaX HPCC version-guarded extras in PR #1119. |
| [x] | **KV-cache flash path** — `mha_kvcache_flashattn.cc`: `INFINICORE_FLASH_OP` → `::name` for Iluvatar; stream guard + `k_cache`/`v_cache` ATen paths include Iluvatar. |
| [x] | **Varlen flash path** — `mha_varlen_flashattn.cc`: global-scope op macro for Iluvatar; `CUDAStreamGuard` gated consistently with other CUDA-like backends; extra args wired for Iluvatar. |
| [x] | **xmake: vendor flash-attn target** — `xmake/iluvatar.lua`: `flash-attn-iluvatar` phony target; resolve `flash_attn_2_cuda*.so` via `--flash-attn`, `FLASH_ATTN_2_CUDA_SO`, or container path. |
| [x] | **xmake: link extension into `infinicore_cpp_api`** — `xmake.lua`: `iluvatar-gpu` adds `flash-attn-iluvatar` dep, discovers `flash_attn_2_cuda*.so`, `-Wl,--no-as-needed`, `-Wl,-rpath,<dir>`. |
| [x] | **Torch / ABI alignment** — `xmake.lua`: `_GLIBCXX_USE_CXX11_ABI` taken from the same Python `torch` for `infinicore_cpp_api` and `_infinicore` (`before_build`). |
| [x] | **Include paths for ATen + runtime** — `xmake.lua`: `CUDA_HOME` / `COREX_HOME` (and `targets/x86_64-linux/include`) for headers such as `c10/cuda/CUDAStream.h`. |

---

## B. Follow-ups from PR #1119 commits (not required for Iluvatar but worth tracking)

These showed up in PR #1119’s commit history; they are **not** part of the current Iluvatar `git diff` unless noted.

| Status | Item |
| :---: | --- |
| [ ] | **`xmake` `-y` / non-interactive installs** — PR #1119 added `add -y in xmake`; no matching hunk in current Iluvatar diff. |
| [ ] | **Reduce `contiguous()` use** — PR #1119 had a “reduce contiguous” commit; not reflected in current Iluvatar flash-attn diff. |
| [ ] | **HPCC / vendor-specific include fixes** — PR #1119 “fix hpcc include”; Iluvatar uses CoreX/CUDA paths instead; confirm no equivalent header gap remains on target images. |

---

## C. Workspace hygiene (`git status` vs a clean PR)

| Status | Item |
| :---: | --- |
| [ ] | **`third_party` submodules intact** — `git status` shows **deleted** `third_party/nlohmann_json` and `third_party/spdlog` (unrelated to flash-attn; restore before merge unless intentional). |
| [ ] | **Python vendored headers match `include/`** — `python/infinicore/lib/` is **untracked** and `diff` reports **`flash_attention_adaptor.hpp` differs** from `include/infinicore/adaptor/`. Sync or regenerate so Python builds see the same declarations as C++. |
| [ ] | **No stray build artifacts in version control** — decide whether `python/infinicore/lib/` should be gitignored, generated in CI, or committed consistently. |

---

## D. Optional (typical PR quality bar; not inferred from current diff)

| Status | Item |
| :---: | --- |
| [ ] | **Runtime / CI** — job or doc that builds `iluvatar-gpu` + `aten` + `--flash-attn=…` and runs flash-attn MHA tests. |
| [ ] | **Contributor docs** — short “Iluvatar + flash-attn” section: env vars (`FLASH_ATTN_2_CUDA_SO`, `CUDA_HOME`, `COREX_HOME`), `--flash-attn`, Python `torch` matching ABI. |

---

## Quick summary

**Done in current diff:** adaptor + flash header ABI + both flash MHA ops + Iluvatar xmake wiring + Torch CXX11 ABI + CUDA/CoreX includes — i.e. the **core** of what PR #1119 did for MetaX, adapted for Iluvatar.

**Still open:** submodule deletions, Python `lib/` copy drift, PR #1119 housekeeping commits not ported, tests/docs/CI.
