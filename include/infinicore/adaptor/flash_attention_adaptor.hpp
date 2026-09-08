#ifdef ENABLE_FLASH_ATTN
#pragma once
#include "aten_adaptor.hpp"

#if defined(ENABLE_METAX_API)

// MetaX flash-attn wheels declare their API in terms of c10::optional. On torch >= 2.1 that
// is an alias of std::optional (identical mangling); spelling the declarations the same way
// as the wheel guarantees identical mangling even on older torch stacks. When the c10 header
// no longer exists, torch has fully migrated to std::optional -- and so have wheels built
// against it.
#if __has_include(<c10/util/Optional.h>)
#include <c10/util/Optional.h>
#define INFINICORE_FA_OPTIONAL c10::optional
#else
#define INFINICORE_FA_OPTIONAL std::optional
#endif

// MetaX flash-attn ships two incompatible forward ABIs:
//   INFINICORE_METAX_FA_ABI 253 -- flash_attn 2.5.3 (MACA/HPCC 2.x):
//       mha_fwd / mha_varlen_fwd / mha_fwd_kvcache take 13 / 18 / 18 arguments
//   INFINICORE_METAX_FA_ABI 263 -- flash_attn 2.6.3+metax (MACA/HPCC 3.x):
//       the same functions take 16 / 23 / 21 arguments (softcap, leftpad_k, varlen
//       block_table, s_aux and return_max_logit are appended)
// xmake (xmake/metax.lua) injects INFINICORE_METAX_FA_ABI after inspecting the dynamic
// symbols of the actual `flash_attn_2_cuda` wheel that will be linked. When it is not
// injected (e.g. the wheel is absent at configure time), fall back to the HPCC/MACA
// toolkit major-version probe: >= 3 ships the 2.6.3 ABI, otherwise the legacy 2.5.3 ABI.
#if !defined(INFINICORE_METAX_FA_ABI)
#if defined(INFINICORE_HPCC_VERSION_MAJOR) && (INFINICORE_HPCC_VERSION_MAJOR >= 3)
#define INFINICORE_METAX_FA_ABI 263
#else
#define INFINICORE_METAX_FA_ABI 253
#endif
#endif
#define INFINICORE_METAX_FA263 (INFINICORE_METAX_FA_ABI >= 263)

#else // !ENABLE_METAX_API

#define INFINICORE_FA_OPTIONAL std::optional
#define INFINICORE_METAX_FA263 1

#endif // ENABLE_METAX_API

// NVIDIA flash-attn-nvidia.so uses namespace flash. The pip/MetaX flash_attn_2_cuda extension
// exports the same entry points at global scope (no namespace), matching FLASH_NAMESPACE builds
// where the namespace is empty.
//
// Ascend (aclnn C API path) does NOT use the flash:: namespace at all — the aclnn kernels are
// called directly from the dedicated *._ascend.cc implementation files, so this header is only
// included by the NVIDIA/MetaX/QY code paths.  We still guard the namespace below so that
// existing code compiles unchanged when ENABLE_ASCEND_FLASH_ATTN is defined.
#if !defined(ENABLE_METAX_API) && !defined(ENABLE_ASCEND_FLASH_ATTN)
namespace flash {
#endif
std::vector<at::Tensor>
mha_fwd(at::Tensor &q,                                     // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        const at::Tensor &k,                               // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        const at::Tensor &v,                               // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        INFINICORE_FA_OPTIONAL<at::Tensor> &out_,          // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        INFINICORE_FA_OPTIONAL<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
#if defined(ENABLE_METAX_API)
        INFINICORE_FA_OPTIONAL<at::Tensor> &attn_mask_,
#endif
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
#if !defined(ENABLE_METAX_API) || INFINICORE_METAX_FA263
        const float softcap,
#endif
        const bool return_softmax,
        INFINICORE_FA_OPTIONAL<at::Generator> gen_
#if defined(ENABLE_METAX_API) && INFINICORE_METAX_FA263
        // MetaX `flash_attn_2_cuda` 2.6.3+ (MACA/HPCC 3.x) appends these arguments vs the
        // 2.5.3 wheel and upstream Dao-AILab flash-attn.
        ,
        INFINICORE_FA_OPTIONAL<at::Tensor> &s_aux_,
        bool return_max_logit_
#endif
);

std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,                                 // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,                           // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               const at::Tensor &v,                           // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               INFINICORE_FA_OPTIONAL<at::Tensor> &out_,      // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,                // b+1
               const at::Tensor &cu_seqlens_k,                // b+1
               INFINICORE_FA_OPTIONAL<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
#if !defined(ENABLE_METAX_API) || INFINICORE_METAX_FA263
               INFINICORE_FA_OPTIONAL<const at::Tensor> &leftpad_k_, // batch_size
               INFINICORE_FA_OPTIONAL<at::Tensor> &block_table_,     // batch_size x max_num_blocks_per_seq
#endif
               INFINICORE_FA_OPTIONAL<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
#if !defined(ENABLE_METAX_API) || INFINICORE_METAX_FA263
               const float softcap,
#endif
               const bool return_softmax,
               INFINICORE_FA_OPTIONAL<at::Generator> gen_
#if defined(ENABLE_METAX_API) && INFINICORE_METAX_FA263
               // MetaX `flash_attn_2_cuda` 2.6.3+ (MACA/HPCC 3.x) appends these arguments vs the
               // 2.5.3 wheel and upstream Dao-AILab flash-attn.
               ,
               INFINICORE_FA_OPTIONAL<at::Tensor> &s_aux_,
               bool return_max_logit_
#endif
);

std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,                            // batch_size x seqlen_q x num_heads, x multiple_of(head_size_og, 8)
        const at::Tensor &q,                               // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,                               // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,                               // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,                             // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,                     // b x h x seqlen_q
        INFINICORE_FA_OPTIONAL<at::Tensor> &dq_,           // batch_size x seqlen_q x num_heads x head_size
        INFINICORE_FA_OPTIONAL<at::Tensor> &dk_,           // batch_size x seqlen_k x num_heads_k x head_size
        INFINICORE_FA_OPTIONAL<at::Tensor> &dv_,           // batch_size x seqlen_k x num_heads_k x head_size
        INFINICORE_FA_OPTIONAL<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,                             // probability to drop
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool deterministic,
        INFINICORE_FA_OPTIONAL<at::Generator> gen_,
        INFINICORE_FA_OPTIONAL<at::Tensor> &rng_state);

std::vector<at::Tensor>
mha_varlen_bwd(const at::Tensor &dout,                            // total_q x num_heads, x head_size
               const at::Tensor &q,                               // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,                               // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,                               // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &out,                             // total_q x num_heads x head_size
               const at::Tensor &softmax_lse,                     // h x total_q, softmax logsumexp
               INFINICORE_FA_OPTIONAL<at::Tensor> &dq_,           // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               INFINICORE_FA_OPTIONAL<at::Tensor> &dk_,           // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               INFINICORE_FA_OPTIONAL<at::Tensor> &dv_,           // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,                    // b+1
               const at::Tensor &cu_seqlens_k,                    // b+1
               INFINICORE_FA_OPTIONAL<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               const int max_seqlen_q,
               const int max_seqlen_k, // max sequence length to choose the kernel
               const float p_dropout,  // probability to drop
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool deterministic,
               INFINICORE_FA_OPTIONAL<at::Generator> gen_,
               INFINICORE_FA_OPTIONAL<at::Tensor> &rng_state);

std::vector<at::Tensor>
mha_fwd_kvcache(at::Tensor &q,                                              // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &kcache,                                   // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &vcache,                                   // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                INFINICORE_FA_OPTIONAL<const at::Tensor> &k_,               // batch_size x seqlen_knew x num_heads_k x head_size
                INFINICORE_FA_OPTIONAL<const at::Tensor> &v_,               // batch_size x seqlen_knew x num_heads_k x head_size
                INFINICORE_FA_OPTIONAL<const at::Tensor> &seqlens_k_,       // batch_size
                INFINICORE_FA_OPTIONAL<const at::Tensor> &rotary_cos_,      // seqlen_ro x (rotary_dim / 2)
                INFINICORE_FA_OPTIONAL<const at::Tensor> &rotary_sin_,      // seqlen_ro x (rotary_dim / 2)
                INFINICORE_FA_OPTIONAL<const at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
#if !defined(ENABLE_METAX_API) || INFINICORE_METAX_FA263
                INFINICORE_FA_OPTIONAL<const at::Tensor> &leftpad_k_, // batch_size
#endif
                INFINICORE_FA_OPTIONAL<at::Tensor> &block_table_,  // batch_size x max_num_blocks_per_seq
                INFINICORE_FA_OPTIONAL<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                INFINICORE_FA_OPTIONAL<at::Tensor> &out_,          // batch_size x seqlen_q x num_heads x head_size
                const float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
#if !defined(ENABLE_METAX_API) || INFINICORE_METAX_FA263
                const float softcap,
#endif
                bool is_rotary_interleaved, // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int num_splits
#if defined(ENABLE_METAX_API) && INFINICORE_METAX_FA263
                // MetaX `flash_attn_2_cuda` 2.6.3+ (MACA/HPCC 3.x) appends this argument vs the
                // 2.5.3 wheel and upstream Dao-AILab flash-attn.
                ,
                INFINICORE_FA_OPTIONAL<at::Tensor> &s_aux_
#endif
);

#if !defined(ENABLE_METAX_API) && !defined(ENABLE_ASCEND_FLASH_ATTN)
} // namespace flash
#endif
#endif // ENABLE_FLASH_ATTN
