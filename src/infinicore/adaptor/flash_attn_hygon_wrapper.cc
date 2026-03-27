#if defined(ENABLE_FLASH_ATTN) && defined(ENABLE_HYGON_API) && !defined(ENABLE_NVIDIA_API)

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <dlfcn.h>
#include <optional>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Function pointer types for the extern "C" functions exported by the DCU
// flash_attn shared library (built from flash-attention-cutlass-master).
// We resolve these at runtime via dlsym to avoid hard link-time dependency
// on the prebuilt .so (which requires libtorch_python.so).
// ---------------------------------------------------------------------------

using mha_fwd_kvcache_fn_t = std::vector<at::Tensor> (*)(
    at::Tensor &q,
    const at::Tensor &kcache,
    const at::Tensor &vcache,
    c10::optional<const at::Tensor> &k_,
    c10::optional<const at::Tensor> &v_,
    c10::optional<const at::Tensor> &seqlens_k_,
    c10::optional<const at::Tensor> &rotary_cos_,
    c10::optional<const at::Tensor> &rotary_sin_,
    c10::optional<const at::Tensor> &cache_batch_idx_,
    c10::optional<const at::Tensor> &leftpad_k_,
    c10::optional<at::Tensor> &block_table_,
    c10::optional<at::Tensor> &alibi_slopes_,
    c10::optional<at::Tensor> &out_,
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    bool is_rotary_interleaved,
    int num_splits,
    const c10::optional<at::Tensor> &s_aux_);

using mha_varlen_fwd_fn_t = std::vector<at::Tensor> (*)(
    at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    c10::optional<at::Tensor> &out_,
    const at::Tensor &cu_seqlens_q,
    const at::Tensor &cu_seqlens_k,
    c10::optional<at::Tensor> &seqused_k,
    c10::optional<const at::Tensor> &leftpad_k_,
    c10::optional<at::Tensor> &block_table_,
    c10::optional<at::Tensor> &alibi_slopes_,
    int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool return_softmax,
    c10::optional<at::Tensor> q_descale_,
    c10::optional<at::Tensor> k_descale_,
    c10::optional<at::Tensor> v_descale_,
    c10::optional<at::Generator> gen_,
    const c10::optional<at::Tensor> &s_aux_);

static void *resolve_symbol(const char *name) {
    void *sym = dlsym(RTLD_DEFAULT, name);
    if (sym) {
        return sym;
    }
    throw std::runtime_error(
        std::string("flash_attn symbol not found: ") + name +
        ". Ensure flash_attn_2_cuda is loaded before calling this function "
        "(e.g. import torch; import flash_attn_2_cuda).");
}

// ---------------------------------------------------------------------------
// Wrappers in the flash:: namespace.
// These match the signatures declared in
//   include/infinicore/adaptor/flash_attention_adaptor.hpp
// and bridge the namespace gap between InfiniCore and the DCU library.
// ---------------------------------------------------------------------------

namespace flash {

std::vector<at::Tensor>
mha_fwd_kvcache(at::Tensor &q,
                const at::Tensor &kcache,
                const at::Tensor &vcache,
                std::optional<const at::Tensor> &k_,
                std::optional<const at::Tensor> &v_,
                std::optional<const at::Tensor> &seqlens_k_,
                std::optional<const at::Tensor> &rotary_cos_,
                std::optional<const at::Tensor> &rotary_sin_,
                std::optional<const at::Tensor> &cache_batch_idx_,
                std::optional<const at::Tensor> &leftpad_k_,
                std::optional<at::Tensor> &block_table_,
                std::optional<at::Tensor> &alibi_slopes_,
                std::optional<at::Tensor> &out_,
                const float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                bool is_rotary_interleaved,
                int num_splits) {
    static auto fn = reinterpret_cast<mha_fwd_kvcache_fn_t>(
        resolve_symbol("mha_fwd_kvcache"));
    c10::optional<at::Tensor> s_aux = c10::nullopt;
    return fn(
        q, kcache, vcache,
        k_, v_, seqlens_k_,
        rotary_cos_, rotary_sin_, cache_batch_idx_, leftpad_k_,
        block_table_, alibi_slopes_, out_,
        softmax_scale, is_causal,
        window_size_left, window_size_right,
        softcap, is_rotary_interleaved, num_splits, s_aux);
}

std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,
               const at::Tensor &k,
               const at::Tensor &v,
               std::optional<at::Tensor> &out_,
               const at::Tensor &cu_seqlens_q,
               const at::Tensor &cu_seqlens_k,
               std::optional<at::Tensor> &seqused_k,
               std::optional<const at::Tensor> &leftpad_k_,
               std::optional<at::Tensor> &block_table_,
               std::optional<at::Tensor> &alibi_slopes_,
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool return_softmax,
               std::optional<at::Generator> gen_) {
    static auto fn = reinterpret_cast<mha_varlen_fwd_fn_t>(
        resolve_symbol("mha_varlen_fwd"));
    c10::optional<at::Tensor> q_descale = c10::nullopt;
    c10::optional<at::Tensor> k_descale = c10::nullopt;
    c10::optional<at::Tensor> v_descale = c10::nullopt;
    c10::optional<at::Tensor> s_aux = c10::nullopt;
    return fn(
        q, k, v, out_,
        cu_seqlens_q, cu_seqlens_k,
        seqused_k, leftpad_k_, block_table_, alibi_slopes_,
        max_seqlen_q, max_seqlen_k,
        p_dropout, softmax_scale, zero_tensors, is_causal,
        window_size_left, window_size_right,
        softcap, return_softmax,
        q_descale, k_descale, v_descale, gen_, s_aux);
}

// ---------------------------------------------------------------------------
// vllm_mha_varlen_fwd — same signature as mha_varlen_fwd but resolves a
// different symbol.  Used for prefill with paged KV cache on Hygon.
// ---------------------------------------------------------------------------

std::vector<at::Tensor>
vllm_mha_varlen_fwd(at::Tensor &q,
                    const at::Tensor &k,
                    const at::Tensor &v,
                    std::optional<at::Tensor> &out_,
                    const at::Tensor &cu_seqlens_q,
                    const at::Tensor &cu_seqlens_k,
                    std::optional<at::Tensor> &seqused_k,
                    std::optional<const at::Tensor> &leftpad_k_,
                    std::optional<at::Tensor> &block_table_,
                    std::optional<at::Tensor> &alibi_slopes_,
                    int max_seqlen_q,
                    const int max_seqlen_k,
                    const float p_dropout,
                    const float softmax_scale,
                    const bool zero_tensors,
                    bool is_causal,
                    int window_size_left,
                    int window_size_right,
                    const float softcap,
                    const bool return_softmax,
                    std::optional<at::Generator> gen_) {
    static auto fn = reinterpret_cast<mha_varlen_fwd_fn_t>(
        resolve_symbol("vllm_mha_varlen_fwd"));
    c10::optional<at::Tensor> q_descale = c10::nullopt;
    c10::optional<at::Tensor> k_descale = c10::nullopt;
    c10::optional<at::Tensor> v_descale = c10::nullopt;
    c10::optional<at::Tensor> s_aux = c10::nullopt;
    return fn(
        q, k, v, out_,
        cu_seqlens_q, cu_seqlens_k,
        seqused_k, leftpad_k_, block_table_, alibi_slopes_,
        max_seqlen_q, max_seqlen_k,
        p_dropout, softmax_scale, zero_tensors, is_causal,
        window_size_left, window_size_right,
        softcap, return_softmax,
        q_descale, k_descale, v_descale, gen_, s_aux);
}

// ---------------------------------------------------------------------------
// paged_attention — decode-only paged attention for Hygon.
// Signature: (out, q, k_cache, v_cache, scale, block_table, context_lens,
//             alibi_slopes, kv_cache_dtype, q_scale, k_scale, v_scale,
//             max_context_len, s_aux) -> void
// ---------------------------------------------------------------------------

using paged_attention_fn_t = void (*)(
    at::Tensor &out,
    at::Tensor &q,
    at::Tensor &k_cache,
    at::Tensor &v_cache,
    double scale,
    at::Tensor &block_table,
    at::Tensor &context_lens,
    const c10::optional<at::Tensor> &alibi_slopes,
    const std::string &kv_cache_dtype,
    const c10::optional<at::Tensor> &q_scale,
    const c10::optional<at::Tensor> &k_scale,
    const c10::optional<at::Tensor> &v_scale,
    int max_context_len,
    const c10::optional<at::Tensor> &s_aux);

void paged_attention(at::Tensor &out,
                     at::Tensor &q,
                     at::Tensor &k_cache,
                     at::Tensor &v_cache,
                     double scale,
                     at::Tensor &block_table,
                     at::Tensor &context_lens,
                     std::optional<at::Tensor> alibi_slopes,
                     const std::string &kv_cache_dtype,
                     std::optional<at::Tensor> q_scale,
                     std::optional<at::Tensor> k_scale,
                     std::optional<at::Tensor> v_scale,
                     int max_context_len) {
    static auto fn = reinterpret_cast<paged_attention_fn_t>(
        resolve_symbol("paged_attention"));
    c10::optional<at::Tensor> s_aux = c10::nullopt;
    fn(out, q, k_cache, v_cache, scale, block_table, context_lens,
       alibi_slopes, kv_cache_dtype, q_scale, k_scale, v_scale,
       max_context_len, s_aux);
}

} // namespace flash

#endif // ENABLE_FLASH_ATTN && ENABLE_HYGON_API && !ENABLE_NVIDIA_API
