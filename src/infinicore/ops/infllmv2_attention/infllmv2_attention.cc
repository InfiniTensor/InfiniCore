/**
 * InfLLM-V2 attention ops (varlen + kvcache).
 * - With ENABLE_FLASH_ATTN: uses mha_varlen / mha_kvcache (Flash-style) fallback.
 * - With ENABLE_INFLLMV2 + ENABLE_ATEN: calls InfLLM-V2 C++ API (mha_varlen_fwd, mha_fwd_kvcache).
 *   Build InfiniCore with: xmake config --aten=y --infllmv2=/path/to/infllmv2_cuda_impl/build/lib.*
 *   (or full path to the .so). Link is done in xmake via ldflags to that .so.
 */
#include "infinicore/ops/infllmv2_attention.hpp"

#include "../../utils.hpp"
#include <fstream>

#if defined(ENABLE_INFLLMV2) && defined(ENABLE_ATEN)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/ops/infllmv2_api.hpp"
#ifdef ENABLE_NVIDIA_API
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#endif
#elif defined(ENABLE_FLASH_ATTN)
#include "infinicore/adaptor/flash_attention_adaptor.hpp"
#include "infinicore/ops/mha_kvcache.hpp"
#include "infinicore/ops/mha_varlen.hpp"
#endif

namespace infinicore::op {

namespace {
#if defined(ENABLE_INFLLMV2) && defined(ENABLE_ATEN)
inline void maybe_log_kvcache_inputs(const char *op_name,
                                     const at::Tensor &q,
                                     const at::Tensor &kcache,
                                     const at::Tensor &vcache,
                                     const at::Tensor &seqlens_k,
                                     bool causal,
                                     float scale) {
    const char *log_path = std::getenv("INFINI_DEBUG_LOG");
    const char *flag = std::getenv("INFINICORE_INFLLMV2_DUMP_ATEN");
    if (!log_path || !flag || flag[0] == '\0' || flag[0] == '0') {
        return;
    }
    try {
        std::ofstream f(log_path, std::ios::app);
        if (!f) {
            return;
        }
        auto cpu_lens = seqlens_k.to(at::kCPU);
        int32_t len0 = cpu_lens.numel() > 0 ? cpu_lens.data_ptr<int32_t>()[0] : -1;
        f << "[infinicore][infllmv2][" << op_name << "]"
          << " q=" << q.sizes()
          << " kcache=" << kcache.sizes()
          << " vcache=" << vcache.sizes()
          << " seqlens_k=" << seqlens_k.sizes()
          << " seqlens0=" << len0
          << " causal=" << (causal ? 1 : 0)
          << " scale=" << scale
          << " q_stride=" << q.strides()
          << " k_stride=" << kcache.strides()
          << " v_stride=" << vcache.strides()
          << "\n";
    } catch (...) {
    }
}
#else
inline void maybe_log_kvcache_inputs(const char * /*op_name*/,
                                     ...) {
    // no-op when ATen is not enabled
}
#endif
} // namespace

Tensor infllmv2_varlen(const Tensor &q,
                       const Tensor &k,
                       const Tensor &v,
                       const Tensor &cu_seqlens_q,
                       const Tensor &cu_seqlens_k,
                       int max_seqlen_q,
                       int max_seqlen_k,
                       float scale,
                       bool causal,
                       int window_size_left,
                       int window_size_right) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, k, v, cu_seqlens_q, cu_seqlens_k);

#if defined(ENABLE_INFLLMV2) && defined(ENABLE_ATEN)
    // Direct InfLLM-V2 kernels (link against infllmv2_cuda_impl).
    const auto &shape = q->shape();
    auto out = Tensor::empty(shape, q->dtype(), q->device());

#ifdef ENABLE_NVIDIA_API
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    auto q_at = infinicore::adaptor::to_aten_tensor(q);
    auto k_at = infinicore::adaptor::to_aten_tensor(k);
    auto v_at = infinicore::adaptor::to_aten_tensor(v);
    auto cu_q_at = infinicore::adaptor::to_aten_tensor(cu_seqlens_q);
    auto cu_k_at = infinicore::adaptor::to_aten_tensor(cu_seqlens_k);
    auto out_at = std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(out));

    c10::optional<at::Tensor> seqused_k = c10::nullopt;
    c10::optional<const at::Tensor> leftpad_k = c10::nullopt;
    c10::optional<at::Tensor> block_table = c10::nullopt;
    c10::optional<at::Tensor> alibi_slopes = c10::nullopt;
    c10::optional<at::Generator> gen_ = c10::nullopt;
    c10::optional<at::Tensor> blockmask_ = c10::nullopt;

    mha_varlen_fwd(
        q_at,
        k_at,
        v_at,
        out_at,
        cu_q_at,
        cu_k_at,
        seqused_k,
        leftpad_k,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        0.0f,
        scale,
        false,
        causal,
        window_size_left,
        window_size_right,
        0.0f,
        false,
        gen_,
        blockmask_);
    return out;

#elif defined(ENABLE_FLASH_ATTN)
    // Fallback: FlashAttention-based varlen op (same kernel family as InfLLM-V2).
    auto dummy_block_table = infinicore::Tensor::zeros(
        {cu_seqlens_q->shape()[0] - 1, 1},
        cu_seqlens_q->dtype(),
        cu_seqlens_q->device());
    (void)causal;
    (void)window_size_left;
    (void)window_size_right;
    return infinicore::op::mha_varlen(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        dummy_block_table,
        max_seqlen_q,
        max_seqlen_k,
        std::nullopt,
        scale);
#else
    (void)k;
    (void)v;
    (void)cu_seqlens_q;
    (void)cu_seqlens_k;
    (void)max_seqlen_q;
    (void)max_seqlen_k;
    (void)scale;
    (void)causal;
    throw std::runtime_error(
        "InfLLM-V2 varlen attention requires ENABLE_INFLLMV2+ENABLE_ATEN or ENABLE_FLASH_ATTN build");
#endif
}

Tensor infllmv2_kvcache(const Tensor &q,
                        const Tensor &k_cache,
                        const Tensor &v_cache,
                        const Tensor &cache_lens,
                        float scale,
                        bool causal,
                        int window_size_left,
                        int window_size_right) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, k_cache, v_cache, cache_lens);

#if defined(ENABLE_INFLLMV2) && defined(ENABLE_ATEN)
    // Direct InfLLM-V2 fwd_kvcache (link against infllmv2_cuda_impl).
    const auto &shape = q->shape();
    auto out = Tensor::empty(shape, q->dtype(), q->device());

#ifdef ENABLE_NVIDIA_API
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    auto q_at = infinicore::adaptor::to_aten_tensor(q);
    auto kcache_at = infinicore::adaptor::to_aten_tensor(k_cache);
    auto vcache_at = infinicore::adaptor::to_aten_tensor(v_cache);
    auto seqlens_k_at = std::optional<const at::Tensor>(infinicore::adaptor::to_aten_tensor(cache_lens));
    auto out_at = std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(out));

    // Keep BF16 compute in the kernel (no BF16->FP16 cast).
    at::Tensor out_kernel = out_at.value();

    c10::optional<const at::Tensor> k_new = c10::nullopt;
    c10::optional<const at::Tensor> v_new = c10::nullopt;
    c10::optional<const at::Tensor> rotary_cos = c10::nullopt;
    c10::optional<const at::Tensor> rotary_sin = c10::nullopt;
    c10::optional<const at::Tensor> cache_batch_idx = c10::nullopt;
    c10::optional<const at::Tensor> leftpad_k = c10::nullopt;
    c10::optional<at::Tensor> block_table = c10::nullopt;
    c10::optional<at::Tensor> alibi_slopes = c10::nullopt;
    c10::optional<at::Tensor> blockmask_ = c10::nullopt;

    maybe_log_kvcache_inputs("kvcache", q_at, kcache_at, vcache_at, seqlens_k_at.value(), causal, scale);

    // Let FlashAttn/InfLLM-v2 allocate output internally. Passing an explicit out_ tensor
    // can interact badly with internal q reshapes in the seqlen_q==1 GQA fast path.
    c10::optional<at::Tensor> out_kernel_opt = c10::nullopt;
    auto outs = mha_fwd_kvcache(
        q_at,
        kcache_at,
        vcache_at,
        k_new,
        v_new,
        seqlens_k_at,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        leftpad_k,
        block_table,
        alibi_slopes,
        out_kernel_opt,
        scale,
        causal,
        window_size_left,
        window_size_right,
        0.0f,
        false, // is_rotary_interleaved (no rotary for MiniCPM-SALA minicpm4)
        0,
        blockmask_);
    out_kernel = outs[0];
    out_at.value().copy_(out_kernel);
    return out;

#elif defined(ENABLE_FLASH_ATTN)
    (void)causal;
    (void)window_size_left;
    (void)window_size_right;
    auto device = q->device();
    auto bs = cache_lens->shape()[0];
    auto one = infinicore::Tensor::ones({bs, 1}, cache_lens->dtype(), device);
    auto block_table = one;
    auto seqlens_k = cache_lens;
    return infinicore::op::mha_kvcache(
        q,
        k_cache,
        v_cache,
        seqlens_k,
        block_table,
        std::nullopt,
        scale);
#else
    (void)k_cache;
    (void)v_cache;
    (void)cache_lens;
    (void)scale;
    (void)causal;
    throw std::runtime_error(
        "InfLLM-V2 kvcache attention requires ENABLE_INFLLMV2+ENABLE_ATEN or ENABLE_FLASH_ATTN build");
#endif
}

Tensor infllmv2_kvcache_update(const Tensor &q,
                               const Tensor &k_cache,
                               const Tensor &v_cache,
                               const Tensor &k_new,
                               const Tensor &v_new,
                               const Tensor &cache_lens,
                               float scale,
                               bool causal,
                               int window_size_left,
                               int window_size_right) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, k_cache, v_cache, k_new, v_new, cache_lens);

#if defined(ENABLE_INFLLMV2) && defined(ENABLE_ATEN)
    const auto &shape = q->shape();
    auto out = Tensor::empty(shape, q->dtype(), q->device());

#ifdef ENABLE_NVIDIA_API
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    auto q_at = infinicore::adaptor::to_aten_tensor(q);
    auto kcache_at = infinicore::adaptor::to_aten_tensor(k_cache);
    auto vcache_at = infinicore::adaptor::to_aten_tensor(v_cache);
    auto knew_at = infinicore::adaptor::to_aten_tensor(k_new);
    auto vnew_at = infinicore::adaptor::to_aten_tensor(v_new);
    auto seqlens_k_at = std::optional<const at::Tensor>(infinicore::adaptor::to_aten_tensor(cache_lens));
    auto out_at = std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(out));

    // Keep BF16 compute in the kernel (no BF16->FP16 cast).
    at::Tensor out_kernel = out_at.value();

    c10::optional<const at::Tensor> k_new_opt = std::optional<const at::Tensor>(knew_at);
    c10::optional<const at::Tensor> v_new_opt = std::optional<const at::Tensor>(vnew_at);
    c10::optional<const at::Tensor> rotary_cos = c10::nullopt;
    c10::optional<const at::Tensor> rotary_sin = c10::nullopt;
    c10::optional<const at::Tensor> cache_batch_idx = c10::nullopt;
    c10::optional<const at::Tensor> leftpad_k = c10::nullopt;
    c10::optional<at::Tensor> block_table = c10::nullopt;
    c10::optional<at::Tensor> alibi_slopes = c10::nullopt;
    c10::optional<at::Tensor> blockmask_ = c10::nullopt;

    maybe_log_kvcache_inputs("kvcache_update", q_at, kcache_at, vcache_at, seqlens_k_at.value(), causal, scale);

    c10::optional<at::Tensor> out_kernel_opt = c10::nullopt;
    auto outs = mha_fwd_kvcache(
        q_at,
        kcache_at,
        vcache_at,
        k_new_opt,
        v_new_opt,
        seqlens_k_at,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        leftpad_k,
        block_table,
        alibi_slopes,
        out_kernel_opt,
        scale,
        causal,
        window_size_left,
        window_size_right,
        0.0f,
        false,
        0,
        blockmask_);
    out_kernel = outs[0];
    out_at.value().copy_(out_kernel);
    return out;

#elif defined(ENABLE_FLASH_ATTN)
    (void)k_new;
    (void)v_new;
    // FlashAttn adaptor path currently doesn't support in-place cache update in this wrapper.
    // Fall back to normal kvcache (expects cache already updated by caller).
    return infllmv2_kvcache(
        q, k_cache, v_cache, cache_lens, scale, causal, window_size_left, window_size_right);
#else
    (void)k_cache;
    (void)v_cache;
    (void)k_new;
    (void)v_new;
    (void)cache_lens;
    (void)scale;
    (void)causal;
    (void)window_size_left;
    (void)window_size_right;
    throw std::runtime_error(
        "InfLLM-V2 kvcache_update attention requires ENABLE_INFLLMV2+ENABLE_ATEN build");
#endif
}

} // namespace infinicore::op
