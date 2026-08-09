#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
#include "infinicore/ops/mha_varlen.hpp"

#ifdef ENABLE_INFINIOPS_LINKED_FLASH_ATTN_VARLEN_FUNC
#include "../infiniops_impl.hpp"

#include "base/flash_attn_varlen_func.h"

#include <cstdint>
#include <vector>
#endif

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ops/scaled_dot_product_attention.h>
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#ifdef ENABLE_FLASH_ATTN
#include "infinicore/adaptor/flash_attention_adaptor.hpp"
#endif

#include <stdexcept>

namespace infinicore::op::mha_varlen_impl::flashattn {
namespace {

#ifdef ENABLE_INFINIOPS_LINKED_FLASH_ATTN_VARLEN_FUNC
using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

bool canUseInfiniOps(const Tensor &out,
                     const Tensor &q,
                     const Tensor &k,
                     const Tensor &v,
                     const Tensor &cum_seqlens_q,
                     const Tensor &cum_seqlens_k,
                     const std::optional<Tensor> &block_table,
                     int max_seqlen_q,
                     int max_seqlen_k,
                     const std::optional<Tensor> &alibi_slopes) {
    const bool paged = block_table.has_value();
    const auto dtype = q->dtype();
    if (out->device().getType() != Device::Type::NVIDIA
        || q->ndim() != 3
        || out->ndim() != 3
        || ((paged && (k->ndim() != 4 || v->ndim() != 4))
            || (!paged && (k->ndim() != 3 || v->ndim() != 3)))
        || k->shape() != v->shape()
        || out->shape() != q->shape()
        || (dtype != DataType::F16 && dtype != DataType::BF16)
        || out->dtype() != dtype
        || k->dtype() != dtype
        || v->dtype() != dtype
        || q->size(1) == 0
        || k->size(k->ndim() - 2) == 0
        || q->size(1) % k->size(k->ndim() - 2) != 0
        || q->size(2) == 0
        || q->size(2) > 256
        || q->size(2) % 8 != 0
        || q->size(2) != k->size(k->ndim() - 1)
        || q->stride(2) != 1
        || out->stride(2) != 1
        || k->stride(k->ndim() - 1) != 1
        || v->stride(v->ndim() - 1) != 1
        || cum_seqlens_q->ndim() != 1
        || cum_seqlens_k->ndim() != 1
        || cum_seqlens_q->shape() != cum_seqlens_k->shape()
        || cum_seqlens_q->numel() < 2
        || cum_seqlens_q->dtype() != DataType::I32
        || cum_seqlens_k->dtype() != DataType::I32
        || !cum_seqlens_q->is_contiguous()
        || !cum_seqlens_k->is_contiguous()
        || max_seqlen_q <= 0
        || max_seqlen_k <= 0) {
        return false;
    }

    if (block_table
        && (block_table.value()->ndim() != 2
            || block_table.value()->size(0) + 1 != cum_seqlens_q->size(0)
            || block_table.value()->dtype() != DataType::I32
            || !block_table.value()->is_contiguous()
            || k->size(1) % 256 != 0)) {
        return false;
    }

    if (alibi_slopes
        && ((alibi_slopes.value()->ndim() != 1
             && alibi_slopes.value()->ndim() != 2)
            || alibi_slopes.value()->dtype() != DataType::F32
            || !alibi_slopes.value()->is_contiguous()
            || alibi_slopes.value()->device().getType() != out->device().getType()
            || alibi_slopes.value()->device().getIndex() != out->device().getIndex()
            || (alibi_slopes.value()->ndim() == 1
                && alibi_slopes.value()->size(0) != q->size(1))
            || (alibi_slopes.value()->ndim() == 2
                && (alibi_slopes.value()->size(0) + 1
                        != cum_seqlens_q->size(0)
                    || alibi_slopes.value()->size(1) != q->size(1))))) {
        return false;
    }

    return true;
}
#endif

} // namespace

struct PlannedMeta {
    graph::GraphTensor out, q, k, v, cum_seqlens_q, cum_seqlens_k;
    std::optional<graph::GraphTensor> block_table;
    int max_seqlen_q, max_seqlen_k;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
#ifdef ENABLE_INFINIOPS_LINKED_FLASH_ATTN_VARLEN_FUNC
    bool use_infiniops{false};
    std::optional<TensorMeta> infiniops_out, infiniops_q, infiniops_k,
        infiniops_v, infiniops_cum_seqlens_q, infiniops_cum_seqlens_k;
    std::optional<TensorMeta> infiniops_block_table, infiniops_alibi_slopes;
#endif
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           const Tensor &cum_seqlens_q,
           const Tensor &cum_seqlens_k,
           std::optional<Tensor> block_table,
           int max_seqlen_q,
           int max_seqlen_k,
           std::optional<Tensor> alibi_slopes,
           float scale) {

    auto planned = new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        graph::GraphTensor(cum_seqlens_q),
        graph::GraphTensor(cum_seqlens_k),
        block_table ? std::optional<graph::GraphTensor>(graph::GraphTensor(*block_table)) : std::nullopt,
        max_seqlen_q,
        max_seqlen_k,
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale};

#ifdef ENABLE_INFINIOPS_LINKED_FLASH_ATTN_VARLEN_FUNC
    planned->use_infiniops = canUseInfiniOps(
        out, q, k, v, cum_seqlens_q, cum_seqlens_k, block_table,
        max_seqlen_q, max_seqlen_k, alibi_slopes);
    if (planned->use_infiniops) {
        planned->infiniops_out.emplace(out);
        planned->infiniops_q.emplace(q);
        planned->infiniops_k.emplace(k);
        planned->infiniops_v.emplace(v);
        planned->infiniops_cum_seqlens_q.emplace(cum_seqlens_q);
        planned->infiniops_cum_seqlens_k.emplace(cum_seqlens_k);
        if (block_table) {
            planned->infiniops_block_table.emplace(*block_table);
        }
        if (alibi_slopes) {
            planned->infiniops_alibi_slopes.emplace(*alibi_slopes);
        }
    }
#endif

    return planned;
}

namespace {

#ifdef ENABLE_FLASH_ATTN
// MetaX/hpcc pip `flash_attn_2_cuda` exports `mha_varlen_fwd` at global scope (no namespace),
// while NVIDIA `flash-attn-nvidia.so` uses `flash::mha_varlen_fwd`.
#if defined(ENABLE_METAX_API)
#define INFINICORE_FLASH_OP(name) ::name
#else
#define INFINICORE_FLASH_OP(name) flash::name
#endif

#endif // ENABLE_FLASH_ATTN
} // namespace

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

#ifdef ENABLE_INFINIOPS_LINKED_FLASH_ATTN_VARLEN_FUNC
    if (p->use_infiniops) {
        infini::ops::Handle handle;
        handle.set_stream(context::getStream());
        infini::ops::Config config;
        config.set_implementation_index(16);

        const std::optional<infini::ops::Tensor> no_tensor;
        const std::optional<infini::ops::Tensor> block_table = p->block_table
                                                                 ? std::optional<infini::ops::Tensor>{
                                                                     p->infiniops_block_table->tensor(*p->block_table)}
                                                                 : std::nullopt;
        const std::optional<infini::ops::Tensor> alibi_slopes = p->alibi_slopes
                                                                  ? std::optional<infini::ops::Tensor>{
                                                                      p->infiniops_alibi_slopes->tensor(*p->alibi_slopes)}
                                                                  : std::nullopt;

        infini::ops::FlashAttnVarlenFunc::Call(
            handle,
            config,
            p->infiniops_q->tensor(p->q),
            p->infiniops_k->tensor(p->k),
            p->infiniops_v->tensor(p->v),
            p->infiniops_cum_seqlens_q->tensor(p->cum_seqlens_q),
            p->infiniops_cum_seqlens_k->tensor(p->cum_seqlens_k),
            alibi_slopes,
            block_table,
            static_cast<std::int64_t>(p->max_seqlen_q),
            static_cast<std::int64_t>(p->max_seqlen_k),
            0.0,
            std::optional<double>{p->scale},
            true,
            std::vector<std::int64_t>{-1, -1},
            0.0,
            false,
            false,
            p->infiniops_out->tensor(p->out),
            no_tensor,
            no_tensor);
        return;
    }
#endif

#if !defined(ENABLE_ATEN)
    (void)p;
    throw std::runtime_error("ATen is not enabled in this build");
#else
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    auto q = infinicore::adaptor::to_aten_tensor(p->q);
    auto k = infinicore::adaptor::to_aten_tensor(p->k);
    auto v = infinicore::adaptor::to_aten_tensor(p->v);

    const bool out_need_copy_back = !p->out->is_contiguous();
    Tensor out_work_ic = out_need_copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_work = infinicore::adaptor::to_aten_tensor(out_work_ic);

    auto cu_seqlens_q = infinicore::adaptor::to_aten_tensor(p->cum_seqlens_q);
    auto cu_seqlens_kv = infinicore::adaptor::to_aten_tensor(p->cum_seqlens_k);

    const bool dense_sdpa = !p->block_table.has_value()
                         && !p->alibi_slopes.has_value()
                         && q.dim() == 3 && k.dim() == 3 && v.dim() == 3
                         && p->max_seqlen_q > 0 && p->max_seqlen_k > 0
                         && p->max_seqlen_q == p->max_seqlen_k
                         && cu_seqlens_q.dim() == 1
                         && cu_seqlens_q.size(0) == cu_seqlens_kv.size(0)
                         && q.size(0) == (cu_seqlens_q.size(0) - 1) * p->max_seqlen_q
                         && k.size(0) == (cu_seqlens_kv.size(0) - 1) * p->max_seqlen_k
                         && ((q.size(2) > 256) || (v.size(2) != q.size(2)));
    if (dense_sdpa) {
        const int64_t batch_size = cu_seqlens_q.size(0) - 1;
        const int64_t seqlen = p->max_seqlen_q;
        const int64_t num_heads = q.size(1);
        const int64_t num_kv_heads = k.size(1);
        const int64_t head_dim = q.size(2);
        const int64_t value_dim = v.size(2);
        auto q_4d = q.reshape({batch_size, seqlen, num_heads, head_dim}).permute({0, 2, 1, 3});
        auto k_4d = k.reshape({batch_size, seqlen, num_kv_heads, head_dim}).permute({0, 2, 1, 3});
        auto v_4d = v.reshape({batch_size, seqlen, num_kv_heads, value_dim}).permute({0, 2, 1, 3});
        if (num_heads != num_kv_heads) {
            if (num_heads % num_kv_heads != 0) {
                throw std::runtime_error("mha_varlen dense SDPA fallback requires num_heads to be divisible by num_kv_heads");
            }
            const int64_t groups = num_heads / num_kv_heads;
            k_4d = k_4d.unsqueeze(2).expand({batch_size, num_kv_heads, groups, seqlen, head_dim}).reshape({batch_size, num_heads, seqlen, head_dim});
            v_4d = v_4d.unsqueeze(2).expand({batch_size, num_kv_heads, groups, seqlen, value_dim}).reshape({batch_size, num_heads, seqlen, value_dim});
        }
        auto result = at::scaled_dot_product_attention(
            q_4d,
            k_4d,
            v_4d,
            std::nullopt,
            0.0,
            true,
            std::optional<double>(static_cast<double>(p->scale)));
        out_work.copy_(result.permute({0, 2, 1, 3}).reshape({q.size(0), num_heads, value_dim}));
        if (out_need_copy_back) {
            p->out->copy_from(out_work_ic);
        }
        return;
    }

#ifdef ENABLE_FLASH_ATTN
    auto out = std::optional<at::Tensor>(out_work);
    std::optional<at::Tensor> seqused_k = std::nullopt;
    std::optional<const at::Tensor> leftpad_k = std::nullopt;
    auto block_table = p->block_table ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->block_table)) : std::nullopt;
    auto max_seqlen_q = p->max_seqlen_q;
    auto max_seqlen_k = p->max_seqlen_k;
    auto alibi_slopes = p->alibi_slopes ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->alibi_slopes)) : std::nullopt;
    auto scale = p->scale;

#if defined(ENABLE_METAX_API) && defined(INFINICORE_HPCC_VERSION_MAJOR) && (INFINICORE_HPCC_VERSION_MAJOR >= 3)
    std::optional<at::Tensor> flash_attn_mars_ext = std::nullopt;
#endif

    INFINICORE_FLASH_OP(mha_varlen_fwd)
    (
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_k,
        leftpad_k,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        scale,
        false,
        true,
        -1,
        -1,
        0.0,
        false,
        std::nullopt
#if defined(ENABLE_METAX_API) && defined(INFINICORE_HPCC_VERSION_MAJOR) && (INFINICORE_HPCC_VERSION_MAJOR >= 3)
        ,
        flash_attn_mars_ext
#endif
    );

    if (out_need_copy_back) {
        p->out->copy_from(out_work_ic);
    }

#else
    throw std::runtime_error("FlashAttention is not enabled in this build and dense SDPA fallback is not applicable");
#endif
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(MultiheadAttentionVarlen, &plan, &run, &cleanup);

} // namespace infinicore::op::mha_varlen_impl::flashattn
#endif
