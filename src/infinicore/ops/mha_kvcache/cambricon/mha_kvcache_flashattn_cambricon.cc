#if defined(ENABLE_CAMBRICON_API) && defined(ENABLE_FLASH_ATTN)

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/cambricon_flash_attention.hpp"
#include "infinicore/ops/mha_kvcache.hpp"

#include <stdexcept>
#include <vector>

namespace infinicore::op::mha_kvcache_impl::cambricon {

using adaptor::cambricon_flash_attn::gather_paged_cache;
using adaptor::cambricon_flash_attn::to_host_i32;

struct PlannedMeta {
    graph::GraphTensor out, q, k_cache, v_cache, seqlens_k, block_table;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k_cache,
           const Tensor &v_cache,
           const Tensor &seqlens_k,
           const Tensor &block_table,
           std::optional<Tensor> alibi_slopes,
           float scale) {
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k_cache),
        graph::GraphTensor(v_cache),
        graph::GraphTensor(seqlens_k),
        graph::GraphTensor(block_table),
        alibi_slopes
            ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes))
            : std::nullopt,
        scale};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    torch_mlu::mlu::MLUStreamGuard guard(
        infinicore::adaptor::get_mlu_stream());
    auto q = infinicore::adaptor::to_aten_tensor(p->q).contiguous();
    auto k_cache = infinicore::adaptor::to_aten_tensor(p->k_cache);
    auto v_cache = infinicore::adaptor::to_aten_tensor(p->v_cache);
    auto seqlens_tensor
        = infinicore::adaptor::to_aten_tensor(p->seqlens_k);
    auto block_table
        = infinicore::adaptor::to_aten_tensor(p->block_table);
    auto lengths = to_host_i32(seqlens_tensor);

    if (q.dim() != 4
        || static_cast<size_t>(q.size(0)) != lengths.size()) {
        throw std::runtime_error(
            "Cambricon MHA KV-cache expects q in BSND layout");
    }
    auto packed = gather_paged_cache(
        k_cache, v_cache, block_table, lengths);

    const int64_t batch_size = q.size(0);
    const int64_t query_length = q.size(1);
    std::vector<int32_t> cu_q(batch_size + 1, 0);
    std::vector<int32_t> cu_k(batch_size + 1, 0);
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        cu_q[batch + 1]
            = cu_q[batch] + static_cast<int32_t>(query_length);
        cu_k[batch + 1] = cu_k[batch] + lengths[batch];
    }
    auto int_options = q.options().dtype(at::kInt);
    auto cu_seqlens_q = at::tensor(cu_q, int_options);
    auto cu_seqlens_k = at::tensor(cu_k, int_options);

    const bool copy_back = !p->out->is_contiguous();
    Tensor out_work_ic
        = copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_tensor
        = infinicore::adaptor::to_aten_tensor(out_work_ic);
    auto out_varlen = out_tensor.reshape(
        {batch_size * query_length, q.size(2), q.size(3)});
    auto out = std::optional<at::Tensor>(out_varlen);
    auto q_varlen = q.reshape(
        {batch_size * query_length, q.size(2), q.size(3)});
    std::optional<at::Tensor> seqused_k = std::nullopt;
    auto alibi = p->alibi_slopes
                   ? std::optional<at::Tensor>(
                       infinicore::adaptor::to_aten_tensor(*p->alibi_slopes))
                   : std::nullopt;

    ::mha_varlen_fwd(
        q_varlen, packed.key, packed.value, out,
        cu_seqlens_q, cu_seqlens_k, seqused_k, alibi,
        static_cast<int>(query_length), packed.max_seqlen,
        0.0F, p->scale, false, true, -1, -1, false, std::nullopt);

    if (copy_back) {
        p->out->copy_from(out_work_ic);
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MhaKVCache::plan_dispatcher().registerDevice(
        Device::Type::CAMBRICON, &plan);
    MhaKVCache::run_dispatcher().registerDevice(
        Device::Type::CAMBRICON, &run);
    MhaKVCache::cleanup_dispatcher().registerDevice(
        Device::Type::CAMBRICON, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_kvcache_impl::cambricon

#endif
