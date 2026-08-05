#if defined(ENABLE_ILUVATAR_VENDOR_OPS)
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/iluvatar_vendor_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::paged_attention_mla_impl::iluvatar {

void run(Tensor output,
         const Tensor &query,
         const Tensor &kv_cache,
         float scale,
         const Tensor &block_tables,
         const Tensor &context_lens,
         int64_t max_context_len) {
    if (block_tables->dtype() != DataType::I32) {
        throw std::runtime_error("paged_attention_mla Iluvatar implementation expects int32 block_tables");
    }
    if (!adaptor::iluvatar_vendor::paged_attention_mla_available()) {
        throw std::runtime_error("paged_attention_mla requires the Iluvatar vendor extension");
    }
    auto softmax_lse = Tensor::empty(
        {query->size(0), query->size(1)}, DataType::F32, query->device());
    auto output_at = adaptor::to_aten_tensor(output);
    auto query_at = adaptor::to_aten_tensor(query);
    auto kv_cache_at = adaptor::to_aten_tensor(kv_cache);
    auto block_tables_at = adaptor::to_aten_tensor(block_tables);
    auto context_lens_at = adaptor::to_aten_tensor(context_lens);
    auto softmax_lse_at = adaptor::to_aten_tensor(softmax_lse);
    adaptor::iluvatar_vendor::paged_attention_mla(
        output_at, query_at,
        kv_cache_at, scale,
        block_tables_at, context_lens_at,
        max_context_len, false, softmax_lse_at);
}

static bool registered = []() {
    vendor_ops::paged_attention_mla_dispatcher().registerDevice(Device::Type::ILUVATAR, &run);
    return true;
}();

} // namespace infinicore::op::paged_attention_mla_impl::iluvatar
#endif
