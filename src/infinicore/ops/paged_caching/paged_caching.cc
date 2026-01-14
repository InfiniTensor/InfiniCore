#include "infinicore/ops/paged_caching.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<PagedCaching::schema> &PagedCaching::dispatcher() {
    static common::OpDispatcher<PagedCaching::schema> dispatcher_;
    return dispatcher_;
};

void PagedCaching::execute(Tensor k_cache, Tensor v_cache, Tensor k, Tensor v, Tensor slot_mapping) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(k_cache, v_cache, k, v, slot_mapping);
    infinicore::context::setDevice(k_cache->device());
    dispatcher().lookup(k_cache->device().getType())(k_cache, v_cache, k, v, slot_mapping);
}

void paged_caching_(Tensor k_cache, Tensor v_cache, Tensor k, Tensor v, Tensor slot_mapping) {
    PagedCaching::execute(k_cache, v_cache, k, v, slot_mapping);
}

} // namespace infinicore::op
