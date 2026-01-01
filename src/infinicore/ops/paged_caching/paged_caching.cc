#include "infinicore/ops/paged_caching.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<PagedCaching::schema> &PagedCaching::dispatcher() {
    static common::OpDispatcher<PagedCaching::schema> dispatcher_;
    return dispatcher_;
};

void PagedCaching::execute(Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, Tensor slot_mapping) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(k, v, k_cache, v_cache, slot_mapping);
    infinicore::context::setDevice(k->device());
    dispatcher().lookup(k->device().getType())(k, v, k_cache, v_cache, slot_mapping);
}

void paged_caching_(Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, Tensor slot_mapping) {
    PagedCaching::execute(k, v, k_cache, v_cache, slot_mapping);
}

} // namespace infinicore::op
