#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class PagedCaching {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, Tensor);
    static void execute(Tensor k_cache, Tensor v_cache, Tensor k, Tensor v, Tensor slot_mapping);
    static common::OpDispatcher<schema> &dispatcher();
};

void paged_caching_(Tensor k_cache, Tensor v_cache, Tensor k, Tensor v, Tensor slot_mapping);

} // namespace infinicore::op
