#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class KVCaching {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, Tensor);
    static void execute(Tensor k_cache,
                        Tensor v_cache,
                        Tensor k,
                        Tensor v,
                        Tensor past_kv_lengths);
    static common::OpDispatcher<schema> &dispatcher();
};

void kv_caching_(Tensor k_cache,
                 Tensor v_cache,
                 Tensor k,
                 Tensor v,
                 Tensor past_kv_lengths);
} // namespace infinicore::op
