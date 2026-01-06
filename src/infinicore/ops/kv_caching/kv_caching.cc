#include "infinicore/ops/kv_caching.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<KVCaching::schema> &KVCaching::dispatcher() {
    static common::OpDispatcher<KVCaching::schema> dispatcher_;
    return dispatcher_;
};

void KVCaching::execute(Tensor k_cache,
                        Tensor v_cache,
                        Tensor k,
                        Tensor v,
                        Tensor past_kv_lengths) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(k_cache, v_cache, k, v, past_kv_lengths);
    infinicore::context::setDevice(k_cache->device());
    auto device_type = k_cache->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No KVCaching implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(k_cache, v_cache, k, v, past_kv_lengths);
}

Tensor kv_caching(Tensor k_cache,
                  Tensor v_cache,
                  Tensor k,
                  Tensor v,
                  Tensor past_kv_lengths) {
    KVCaching::execute(k_cache, v_cache, k, v, past_kv_lengths);
    return k_cache; // or v_cache, depending on the intended use
}

void kv_caching_(Tensor k_cache,
                 Tensor v_cache,
                 Tensor k,
                 Tensor v,
                 Tensor past_kv_lengths) {
    KVCaching::execute(k_cache, v_cache, k, v, past_kv_lengths);
}
} // namespace infinicore::op
