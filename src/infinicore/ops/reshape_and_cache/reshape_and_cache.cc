#include "infinicore/ops/reshape_and_cache.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(ReshapeAndCache);

ReshapeAndCache::ReshapeAndCache(Tensor &key,
                                 Tensor &value,
                                 Tensor &key_cache,
                                 Tensor &value_cache,
                                 Tensor &slot_mapping,
                                 const std::string &kv_cache_dtype,
                                 Tensor &k_scale,
                                 Tensor &v_scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(key, value, key_cache, value_cache, slot_mapping);
    INFINICORE_GRAPH_OP_DISPATCH(key->device().getType(),
                                 key, value, key_cache, value_cache, slot_mapping,
                                 kv_cache_dtype, k_scale, v_scale);
}

void ReshapeAndCache::execute(Tensor &key,
                              Tensor &value,
                              Tensor &key_cache,
                              Tensor &value_cache,
                              Tensor &slot_mapping,
                              const std::string &kv_cache_dtype,
                              Tensor &k_scale,
                              Tensor &v_scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        ReshapeAndCache,
        key, value, key_cache, value_cache, slot_mapping,
        kv_cache_dtype, k_scale, v_scale);
}

void reshape_and_cache(Tensor &key,          // [num_tokens, num_heads, head_size]
                       Tensor &value,        // [num_tokens, num_heads, head_size]
                       Tensor &key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
                       Tensor &value_cache,  // [num_blocks, num_heads, head_size, block_size]
                       Tensor &slot_mapping, // [num_tokens]
                       const std::string &kv_cache_dtype,
                       Tensor &k_scale,
                       Tensor &v_scale) {
    ReshapeAndCache::execute(key, value, key_cache, value_cache, slot_mapping,
                             kv_cache_dtype, k_scale, v_scale);
}

} // namespace infinicore::op
