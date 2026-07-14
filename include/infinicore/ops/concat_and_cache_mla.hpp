#pragma once

#include "../device.hpp"
#include "../tensor.hpp"
#include <string>

namespace infinicore::op {

void concat_and_cache_mla_(const Tensor &kv_c,
                           const Tensor &k_pe,
                           Tensor kv_cache,
                           const Tensor &slot_mapping,
                           const std::string &kv_cache_dtype,
                           Tensor scale);

} // namespace infinicore::op
