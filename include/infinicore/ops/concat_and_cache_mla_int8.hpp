#pragma once

#include "../device.hpp"
#include "../tensor.hpp"

namespace infinicore::op {

void concat_and_cache_mla_int8_(const Tensor &kv_c_int8,
                                const Tensor &kv_c_scale,
                                const Tensor &k_pe_int8,
                                const Tensor &k_pe_scale,
                                Tensor kv_cache,
                                Tensor kv_cache_scale,
                                const Tensor &slot_mapping);

} // namespace infinicore::op
