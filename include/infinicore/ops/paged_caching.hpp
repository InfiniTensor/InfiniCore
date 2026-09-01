#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(PagedCaching, Tensor, Tensor, const Tensor &, const Tensor &, const Tensor &, std::optional<Tensor>, std::optional<Tensor>);

void paged_caching_(Tensor k_cache, Tensor v_cache, const Tensor &k, const Tensor &v, const Tensor &slot_mapping,
                    std::optional<Tensor> k_scale = std::nullopt, std::optional<Tensor> v_scale = std::nullopt);

} // namespace infinicore::op
