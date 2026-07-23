#pragma once

#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(
    Fp8MlaRmsnormCache,
    Tensor,
    const Tensor &, const Tensor &, const Tensor &, const Tensor &,
    double);

INFINICORE_GRAPH_OP_CLASS(
    Fp8MlaRmsnormDualCache,
    Tensor, Tensor,
    const Tensor &, const Tensor &, const Tensor &, const Tensor &,
    double);

void fp8_mla_rmsnorm_cache_(
    Tensor cache,
    const Tensor &compressed_kv,
    const Tensor &norm_weight,
    const Tensor &rope,
    const Tensor &slot_mapping,
    double eps);

void fp8_mla_rmsnorm_dual_cache_(
    Tensor cache,
    Tensor vendor_cache,
    const Tensor &compressed_kv,
    const Tensor &norm_weight,
    const Tensor &rope,
    const Tensor &slot_mapping,
    double eps);

} // namespace infinicore::op
