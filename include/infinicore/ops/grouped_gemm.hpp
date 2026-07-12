#pragma once

#include <cstdint>

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(GroupedGemm,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          float,
                          float,
                          const int32_t *);

// Variable-batched matmul shared across `num_groups` weight slabs.
//
// Shapes (all row-major):
//   a            : [M_total, K]
//   b            : [num_groups, N, K]      // out, in -- matches torch linear
//   c            : [M_total, N]
//   group_sizes  : [num_groups], int32     // sum == M_total
//
// For each group g:
//   c[off_g : off_g + group_sizes[g]] = alpha * a[off_g : off_g + group_sizes[g]] @ b[g].T
//                                      + beta * c[...]
// where `off_g = sum(group_sizes[0..g])`.
// `group_sizes_host` (optional): host-side copy of `group_sizes`. When given,
// device backends use it directly and skip the per-call device->host sync of
// the sizes array. Must stay valid until the call returns; pass nullptr (the
// default) under graph capture or when no host copy is available.
Tensor grouped_gemm(const Tensor &a,
                    const Tensor &b,
                    const Tensor &group_sizes,
                    float alpha = 1.0f,
                    float beta = 0.0f,
                    const int32_t *group_sizes_host = nullptr);

void grouped_gemm_(Tensor c,
                   const Tensor &a,
                   const Tensor &b,
                   const Tensor &group_sizes,
                   float alpha,
                   float beta,
                   const int32_t *group_sizes_host = nullptr);

} // namespace infinicore::op
