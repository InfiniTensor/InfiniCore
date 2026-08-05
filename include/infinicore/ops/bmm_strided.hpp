#pragma once

#include "../tensor.hpp"

namespace infinicore::op {

// Batched matrix multiplication that preserves arbitrary valid tensor strides,
// including a transposed output view. The Iluvatar implementation deliberately
// mirrors vLLM's torch.bmm(..., out=transpose_view) MLA projection path.
void bmm_strided_(Tensor output, const Tensor &a, const Tensor &b);

} // namespace infinicore::op
