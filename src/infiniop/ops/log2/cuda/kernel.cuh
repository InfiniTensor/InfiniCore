#ifndef __LOG2_CUDA_H__
#define __LOG2_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::log2::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Log2>;
} // namespace op::log2::cuda

#endif // __LOG2_CUDA_H__
