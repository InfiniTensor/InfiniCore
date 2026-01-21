#ifndef __LOG1P_CUDA_H__
#define __LOG1P_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::log1p::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Log1p>;
} // namespace op::log1p::cuda

#endif // __LOG1P_CUDA_H__
