#ifndef __SQRT_CUDA_H__
#define __SQRT_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::sqrt::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Sqrt>;
} // namespace op::sqrt::cuda

#endif // __SQRT_CUDA_H__
