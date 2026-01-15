#ifndef __CEIL_CUDA_H__
#define __CEIL_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::ceil::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Ceil>;
} // namespace op::ceil::cuda

#endif // __CEIL_CUDA_H__
