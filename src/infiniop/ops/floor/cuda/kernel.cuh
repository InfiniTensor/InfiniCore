#ifndef __FLOOR_CUDA_H__
#define __FLOOR_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::floor::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Floor>;
} // namespace op::floor::cuda

#endif // __FLOOR_CUDA_H__
