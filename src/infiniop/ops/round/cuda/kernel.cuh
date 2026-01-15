#ifndef __ROUND_CUDA_H__
#define __ROUND_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::round::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Round>;
} // namespace op::round::cuda

#endif // __ROUND_CUDA_H__
