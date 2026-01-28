#ifndef __SQUARE_CUDA_H__
#define __SQUARE_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::square::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Square>;
} // namespace op::square::cuda

#endif // __SQUARE_CUDA_H__
