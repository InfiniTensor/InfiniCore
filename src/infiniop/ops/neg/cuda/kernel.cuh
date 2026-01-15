#ifndef __NEG_CUDA_H__
#define __NEG_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::neg::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Neg>;
} // namespace op::neg::cuda

#endif // __NEG_CUDA_H__
