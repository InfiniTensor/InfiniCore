#ifndef __TANH_CUDA_H__
#define __TANH_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::tanh::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Tanh>;
} // namespace op::tanh::cuda

#endif // __TANH_CUDA_H__
