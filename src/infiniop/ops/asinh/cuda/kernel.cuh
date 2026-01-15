#ifndef __ASINH_CUDA_H__
#define __ASINH_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::asinh::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Asinh>;
} // namespace op::asinh::cuda

#endif // __ASINH_CUDA_H__
