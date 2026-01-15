#ifndef __ABS_CUDA_H__
#define __ABS_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::abs::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Abs>;
} // namespace op::abs::cuda

#endif // __ABS_CUDA_H__
