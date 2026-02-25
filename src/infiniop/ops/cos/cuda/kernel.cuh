#ifndef __COS_CUDA_H__
#define __COS_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::cos::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Cos>;
} // namespace op::cos::cuda

#endif // __COS_CUDA_H__
