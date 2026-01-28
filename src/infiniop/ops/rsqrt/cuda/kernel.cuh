#ifndef __RSQRT_CUDA_H__
#define __RSQRT_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::rsqrt::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Rsqrt>;
} // namespace op::rsqrt::cuda

#endif // __RSQRT_CUDA_H__
