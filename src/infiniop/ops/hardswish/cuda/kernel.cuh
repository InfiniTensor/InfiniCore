#ifndef __HARDSWISH_CUDA_H__
#define __HARDSWISH_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::hardswish::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Hardswish>;
} // namespace op::hardswish::cuda

#endif // __HARDSWISH_CUDA_H__
