#ifndef __LOG_CUDA_H__
#define __LOG_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::log::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Log>;
} // namespace op::log::cuda

#endif // __LOG_CUDA_H__
