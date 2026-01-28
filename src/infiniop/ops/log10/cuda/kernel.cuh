#ifndef __LOG10_CUDA_H__
#define __LOG10_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::log10::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Log10>;
} // namespace op::log10::cuda

#endif // __LOG10_CUDA_H__
