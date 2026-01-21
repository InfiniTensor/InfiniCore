#ifndef __ISNAN_CUDA_H__
#define __ISNAN_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::isnan::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::IsNan>;
} // namespace op::isnan::cuda

#endif // __ISNAN_CUDA_H__
