#ifndef __ISFINITE_CUDA_H__
#define __ISFINITE_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::isfinite::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::IsFinite>;
} // namespace op::isfinite::cuda

#endif // __ISFINITE_CUDA_H__
