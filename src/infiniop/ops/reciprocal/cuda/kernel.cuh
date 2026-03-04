#ifndef __RECIPROCAL_CUDA_H__
#define __RECIPROCAL_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::reciprocal::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Reciprocal>;
} // namespace op::reciprocal::cuda

#endif // __RECIPROCAL_CUDA_H__
