#ifndef __ASIN_CUDA_H__
#define __ASIN_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::asin::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Asin>;
} // namespace op::asin::cuda

#endif // __ASIN_CUDA_H__
