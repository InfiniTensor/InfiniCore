#ifndef __SIN_CUDA_H__
#define __SIN_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::sin::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Sin>;
} // namespace op::sin::cuda

#endif // __SIN_CUDA_H__
