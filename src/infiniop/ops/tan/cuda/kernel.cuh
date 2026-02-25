#ifndef __TAN_CUDA_H__
#define __TAN_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::tan::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Tan>;
} // namespace op::tan::cuda

#endif // __TAN_CUDA_H__
