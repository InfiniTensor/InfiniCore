#ifndef __COSH_CUDA_H__
#define __COSH_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::cosh::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Cosh>;
} // namespace op::cosh::cuda

#endif // __COSH_CUDA_H__
