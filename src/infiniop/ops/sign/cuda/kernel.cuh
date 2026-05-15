#ifndef __SIGN_CUDA_H__
#define __SIGN_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::sign::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Sign>;
} // namespace op::sign::cuda

#endif // __SIGN_CUDA_H__
