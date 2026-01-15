#ifndef __ACOS_CUDA_H__
#define __ACOS_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::acos::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Acos>;
} // namespace op::acos::cuda

#endif // __ACOS_CUDA_H__
