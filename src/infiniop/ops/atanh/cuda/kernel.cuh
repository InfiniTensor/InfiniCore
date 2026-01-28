#ifndef __ATANH_CUDA_H__
#define __ATANH_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::atanh::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Atanh>;
} // namespace op::atanh::cuda

#endif // __ATANH_CUDA_H__
