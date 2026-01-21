#ifndef __ISINF_CUDA_H__
#define __ISINF_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::isinf::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::IsInf>;
} // namespace op::isinf::cuda

#endif // __ISINF_CUDA_H__
