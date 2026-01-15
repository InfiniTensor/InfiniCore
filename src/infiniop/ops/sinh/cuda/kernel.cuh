#ifndef __SINH_CUDA_H__
#define __SINH_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::sinh::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Sinh>;
} // namespace op::sinh::cuda

#endif // __SINH_CUDA_H__
