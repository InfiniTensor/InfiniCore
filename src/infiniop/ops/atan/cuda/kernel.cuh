#ifndef __ATAN_CUDA_H__
#define __ATAN_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::atan::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Atan>;
} // namespace op::atan::cuda

#endif // __ATAN_CUDA_H__
