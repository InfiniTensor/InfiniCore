#ifndef __EXP2_CUDA_H__
#define __EXP2_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::exp2::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Exp2>;
} // namespace op::exp2::cuda

#endif // __EXP2_CUDA_H__
