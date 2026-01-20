#ifndef __EXP_CUDA_H__
#define __EXP_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::exp::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Exp>;
} // namespace op::exp::cuda

#endif // __EXP_CUDA_H__
