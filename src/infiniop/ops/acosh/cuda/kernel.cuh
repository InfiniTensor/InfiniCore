#ifndef __ACOSH_CUDA_H__
#define __ACOSH_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::acosh::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Acosh>;
} // namespace op::acosh::cuda

#endif // __ACOSH_CUDA_H__
