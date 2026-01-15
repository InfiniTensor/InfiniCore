#ifndef __ERF_CUDA_H__
#define __ERF_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::erf::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Erf>;
} // namespace op::erf::cuda

#endif // __ERF_CUDA_H__
