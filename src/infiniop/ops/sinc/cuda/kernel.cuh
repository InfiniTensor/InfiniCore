#ifndef __SINC_CUDA_H__
#define __SINC_CUDA_H__

#include "../../../elementwise/unary.h"

namespace op::sinc::cuda {
using Op = op::elementwise::unary::cuda::UnaryOp<op::elementwise::unary::UnaryMode::Sinc>;
} // namespace op::sinc::cuda

#endif // __SINC_CUDA_H__
