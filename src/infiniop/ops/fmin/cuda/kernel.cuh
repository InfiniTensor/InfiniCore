#ifndef __FMIN_CUDA_H__
#define __FMIN_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::fmin::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::Fmin>;
} // namespace op::fmin::cuda

#endif // __FMIN_CUDA_H__
