#ifndef __REMAINDER_CUDA_H__
#define __REMAINDER_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::remainder::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::Remainder>;
} // namespace op::remainder::cuda

#endif // __REMAINDER_CUDA_H__
