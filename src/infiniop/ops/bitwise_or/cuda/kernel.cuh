#ifndef __BITWISE_OR_CUDA_H__
#define __BITWISE_OR_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::bitwise_or::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::BitwiseOr>;
} // namespace op::bitwise_or::cuda

#endif // __BITWISE_OR_CUDA_H__
