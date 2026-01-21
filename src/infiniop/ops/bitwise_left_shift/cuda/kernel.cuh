#ifndef __BITWISE_LEFT_SHIFT_CUDA_H__
#define __BITWISE_LEFT_SHIFT_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::bitwise_left_shift::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::BitwiseLeftShift>;
} // namespace op::bitwise_left_shift::cuda

#endif // __BITWISE_LEFT_SHIFT_CUDA_H__
