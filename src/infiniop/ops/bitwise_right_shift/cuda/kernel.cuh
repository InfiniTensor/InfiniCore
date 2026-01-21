#ifndef __BITWISE_RIGHT_SHIFT_CUDA_H__
#define __BITWISE_RIGHT_SHIFT_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::bitwise_right_shift::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::BitwiseRightShift>;
} // namespace op::bitwise_right_shift::cuda

#endif // __BITWISE_RIGHT_SHIFT_CUDA_H__
