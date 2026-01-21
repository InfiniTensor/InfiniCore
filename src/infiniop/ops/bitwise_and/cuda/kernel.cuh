#ifndef __BITWISE_AND_CUDA_H__
#define __BITWISE_AND_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::bitwise_and::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::BitwiseAnd>;
} // namespace op::bitwise_and::cuda

#endif // __BITWISE_AND_CUDA_H__
