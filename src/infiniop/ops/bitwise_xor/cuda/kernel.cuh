#ifndef __BITWISE_XOR_CUDA_H__
#define __BITWISE_XOR_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::bitwise_xor::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::BitwiseXor>;
} // namespace op::bitwise_xor::cuda

#endif // __BITWISE_XOR_CUDA_H__
