#ifndef __LOGICAL_XOR_CUDA_H__
#define __LOGICAL_XOR_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::logical_xor::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::LogicalXor>;
} // namespace op::logical_xor::cuda

#endif // __LOGICAL_XOR_CUDA_H__
