#ifndef __LOGICAL_AND_CUDA_H__
#define __LOGICAL_AND_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::logical_and::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::LogicalAnd>;
} // namespace op::logical_and::cuda

#endif // __LOGICAL_AND_CUDA_H__
