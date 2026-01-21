#ifndef __LOGICAL_OR_CUDA_H__
#define __LOGICAL_OR_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::logical_or::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::LogicalOr>;
} // namespace op::logical_or::cuda

#endif // __LOGICAL_OR_CUDA_H__
