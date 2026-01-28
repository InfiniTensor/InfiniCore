#ifndef __GE_CUDA_H__
#define __GE_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::ge::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::GreaterOrEqual>;
} // namespace op::ge::cuda

#endif // __GE_CUDA_H__
