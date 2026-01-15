#ifndef __MAX_CUDA_H__
#define __MAX_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::max::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::Max>;
} // namespace op::max::cuda

#endif // __MAX_CUDA_H__
