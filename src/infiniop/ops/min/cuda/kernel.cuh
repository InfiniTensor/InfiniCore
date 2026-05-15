#ifndef __MIN_CUDA_H__
#define __MIN_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::min::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::Min>;
} // namespace op::min::cuda

#endif // __MIN_CUDA_H__
