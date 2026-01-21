#ifndef __NE_CUDA_H__
#define __NE_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::ne::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::NotEqual>;
} // namespace op::ne::cuda

#endif // __NE_CUDA_H__
