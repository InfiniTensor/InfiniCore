#ifndef __COPYSIGN_CUDA_H__
#define __COPYSIGN_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::copysign::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::CopySign>;
} // namespace op::copysign::cuda

#endif // __COPYSIGN_CUDA_H__
