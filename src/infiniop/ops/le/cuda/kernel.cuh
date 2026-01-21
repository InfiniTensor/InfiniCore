#ifndef __LE_CUDA_H__
#define __LE_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::le::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::LessOrEqual>;
} // namespace op::le::cuda

#endif // __LE_CUDA_H__
