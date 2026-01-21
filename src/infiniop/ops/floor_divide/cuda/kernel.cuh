#ifndef __FLOOR_DIVIDE_CUDA_H__
#define __FLOOR_DIVIDE_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::floor_divide::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::FloorDivide>;
} // namespace op::floor_divide::cuda

#endif // __FLOOR_DIVIDE_CUDA_H__
