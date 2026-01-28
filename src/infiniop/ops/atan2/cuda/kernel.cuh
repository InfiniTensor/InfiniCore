#ifndef __ATAN2_CUDA_H__
#define __ATAN2_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::atan2::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::Atan2>;
} // namespace op::atan2::cuda

#endif // __ATAN2_CUDA_H__
