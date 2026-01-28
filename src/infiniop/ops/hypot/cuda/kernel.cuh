#ifndef __HYPOT_CUDA_H__
#define __HYPOT_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::hypot::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::Hypot>;
} // namespace op::hypot::cuda

#endif // __HYPOT_CUDA_H__
