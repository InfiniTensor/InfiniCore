#ifndef __LT_CUDA_H__
#define __LT_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::lt::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::Less>;
} // namespace op::lt::cuda

#endif // __LT_CUDA_H__
