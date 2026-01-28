#ifndef __GT_CUDA_H__
#define __GT_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::gt::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::Greater>;
} // namespace op::gt::cuda

#endif // __GT_CUDA_H__
