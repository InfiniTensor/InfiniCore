#ifndef __EQ_CUDA_H__
#define __EQ_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::eq::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::Equal>;
} // namespace op::eq::cuda

#endif // __EQ_CUDA_H__
