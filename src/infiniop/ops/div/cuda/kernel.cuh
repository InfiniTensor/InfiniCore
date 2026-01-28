#ifndef __DIV_CUDA_H__
#define __DIV_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::div::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::Divide>;
} // namespace op::div::cuda

#endif // __DIV_CUDA_H__
