#ifndef __FMAX_CUDA_H__
#define __FMAX_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::fmax::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::Fmax>;
} // namespace op::fmax::cuda

#endif // __FMAX_CUDA_H__
