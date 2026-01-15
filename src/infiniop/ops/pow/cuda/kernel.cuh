#ifndef __POW_CUDA_H__
#define __POW_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::pow::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::Pow>;
} // namespace op::pow::cuda

#endif // __POW_CUDA_H__
