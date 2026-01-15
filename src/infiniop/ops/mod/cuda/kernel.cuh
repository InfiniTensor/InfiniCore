#ifndef __MOD_CUDA_H__
#define __MOD_CUDA_H__

#include "../../../elementwise/binary.h"

namespace op::mod::cuda {
using Op = op::elementwise::binary::cuda::BinaryOp<op::elementwise::binary::BinaryMode::Mod>;
} // namespace op::mod::cuda

#endif // __MOD_CUDA_H__
