#ifndef GU_MUL_H
#define GU_MUL_H

#include "infinicore/tensor.hpp"
#include "infinicore/context/context.hpp"
#include "infiniop/ops/mul.h" 

namespace infinicore::op {

Tensor mul(Tensor a, Tensor b, infiniopHandle_t handle);

} // namespace infinicore::op

#endif // GU_MUL_H