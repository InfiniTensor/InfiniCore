#ifndef GU_TOPK_SOFTMAX_H
#define GU_TOPK_SOFTMAX_H

#include "infinicore/tensor.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/nn.hpp"
#include "infinicore/device.hpp"
#include "infinicore/ops/linear.hpp"
#include "infiniop/ops/topksoftmax.h"
#include <cstring>
#include <stdexcept> 
#include <utility> // for std::pair

namespace infinicore::op {

std::pair<Tensor, Tensor> topk_softmax(
    Tensor input, 
    int k, 
    bool normalize, 
    infiniopHandle_t handle
);

} // namespace infinicore::op

#endif // GU_TOPK_SOFTMAX_H