#include "infinicore/ops/moe_sum.hpp"

#include "infinicore/ops/sum.hpp"

#include <stdexcept>

namespace infinicore::op {

Tensor moe_sum(const Tensor &input) {
    if (input->ndim() != 3) {
        throw std::runtime_error(
            "moe_sum expects input rank 3 [M, topk, H], got ndim=" +
            std::to_string(input->ndim()));
    }
    return sum(input, /*dim=*/{1}, /*keepdim=*/false);
}

void moe_sum_(Tensor out, const Tensor &input) {
    if (input->ndim() != 3) {
        throw std::runtime_error(
            "moe_sum_ expects input rank 3 [M, topk, H], got ndim=" +
            std::to_string(input->ndim()));
    }
    sum_(out, input, /*dim=*/{1}, /*keepdim=*/false);
}

} // namespace infinicore::op
