#include "matmul.hpp"
#include <iostream>

namespace infinicore::py {

Tensor matmul(const Tensor &a, const Tensor &b) {
    auto result = infinicore::op::matmul(a.get(), b.get());
    return Tensor{result};
}

void matmul_(Tensor &c, const Tensor &a, const Tensor &b) {
    infinicore::op::matmul_(c.get(), a.get(), b.get());
}

} // namespace infinicore::py
