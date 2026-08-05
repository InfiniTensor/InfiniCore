#pragma once

#include "../device.hpp"
#include "../tensor.hpp"

namespace infinicore::op {

void concat_mla_q_(const Tensor &ql_nope, const Tensor &q_pe, Tensor q_out);
Tensor concat_mla_q(const Tensor &ql_nope, const Tensor &q_pe);

} // namespace infinicore::op
