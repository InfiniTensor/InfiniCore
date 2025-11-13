#pragma once

#include "common/op.hpp"
#include <pybind11/pybind11.h>

namespace infinicore::op {

Tensor linear(Tensor input, Tensor weight, pybind11::object bias);

void linear_(Tensor out, Tensor input, Tensor weight, pybind11::object bias);

} // namespace infinicore::op
