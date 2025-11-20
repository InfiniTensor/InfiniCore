#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../utils.hpp"
#include "context.hpp"
#include "device.hpp"
#include "dtype.hpp"
#include "ops.hpp"
#include "nn.hpp"
#include "tensor.hpp"

namespace infinicore {

PYBIND11_MODULE(_infinicore, m) {
    context::bind(m);
    device::bind(m);
    dtype::bind(m);
    ops::bind(m);
    tensor::bind(m);
    pybind11_nn::bind(m);
}

} // namespace infinicore
