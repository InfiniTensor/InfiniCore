#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "context.hpp"
#include "device.hpp"
#include "dtype.hpp"
#include "op.hpp"
#include "tensor.hpp"

namespace infinicore {

PYBIND11_MODULE(_infinicore, m) {
    context::bind(m);
    device::bind(m);
    dtype::bind(m);
    op::bind(m);
    tensor::bind(m);
}

} // namespace infinicore
