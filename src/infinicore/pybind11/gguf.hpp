#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Forward declarations - we'll include the actual header in the .cc file
namespace infinicore {
namespace test {
    class GGUFFileReader;
    struct GGUFTensorInfo;
    struct GGUFKeyValue;
}
}

namespace py = pybind11;

namespace infinicore::gguf {

inline void bind(py::module &m) {
    // Note: Actual implementation will be in a separate .cc file
    // to avoid including the full GGUF header in the main module
}

} // namespace infinicore::gguf
