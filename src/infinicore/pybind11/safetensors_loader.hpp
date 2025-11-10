#pragma once

#include <string>
#include <pybind11/pybind11.h>
#include "infinicore.hpp"

namespace infinicore::safetensors {

infinicore::Tensor load_tensor(const std::string& file_path, const std::string& tensor_name);

void bind(pybind11::module& m);

} // namespace infinicore::safetensors
