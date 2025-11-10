#include "safetensors_loader.hpp"
#include "infinicore.hpp"

#include <pybind11/embed.h> // For Py_Initialize and Python C API
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>

namespace py = pybind11;

namespace infinicore::safetensors {

// Helper to convert torch::dtype to infinicore::DataType
infinicore::DataType to_infinicore_dtype(py::object torch_dtype_obj) {
    py::module torch_module = py::module::import("torch");
    if (torch_dtype_obj.is(torch_module.attr("float32"))) return infinicore::DataType::F32;
    if (torch_dtype_obj.is(torch_module.attr("float64"))) return infinicore::DataType::F64;
    if (torch_dtype_obj.is(torch_module.attr("int64"))) return infinicore::DataType::I64;
    if (torch_dtype_obj.is(torch_module.attr("int32"))) return infinicore::DataType::I32;
    // Add more dtypes as needed
    throw std::runtime_error("Unsupported torch.dtype for Infinicore Tensor conversion.");
}

infinicore::Tensor load_tensor(const std::string& file_path, const std::string& tensor_name) {
    // Ensure the Python interpreter is initialized. In a real application, this should be done once.
    if (!Py_IsInitialized()) {
        py::scoped_interpreter guard{}; // Start the interpreter and keep it alive
    }

    try {
        // Import the Python module and function
        py::module infinicore_module = py::module::import("infinicore.safetensors_python_loader");
        py::function load_func = infinicore_module.attr("load_tensor_from_safetensors");

        // Call the Python function
        py::object result_tensor_obj = load_func(file_path, tensor_name);

        // Convert py::object to torch::Tensor (this requires pybind11 to know about torch types,
        // or we can extract raw data using __array_interface__ or similar if it's a NumPy-compatible tensor)
        // For now, let's assume result_tensor_obj is a torch.Tensor and extract its raw data.
        // This part needs careful handling of different possible Python tensor types.

        // Ensure the tensor is on CPU and contiguous for raw data access
        py::module torch_module = py::module::import("torch");
        py::object cpu_contiguous_tensor = result_tensor_obj.attr("contiguous")().attr("cpu")();

        // Extract data pointer, shape, and dtype
        uintptr_t data_ptr = cpu_contiguous_tensor.attr("data_ptr")().cast<uintptr_t>();

        // Get shape as a list/tuple of integers, then convert to std::vector<size_t>
        py::list py_shape = cpu_contiguous_tensor.attr("shape");
        std::vector<size_t> shape;
        for (const auto& dim_obj : py_shape) {
            shape.push_back(dim_obj.cast<size_t>());
        }

        py::object py_dtype = cpu_contiguous_tensor.attr("dtype");
        infinicore::DataType infinicore_dtype = to_infinicore_dtype(py_dtype);

        // Create Infinicore Tensor from blob
        return infinicore::Tensor::from_blob(
            reinterpret_cast<void*>(data_ptr), shape, infinicore_dtype, infinicore::Device());

    } catch (const py::error_already_set &e) {
        // Handle Python exceptions
        spdlog::error("Python Error: {}", e.what());
        throw std::runtime_error(e.what());
    } catch (const std::exception &e) {
        spdlog::error("C++ Error during safetensors loading: {}", e.what());
        throw;
    }
}

void bind(py::module& m) {
    m.def("load_safetensors_tensor", &load_tensor,
          py::arg("file_path"), py::arg("tensor_name"),
          "Loads a tensor from a safetensors file.");
}

} // namespace infinicore::safetensors
