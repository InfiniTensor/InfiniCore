#include <pybind11/pybind11.h>

#include <infinicore.h>

namespace py = pybind11;

PYBIND11_MODULE(_infini, m) {
    m.doc() = "InfiniCore Python bindings.";

    py::enum_<infiniStatus_t>(m, "Status")
        .value("SUCCESS", INFINI_STATUS_SUCCESS)
        .value("INTERNAL_ERROR", INFINI_STATUS_INTERNAL_ERROR)
        .value("NOT_IMPLEMENTED", INFINI_STATUS_NOT_IMPLEMENTED)
        .value("BAD_PARAM", INFINI_STATUS_BAD_PARAM)
        .value("NULL_POINTER", INFINI_STATUS_NULL_POINTER)
        .value("DEVICE_TYPE_NOT_SUPPORTED", INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED)
        .value("DEVICE_NOT_FOUND", INFINI_STATUS_DEVICE_NOT_FOUND)
        .value("DEVICE_NOT_INITIALIZED", INFINI_STATUS_DEVICE_NOT_INITIALIZED)
        .value("DEVICE_ARCHITECTURE_NOT_SUPPORTED", INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED)
        .value("BAD_TENSOR_DTYPE", INFINI_STATUS_BAD_TENSOR_DTYPE)
        .value("BAD_TENSOR_SHAPE", INFINI_STATUS_BAD_TENSOR_SHAPE)
        .value("BAD_TENSOR_STRIDES", INFINI_STATUS_BAD_TENSOR_STRIDES)
        .value("INSUFFICIENT_WORKSPACE", INFINI_STATUS_INSUFFICIENT_WORKSPACE);

    py::enum_<infiniDevice_t>(m, "Device")
        .value("CPU", INFINI_DEVICE_CPU)
        .value("NVIDIA", INFINI_DEVICE_NVIDIA)
        .value("CAMBRICON", INFINI_DEVICE_CAMBRICON)
        .value("ASCEND", INFINI_DEVICE_ASCEND)
        .value("METAX", INFINI_DEVICE_METAX)
        .value("MOORE", INFINI_DEVICE_MOORE)
        .value("ILUVATAR", INFINI_DEVICE_ILUVATAR)
        .value("KUNLUN", INFINI_DEVICE_KUNLUN)
        .value("SUGON", INFINI_DEVICE_SUGON)
        .value("TYPE_COUNT", INFINI_DEVICE_TYPE_COUNT);

    py::enum_<infiniDtype_t>(m, "Dtype")
        .value("INVALID", INFINI_DTYPE_INVALID)
        .value("BYTE", INFINI_DTYPE_BYTE)
        .value("BOOL", INFINI_DTYPE_BOOL)
        .value("I8", INFINI_DTYPE_I8)
        .value("I16", INFINI_DTYPE_I16)
        .value("I32", INFINI_DTYPE_I32)
        .value("I64", INFINI_DTYPE_I64)
        .value("U8", INFINI_DTYPE_U8)
        .value("U16", INFINI_DTYPE_U16)
        .value("U32", INFINI_DTYPE_U32)
        .value("U64", INFINI_DTYPE_U64)
        .value("F8", INFINI_DTYPE_F8)
        .value("F16", INFINI_DTYPE_F16)
        .value("F32", INFINI_DTYPE_F32)
        .value("F64", INFINI_DTYPE_F64)
        .value("C16", INFINI_DTYPE_C16)
        .value("C32", INFINI_DTYPE_C32)
        .value("C64", INFINI_DTYPE_C64)
        .value("C128", INFINI_DTYPE_C128)
        .value("BF16", INFINI_DTYPE_BF16);
}
