#include "infinicore.hpp"
#include "../../utils/custom_types.h"
#include <pybind11/pybind11.h>
#include <cstdint>
#include <string>

namespace py = pybind11;

namespace infinicore::tensor {

namespace {

// Element boxing primitives (raw CPython API).

inline PyObject *box_bool(bool v) {
    return PyBool_FromLong(v ? 1 : 0);
}

inline PyObject *box_int_signed(int64_t v) {
    return PyLong_FromLongLong(v);
}

inline PyObject *box_int_unsigned(uint64_t v) {
    return PyLong_FromUnsignedLongLong(v);
}

inline PyObject *box_float(double v) {
    return PyFloat_FromDouble(v);
}

inline PyObject *box_f16(fp16_t v) {
    return PyFloat_FromDouble(utils::cast<float, fp16_t>(v));
}

inline PyObject *box_bf16(bf16_t v) {
    return PyFloat_FromDouble(utils::cast<float, bf16_t>(v));
}

// RAII guard for an owned PyObject*; cleans up partial list on exception.

class PyObjectGuard {
public:
    explicit PyObjectGuard(PyObject *obj) : obj_(obj) {}
    ~PyObjectGuard() {
        Py_XDECREF(obj_);
    }

    PyObjectGuard(const PyObjectGuard &) = delete;
    PyObjectGuard &operator=(const PyObjectGuard &) = delete;

    PyObject *get() const {
        return obj_;
    }

    PyObject *release() {
        PyObject *r = obj_;
        obj_ = nullptr;
        return r;
    }

private:
    PyObject *obj_;
};

// Builders: tight typed loop on innermost dim, recurse on outer dims.
// shape.size() == 0 returns a bare scalar (matches PyTorch tolist()).

template <typename T, typename Boxer>
PyObject *fill_leaf_list(const T *data, Py_ssize_t n, Boxer box) {
    PyObjectGuard lst(PyList_New(n));
    if (!lst.get()) {
        throw py::error_already_set();
    }
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item = box(data[i]);
        if (!item) {
            throw py::error_already_set();
        }
        PyList_SET_ITEM(lst.get(), i, item); // steals reference
    }
    return lst.release();
}

template <typename T, typename Boxer>
PyObject *build_recursive(const T *&cursor,
                          const Shape &shape,
                          size_t dim,
                          Boxer box) {
    const size_t ndim = shape.size();

    if (dim == ndim) {
        PyObject *item = box(*cursor);
        if (!item) {
            throw py::error_already_set();
        }
        ++cursor;
        return item;
    }

    const Py_ssize_t count = static_cast<Py_ssize_t>(shape[dim]);

    if (dim + 1 == ndim) {
        PyObject *leaf = fill_leaf_list(cursor, count, box);
        cursor += count;
        return leaf;
    }

    PyObjectGuard lst(PyList_New(count));
    if (!lst.get()) {
        throw py::error_already_set();
    }
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject *child = build_recursive(cursor, shape, dim + 1, box);
        PyList_SET_ITEM(lst.get(), i, child);
    }
    return lst.release();
}

template <typename T, typename Boxer>
py::object build_typed(const std::byte *data, const Shape &shape, Boxer box) {
    const T *cursor = reinterpret_cast<const T *>(data);
    PyObject *root = build_recursive<T>(cursor, shape, 0, box);
    return py::reinterpret_steal<py::object>(root);
}

// dtype dispatch (lambdas keep boxers inlinable).
py::object dispatch_build(const std::byte *data, const Shape &shape, DataType dtype) {
    // Case order must match dtype.hpp DataType enum exactly:
    // BYTE, BOOL, I8, I16, I32, I64, U8, U16, U32, U64,
    // F8, F16, F32, F64, C16, C32, C64, C128, BF16
    switch (dtype) {
    case DataType::BYTE:
        throw py::type_error(
            std::string("Unsupported dtype for to_list: ") + toString(dtype));
    case DataType::BOOL:
        return build_typed<bool>(data, shape, [](bool v) { return box_bool(v); });
    case DataType::I8:
        return build_typed<int8_t>(data, shape, [](int8_t v) { return box_int_signed(v); });
    case DataType::I16:
        return build_typed<int16_t>(data, shape, [](int16_t v) { return box_int_signed(v); });
    case DataType::I32:
        return build_typed<int32_t>(data, shape, [](int32_t v) { return box_int_signed(v); });
    case DataType::I64:
        return build_typed<int64_t>(data, shape, [](int64_t v) { return box_int_signed(v); });
    case DataType::U8:
        return build_typed<uint8_t>(data, shape, [](uint8_t v) { return box_int_unsigned(v); });
    case DataType::U16:
        return build_typed<uint16_t>(data, shape, [](uint16_t v) { return box_int_unsigned(v); });
    case DataType::U32:
        return build_typed<uint32_t>(data, shape, [](uint32_t v) { return box_int_unsigned(v); });
    case DataType::U64:
        return build_typed<uint64_t>(data, shape, [](uint64_t v) { return box_int_unsigned(v); });
    case DataType::F8:
        throw py::type_error(
            std::string("Unsupported dtype for to_list: ") + toString(dtype));
    case DataType::F16:
        return build_typed<fp16_t>(data, shape, [](fp16_t v) { return box_f16(v); });
    case DataType::F32:
        return build_typed<float>(data, shape, [](float v) { return box_float(v); });
    case DataType::F64:
        return build_typed<double>(data, shape, [](double v) { return box_float(v); });
    case DataType::C16:
        throw py::type_error(
            std::string("Unsupported dtype for to_list: ") + toString(dtype));
    case DataType::C32:
        throw py::type_error(
            std::string("Unsupported dtype for to_list: ") + toString(dtype));
    case DataType::C64:
        throw py::type_error(
            std::string("Unsupported dtype for to_list: ") + toString(dtype));
    case DataType::C128:
        throw py::type_error(
            std::string("Unsupported dtype for to_list: ") + toString(dtype));
    case DataType::BF16:
        return build_typed<bf16_t>(data, shape, [](bf16_t v) { return box_bf16(v); });
    default:
        throw py::type_error(
            std::string("Unsupported dtype for to_list: ") + toString(dtype));
    }
}

} // namespace

py::object to_list_py(const Tensor &tensor) {
    if (tensor->device().getType() != Device::Type::CPU) {
        throw py::value_error(
            "to_list() requires a CPU tensor; call .to(\"cpu\") before to_list()");
    }
    if (!tensor->is_contiguous()) {
        throw py::value_error(
            "to_list() requires a contiguous tensor; call .contiguous() before to_list()");
    }
    return dispatch_build(tensor->data(), tensor->shape(), tensor->dtype());
}

} // namespace infinicore::tensor
