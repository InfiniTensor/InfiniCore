#include <infinicore.hpp>

namespace infinicore {

std::string toString(const DataType &dtype) {
    switch (dtype) {
    case DataType::BYTE:
        return "uint8";
    case DataType::BOOL:
        return "bool";
    case DataType::I8:
        return "int8";
    case DataType::I16:
        return "int16";
    case DataType::I32:
        return "int32";
    case DataType::I64:
        return "int64";
    case DataType::U8:
        return "uint8";
    case DataType::U16:
        return "uint16";
    case DataType::U32:
        return "uint32";
    case DataType::U64:
        return "uint64";
    case DataType::F8:
        return "float8";
    case DataType::F16:
        return "float16";
    case DataType::F32:
        return "float32";
    case DataType::F64:
        return "float64";
    case DataType::C16:
        return "complex16";
    case DataType::C32:
        return "complex32";
    case DataType::C64:
        return "complex64";
    case DataType::C128:
        return "complex128";
    case DataType::BF16:
        return "bfloat16";
    }

    // TODO: Add error handling.
    return "";
}

size_t dsize(const DataType &dtype) {
    switch (dtype) {
    case DataType::BYTE:
    case DataType::BOOL:
    case DataType::F8:
    case DataType::I8:
    case DataType::U8:
        return 1;
    case DataType::I16:
    case DataType::U16:
    case DataType::F16:
    case DataType::BF16:
    case DataType::C16:
        return 2;
    case DataType::I32:
    case DataType::U32:
    case DataType::F32:
    case DataType::C32:
        return 4;
    case DataType::I64:
    case DataType::U64:
    case DataType::F64:
    case DataType::C64:
        return 8;
    case DataType::C128:
        return 16;
    }

    // TODO: Add error handling.
    return 0;
}

} // namespace infinicore
