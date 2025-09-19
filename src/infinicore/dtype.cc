#include <infinicore.hpp>

namespace infinicore {

std::string toString(const DataType &dtype) {
    switch (dtype) {
    case DataType::BYTE:
        return "BYTE";
    case DataType::BOOL:
        return "BOOL";
    case DataType::I8:
        return "I8";
    case DataType::I16:
        return "I16";
    case DataType::I32:
        return "I32";
    case DataType::I64:
        return "I64";
    case DataType::U8:
        return "U8";
    case DataType::U16:
        return "U16";
    case DataType::U32:
        return "U32";
    case DataType::U64:
        return "U64";
    case DataType::F8:
        return "F8";
    case DataType::F16:
        return "F16";
    case DataType::F32:
        return "F32";
    case DataType::F64:
        return "F64";
    case DataType::C16:
        return "C16";
    case DataType::C32:
        return "C32";
    case DataType::C64:
        return "C64";
    case DataType::C128:
        return "C128";
    case DataType::BF16:
        return "BF16";
    }

    // TODO: Add error handling.
    return "";
}

DataTypePy::DataTypePy(const DataType &dtype) : dtype_{dtype} {}

std::string DataTypePy::toString(const DataTypePy &dtype) {
    std::string str{"infinicore."};

    switch (dtype.dtype_) {
    case DataType::BYTE:
        str += "uint8";
        break;
    case DataType::BOOL:
        str += "bool";
        break;
    case DataType::I8:
        str += "int8";
        break;
    case DataType::I16:
        str += "int16";
        break;
    case DataType::I32:
        str += "int32";
        break;
    case DataType::I64:
        str += "int64";
        break;
    case DataType::U8:
        str += "uint8";
        break;
    case DataType::U16:
        str += "uint16";
        break;
    case DataType::U32:
        str += "uint32";
        break;
    case DataType::U64:
        str += "uint64";
        break;
    case DataType::F8:
        str += "float8";
        break;
    case DataType::F16:
        str += "float16";
        break;
    case DataType::F32:
        str += "float32";
        break;
    case DataType::F64:
        str += "float64";
        break;
    case DataType::C16:
        str += "complex16";
        break;
    case DataType::C32:
        str += "complex32";
        break;
    case DataType::C64:
        str += "complex64";
        break;
    case DataType::C128:
        str += "complex128";
        break;
    case DataType::BF16:
        str += "bfloat16";
        break;
    }

    // TODO: Add error handling.
    return str;
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
