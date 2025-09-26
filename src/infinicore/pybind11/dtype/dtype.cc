#include "dtype.hpp"

namespace infinicore::py {

DataType::DataType(const infinicore::DataType &dtype) : dtype_{dtype} {}

std::string DataType::toString(const DataType &dtype) {
    std::string str{"infinicore."};

    switch (dtype.dtype_) {
    case infinicore::DataType::BYTE:
        str += "uint8";
        break;
    case infinicore::DataType::BOOL:
        str += "bool";
        break;
    case infinicore::DataType::I8:
        str += "int8";
        break;
    case infinicore::DataType::I16:
        str += "int16";
        break;
    case infinicore::DataType::I32:
        str += "int32";
        break;
    case infinicore::DataType::I64:
        str += "int64";
        break;
    case infinicore::DataType::U8:
        str += "uint8";
        break;
    case infinicore::DataType::U16:
        str += "uint16";
        break;
    case infinicore::DataType::U32:
        str += "uint32";
        break;
    case infinicore::DataType::U64:
        str += "uint64";
        break;
    case infinicore::DataType::F8:
        str += "float8";
        break;
    case infinicore::DataType::F16:
        str += "float16";
        break;
    case infinicore::DataType::F32:
        str += "float32";
        break;
    case infinicore::DataType::F64:
        str += "float64";
        break;
    case infinicore::DataType::C16:
        str += "complex16";
        break;
    case infinicore::DataType::C32:
        str += "complex32";
        break;
    case infinicore::DataType::C64:
        str += "complex64";
        break;
    case infinicore::DataType::C128:
        str += "complex128";
        break;
    case infinicore::DataType::BF16:
        str += "bfloat16";
        break;
    }

    // TODO: Add error handling.
    return str;
}

} // namespace infinicore::py
