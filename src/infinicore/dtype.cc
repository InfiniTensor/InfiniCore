#include <infinicore.hpp>

namespace infinicore {

std::string toString(const DataType &dtype) {
    std::string str{"infinicore."};

    switch (dtype) {
    case DataType::BFLOAT16:
        str += "bfloat16";
        break;
    case DataType::FLOAT16:
        str += "float16";
        break;
    case DataType::FLOAT32:
        str += "float32";
        break;
    case DataType::FLOAT64:
        str += "float64";
        break;
    case DataType::INT32:
        str += "int32";
        break;
    case DataType::INT64:
        str += "int64";
        break;
    case DataType::UINT8:
        str += "uint8";
        break;
    }

    return str;
}

} // namespace infinicore
