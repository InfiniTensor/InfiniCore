#include <infinicore.hpp>

namespace infinicore {

std::string to_string(const DataType &dtype) {
    std::string str{"infinicore."};

    switch (dtype) {
    case DataType::bfloat16:
        str += "bfloat16";
        break;
    case DataType::float16:
        str += "float16";
        break;
    case DataType::float32:
        str += "float32";
        break;
    case DataType::float64:
        str += "float64";
        break;
    case DataType::int32:
        str += "int32";
        break;
    case DataType::int64:
        str += "int64";
        break;
    case DataType::uint8:
        str += "uint8";
        break;
    }

    return str;
}

} // namespace infinicore
