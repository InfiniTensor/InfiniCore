#include <infinicore.hpp>

namespace infinicore {

Device::Device(const Type &type, const Index &index) : type_{type}, index_{index} {}

const Device::Type &Device::getType() const {
    return type_;
}

const Device::Index &Device::getIndex() const {
    return index_;
}

std::string Device::toString() const {
    return toString(type_) + ":" + std::to_string(index_);
}

std::string Device::toString(const Type &type) {
    switch (type) {
    case Type::CPU:
        return "CPU";
    case Type::NVIDIA:
        return "NVIDIA";
    case Type::CAMBRICON:
        return "CAMBRICON";
    case Type::ASCEND:
        return "ASCEND";
    case Type::METAX:
        return "METAX";
    case Type::MOORE:
        return "MOORE";
    case Type::ILUVATAR:
        return "ILUVATAR";
    case Type::KUNLUN:
        return "KUNLUN";
    case Type::SUGON:
        return "SUGON";
    }

    // TODO: Add error handling.
    return "";
}

} // namespace infinicore
