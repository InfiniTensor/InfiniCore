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
        return "cpu";
    case Type::CUDA:
        return "cuda";
    case Type::META:
        return "meta";
    }

    // TODO: Add error handling.
    return "";
}

} // namespace infinicore
