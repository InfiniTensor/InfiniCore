#include <infinicore.hpp>

namespace infinicore {

Device::Device(const Type &type, const Index &index) : type_{type}, index_{index} {}

const Device::Type &Device::get_type() const {
    return type_;
}

const Device::Index &Device::get_index() const {
    return index_;
}

std::string Device::to_string() const {
    return to_string(type_) + ":" + std::to_string(index_);
}

std::string Device::to_string(const Type &type) {
    switch (type) {
    case Type::cpu:
        return "cpu";
    case Type::cuda:
        return "cuda";
    case Type::meta:
        return "meta";
    }

    // TODO: Add error handling.
    return "";
}

} // namespace infinicore
