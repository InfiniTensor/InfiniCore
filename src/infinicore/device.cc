#include <map>
#include <stdexcept>
#include <string>

#include "infinicore.hpp"

#include "infinirt.h"

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

std::pair<Device::Type, Device::Index> toInfiniDevice(std::string type, DevicePy::Index index) {
    constexpr auto device_type_count{static_cast<std::size_t>(Device::Type::COUNT)};

    static const std::unordered_map<Device::Type, std::string> torch_device_map{
        {Device::Type::CPU, "cpu"},
        {Device::Type::NVIDIA, "cuda"},
        {Device::Type::CAMBRICON, "mlu"},
        {Device::Type::ASCEND, "npu"},
        {Device::Type::METAX, "cuda"},
        {Device::Type::MOORE, "musa"},
        {Device::Type::ILUVATAR, "cuda"},
        {Device::Type::KUNLUN, "cuda"},
        {Device::Type::SUGON, "cuda"}};

    std::unordered_map<std::string, std::unordered_map<Device::Type, std::size_t>> torch_devices{
        {"cpu", {{Device::Type::CPU, 0}}},
        {"cuda", {{Device::Type::NVIDIA, 0}, {Device::Type::ILUVATAR, 0}, {Device::Type::METAX, 0}, {Device::Type::KUNLUN, 0}, {Device::Type::SUGON, 0}}},
        {"mlu", {{Device::Type::CAMBRICON, 0}}},
        {"npu", {{Device::Type::ASCEND, 0}}},
        {"musa", {{Device::Type::MOORE, 0}}}};

    auto it = torch_devices.find(type);
    if (it == torch_devices.end()) {
        throw std::invalid_argument("Unsupported device type: `" + type + "`.");
    }

    std::array<int, device_type_count> all_device_count;
    auto status{infinirtGetAllDeviceCount(all_device_count.data())};
    if (status != INFINI_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to get all device counts with error code: `" + std::to_string(status) + "`.");
    }

    for (std::size_t i{0}; i < device_type_count; ++i) {
        const auto device_type{static_cast<Device::Type>(i)};
        torch_devices[torch_device_map.at(device_type)][device_type] += all_device_count[i];
    }

    for (const auto &[infini_device_type, count] : torch_devices[type]) {
        for (std::size_t i{0}; i < count; ++i) {
            if (index == 0) {
                return {infini_device_type, index};
            }

            --index;
        }
    }

    throw std::runtime_error("Internal error: Device mapping failed.");
}

DevicePy::DevicePy(const Device &device) : device_{device} {}

DevicePy::DevicePy(const std::string &type, DevicePy::Index index)
    : type_{type}, index_{index} {
    const auto [infini_device_type, infini_device_index]{toInfiniDevice(type, index)};

    device_ = Device{infini_device_type, infini_device_index};
}

const std::string &DevicePy::getType() const {
    return type_;
}

const DevicePy::Index &DevicePy::getIndex() const {
    return index_;
}

std::string DevicePy::toRepresentation() const {
    return "device(type='" + type_ + "', index=" + std::to_string(index_) + ")";
}

std::string DevicePy::toString() const {
    return type_ + ':' + std::to_string(index_);
}

bool Device::operator==(const Device &other) const {
    return type_ == other.type_ && index_ == other.index_;
}
} // namespace infinicore
