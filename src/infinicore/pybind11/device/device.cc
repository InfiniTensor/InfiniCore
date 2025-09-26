#include "device.hpp"

#include <stdexcept>

#include "infinicore/common/utils.hpp"
#include "infinirt.h"

namespace infinicore {

infinicore::Device toInfiniDevice(std::string type, Device::Index index) {
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
    INFINICORE_CHECK_ERROR(infinirtGetAllDeviceCount(all_device_count.data()));

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

} // namespace infinicore

namespace infinicore::py {

Device::Device(const infinicore::Device &device) : device_{device} {}

Device::Device(const std::string &type, Device::Index index)
    : type_{type}, index_{index}, device_{toInfiniDevice(type, index)} {}

const std::string &Device::getType() const {
    return type_;
}

const Device::Index &Device::getIndex() const {
    return index_;
}

std::string Device::toRepresentation() const {
    return "device(type='" + type_ + "', index=" + std::to_string(index_) + ")";
}

std::string Device::toString() const {
    return type_ + ':' + std::to_string(index_);
}

} // namespace infinicore::py
