#pragma once

#include "../../device.hpp"

#include <array>

namespace infinicore::op::common {
template <typename Fn>
class OpDispatcher {
public:
    void registerDevice(Device::Type device_type, Fn fn) {
        table_[(size_t)device_type] = fn;
    }

    void registerDevice(std::initializer_list<Device::Type> device_types, Fn fn) {
        for (auto device_type : device_types) {
            table_[(size_t)device_type] = fn;
        }
    }

    void registerAll(Fn fn) {
        for (size_t device_type = 0; device_type < static_cast<size_t>(Device::Type::COUNT); ++device_type) {
            table_[device_type] = fn;
        }
    }

    Fn lookup(Device::Type device_type) const {
        return table_.at((size_t)device_type);
    }

private:
    std::array<Fn, static_cast<size_t>(Device::Type::COUNT)> table_;
};
} // namespace infinicore::op::common
