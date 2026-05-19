#pragma once

#include "../../device.hpp"

#ifdef ENABLE_MUTUAL_AWARENESS
#include "../../analyzer/optimization_intent.hpp"
#endif

#include <array>
#include <cstddef>

namespace infinicore::op::common {
template <typename Fn>
class OpDispatcher {
public:
    void registerDevice(Device::Type device_type, Fn fn, bool override_existing = true) {
        if (table_[(size_t)device_type] == nullptr || override_existing) {
            table_[(size_t)device_type] = fn;
        }
    }

    void registerDevice(std::initializer_list<Device::Type> device_types, Fn fn, bool override_existing = true) {
        for (auto device_type : device_types) {
            registerDevice(device_type, fn, override_existing);
        }
    }

    void registerAll(Fn fn, bool override_existing = true) {
        for (size_t device_type = 0; device_type < static_cast<size_t>(Device::Type::COUNT); ++device_type) {
            registerDevice((Device::Type)device_type, fn, override_existing);
        }
    }

    Fn lookup(Device::Type device_type) const {
        return table_.at((size_t)device_type);
    }

#ifdef ENABLE_MUTUAL_AWARENESS
    // Goal-aware kernel registration. Backward compatible: callers that don't
    // know about goals keep using the device-only overloads. Only kernels that
    // want to specialize per OptimizationGoal need the goal-aware form.
    static constexpr std::size_t kGoalCount = 4;

    void registerDevice(Device::Type device_type,
                        Fn fn,
                        analyzer::OptimizationGoal goal,
                        bool override_existing = true) {
        std::size_t k = goalKey(device_type, goal);
        if (goal_table_[k] == nullptr || override_existing) {
            goal_table_[k] = fn;
        }
    }

    // Look up a kernel by (device, goal). If no goal-specific kernel is
    // registered, fall back to the device-default kernel registered through
    // the legacy lookup(device_type) path.
    Fn lookup(Device::Type device_type, analyzer::OptimizationGoal goal) const {
        std::size_t k = goalKey(device_type, goal);
        Fn fn = goal_table_[k];
        if (fn != nullptr) {
            return fn;
        }
        return lookup(device_type);
    }

private:
    static std::size_t goalKey(Device::Type device_type, analyzer::OptimizationGoal goal) {
        return static_cast<std::size_t>(device_type) * kGoalCount
               + static_cast<std::size_t>(goal);
    }

    std::array<Fn,
               static_cast<std::size_t>(Device::Type::COUNT) * kGoalCount> goal_table_{};
#endif

private:
    std::array<Fn, static_cast<size_t>(Device::Type::COUNT)> table_{};
};
} // namespace infinicore::op::common
