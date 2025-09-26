#pragma once
#include "infinicore/context/context.hpp"
#include "runtime/runtime.hpp"

#include <array>
#include <vector>

namespace infinicore {
class ContextImpl {
private:
    std::array<std::vector<std::unique_ptr<Runtime>>, size_t(Device::Type::COUNT)> runtime_table_;
    Runtime *current_runtime_ = nullptr;

protected:
    ContextImpl();

public:
    Runtime *getCurrentRuntime();

    Runtime *getCpuRuntime();

    void setDevice(Device);

    size_t getDeviceCount(Device::Type type);

    static ContextImpl &singleton();

    friend class Runtime;
};
} // namespace infinicore
