#pragma once
#include "infinicore/context/context.hpp"
#include "runtime/runtime.hpp"

#include <array>
#include <mutex>
#include <vector>

namespace infinicore {
class ContextImpl {
private:
    // Table of runtimes for every device (type and index)
    std::array<std::vector<std::unique_ptr<Runtime>>, size_t(Device::Type::COUNT)> runtime_table_;
    // Active runtime for current thread. Can use "static thread local" because context is a process singleton.
    static thread_local Runtime *current_runtime_;
    // Mutex to protect runtime table access
    std::mutex runtime_table_mutex_;

protected:
    ContextImpl();

public:
    Runtime *getCurrentRuntime();

    Runtime *getCpuRuntime();

    Runtime *getRuntime(Device device);

    void setDevice(Device);

    size_t getDeviceCount(Device::Type type);

    static ContextImpl &singleton();

    friend class Runtime;
};
} // namespace infinicore
