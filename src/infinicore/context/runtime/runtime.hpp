#pragma once

#include "../allocators/memory_allocator.hpp"
#include "infinicore/context/context.hpp"

#include <infiniop.h>
#include <infinirt.h>

namespace infinicore {
class ContextImpl;
class Runtime {
private:
    Device device_;
    infinirtStream_t stream_;
    infiniopHandle_t infiniop_handle_;
    std::unique_ptr<MemoryAllocator> device_memory_allocator_;
    std::unique_ptr<MemoryAllocator> pinned_host_memory_allocator_;

protected:
    Runtime(Device device);

public:
    ~Runtime();

    Runtime *activate();

    Device device() const;
    infinirtStream_t stream() const;
    infiniopHandle_t infiniopHandle() const;

    void syncStream();
    void syncDevice();

    std::shared_ptr<Memory> allocateMemory(size_t size);
    std::shared_ptr<Memory> allocatePinnedHostMemory(size_t size);

    void memcpyH2D(void *dst, const void *src, size_t size);
    void memcpyD2H(void *dst, const void *src, size_t size);
    void memcpyD2D(void *dst, const void *src, size_t size);

    friend class ContextImpl;
};
} // namespace infinicore
