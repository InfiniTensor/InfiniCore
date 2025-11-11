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

    // Timing methods
    infinirtEvent_t createEvent();
    infinirtEvent_t createEventWithFlags(uint32_t flags);
    void recordEvent(infinirtEvent_t event, infinirtStream_t stream = nullptr);
    bool queryEvent(infinirtEvent_t event);
    void synchronizeEvent(infinirtEvent_t event);
    void destroyEvent(infinirtEvent_t event);
    float elapsedTime(infinirtEvent_t start, infinirtEvent_t end);
    void streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event);

    std::string toString() const;

    friend class ContextImpl;
};
} // namespace infinicore
