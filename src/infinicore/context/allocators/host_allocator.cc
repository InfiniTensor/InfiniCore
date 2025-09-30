#include "host_allocator.hpp"

#include <infinirt.h>

namespace infinicore {
std::byte *HostAllocator::allocate(size_t size) {
    return (std::byte *)std::malloc(size);
}

void HostAllocator::deallocate(std::byte *ptr) {
    std::free(ptr);
}

} // namespace infinicore
