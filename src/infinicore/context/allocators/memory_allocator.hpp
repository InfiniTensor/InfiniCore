#pragma once

#include "infinicore/memory.hpp"

#include "../../utils.hpp"
#include <memory>

namespace infinicore {

struct Stat {
    void increase(size_t amount) {
        current += static_cast<int64_t>(amount);
        peak = std::max(current, peak);
        allocated += static_cast<int64_t>(amount);
    }

    void decrease(size_t amount) {
        current -= static_cast<int64_t>(amount);
        INFINICORE_ASSERT(
            current >= 0,
            "Negative tracked stat in device allocator (likely logic error).");
        freed += static_cast<int64_t>(amount);
    }

    void reset_accumulated() {
        allocated = 0;
        freed = 0;
    }

    void reset_peak() {
        peak = current;
    }

    int64_t current = 0;
    int64_t peak = 0;
    int64_t allocated = 0;
    int64_t freed = 0;
};

enum struct StatType : uint64_t {
    AGGREGATE = 0,
    SMALL_POOL = 1,
    LARGE_POOL = 2,
    NUM_TYPES = 3 // remember to update this whenever a new stat type is added
};

using StatArray = std::array<Stat, static_cast<size_t>(StatType::NUM_TYPES)>;
using StatTypes = std::array<bool, static_cast<size_t>(StatType::NUM_TYPES)>;

class MemoryAllocator {
public:
    virtual ~MemoryAllocator() = default;

    virtual std::byte *allocate(size_t size) = 0;
    virtual void deallocate(std::byte *ptr) = 0;
};
} // namespace infinicore
