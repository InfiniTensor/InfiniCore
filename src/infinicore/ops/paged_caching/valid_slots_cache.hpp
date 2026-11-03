#pragma once
#include <cstdint>
#include <unordered_set>
#include <vector>
namespace infinicore::op::paged_caching_impl::infiniop {
// Reuse validation only after comparing every slot value and the capacity.
// A reused tensor address alone does not identify the host array contents.
class ValidSlotsCache {
public:
    bool validate(const std::vector<int64_t> &slots, int64_t capacity) {
        if (initialized_ && capacity_ == capacity && values_ == slots) {
            return valid_;
        }
        initialized_ = true;
        capacity_ = capacity;
        values_ = slots;
        valid_ = true;
        std::unordered_set<int64_t> seen;
        seen.reserve(slots.size());
        for (const auto slot : slots) {
            if (slot < 0 || slot >= capacity || !seen.insert(slot).second) {
                valid_ = false;
                break;
            }
        }
        return valid_;
    }

private:
    bool initialized_ = false, valid_ = false;
    int64_t capacity_ = 0;
    std::vector<int64_t> values_;
};
} // namespace infinicore::op::paged_caching_impl::infiniop
