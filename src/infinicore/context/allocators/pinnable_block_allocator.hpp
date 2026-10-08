#pragma once

#include "memory_allocator.hpp"

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace infinicore {
class PinnableBlockAllocator : public MemoryAllocator {
    // Represents a single memory block
    struct Block {
        void *ptr = nullptr; // Device pointer
        size_t size = 0;     // Block size in bytes
        bool frozen = false; // True if used in pinned/graph mode
        uint64_t pin_generation = 0;
        bool in_use = false;  // Wether the block is currently in use
        size_t use_count = 0; // Number of Memory owners for this block
    };

    // A simple size-class allocator for small/medium blocks
    struct SizeClass {
        size_t block_size; // Fixed size for this class
        std::vector<std::shared_ptr<Block>> free_blocks;
    };

public:
    PinnableBlockAllocator(Device device);
    ~PinnableBlockAllocator();

    std::byte *allocate(size_t size) override;
    void deallocate(std::byte *ptr) override;

    // Switch pinned/graph mode
    void set_pin_mode(bool pinned) {
        if (pinned && !pinned_mode_) {
            ++pin_generation_;
        }
        pinned_mode_ = pinned;
    }

    // internal use only, force set in_use flag for a mem block
    // return the size of the block
    size_t mark_in_use_(void *ptr, bool in_use);

    // trim cached blocks back to GPU (not pinned)
    void trim();

private:
    Device device_;

    bool pinned_mode_ = false;
    uint64_t pin_generation_ = 0;

    std::vector<SizeClass> size_classes_;
    std::vector<std::shared_ptr<Block>> large_blocks_;
    std::unordered_map<void *, std::shared_ptr<Block>> all_blocks_;

    std::mutex mutex_; // Thread safety
};

} // namespace infinicore
