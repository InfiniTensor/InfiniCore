#include "device_caching_allocator.hpp"

#include <infinirt.h>
#include <spdlog/spdlog.h>
#include <vector>

#include "../../utils.hpp"

namespace infinicore {

DeviceCachingAllocator::DeviceCachingAllocator(Device device)
    : MemoryAllocator(), device_(device), cached_memory_size_(0) {}

DeviceCachingAllocator::~DeviceCachingAllocator() {
    std::lock_guard<std::mutex> lock(allocator_mutex_);

    SPDLOG_DEBUG("[ALLOCATOR] ~DeviceCachingAllocator: Destructor called - free_blocks_ count={}, active_blocks_ count={}, cached_memory_size_={}",
                 free_blocks_.size(), active_blocks_.size(), cached_memory_size_);

    // During destruction, skip freeing cached blocks to avoid CUDA errors
    // The Runtime has already synchronized streams/devices before destroying the allocator
    // Memory will be reclaimed by the OS when the process exits
    // Attempting to free during destruction can cause errors if streams/contexts are invalid
    if (!free_blocks_.empty()) {
        SPDLOG_DEBUG("[ALLOCATOR] ~DeviceCachingAllocator: Skipping free of {} cached blocks ({} bytes) - will be reclaimed by OS",
                     free_blocks_.size(), cached_memory_size_);
    }
    free_blocks_.clear();
    cached_memory_size_ = 0;

    // Check for any remaining active blocks (shouldn't happen in normal operation)
    if (!active_blocks_.empty()) {
        SPDLOG_WARN("[ALLOCATOR] ~DeviceCachingAllocator: {} active blocks remain (shouldn't happen) - will be reclaimed by OS",
                    active_blocks_.size());
    }
    active_blocks_.clear();
}

std::byte *DeviceCachingAllocator::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(allocator_mutex_);

    // Try to get from cache first
    MemoryBlock cached_block(nullptr, 0, nullptr);
    if (tryGetFromCache(size, cached_block)) {
        // Reuse from cache (tryGetFromCache already subtracted from cached_memory_size_)
        void *ptr = cached_block.ptr;

        SPDLOG_DEBUG("[ALLOCATOR] allocate: Reusing from cache - ptr={}, size={}, stream={}, active_blocks_count={}",
                     ptr, cached_block.size, cached_block.stream, active_blocks_.size());

        // Mark as active
        active_blocks_[ptr] = cached_block;

        SPDLOG_DEBUG("[ALLOCATOR] allocate: Added to active_blocks_ - ptr={}, size={}, stream={}, active_blocks_count={}",
                     ptr, cached_block.size, cached_block.stream, active_blocks_.size());

        return (std::byte *)ptr;
    }

    // Cache miss - allocate new memory
    infinirtStream_t stream = context::getStream();
    void *ptr = nullptr;
    INFINICORE_CHECK_ERROR(infinirtMallocAsync(&ptr, size, stream));

    SPDLOG_DEBUG("[ALLOCATOR] allocate: New allocation - ptr={}, size={}, stream={}, active_blocks_count={}",
                 ptr, size, stream, active_blocks_.size());

    // Track as active block
    active_blocks_[ptr] = MemoryBlock(ptr, size, stream);

    SPDLOG_DEBUG("[ALLOCATOR] allocate: Added to active_blocks_ - ptr={}, size={}, stream={}, active_blocks_count={}",
                 ptr, size, stream, active_blocks_.size());

    return (std::byte *)ptr;
}

void DeviceCachingAllocator::deallocate(std::byte *ptr) {
    if (ptr == nullptr) {
        return; // Nothing to free
    }

    std::lock_guard<std::mutex> lock(allocator_mutex_);

    SPDLOG_DEBUG("[ALLOCATOR] deallocate: Requested - ptr={}, active_blocks_count={}",
                 static_cast<void *>(ptr), active_blocks_.size());

    // Check if this is an active block we're tracking
    auto it = active_blocks_.find(ptr);
    if (it == active_blocks_.end()) {
        // Not tracked - might be from before caching was implemented
        // Free it immediately (fallback)
        SPDLOG_WARN("[ALLOCATOR] deallocate: ptr={} not found in active_blocks_, freeing immediately (fallback)",
                    static_cast<void *>(ptr));
        infinirtStream_t stream = context::getStream();
        infinirtStreamSynchronize(stream);
        INFINICORE_CHECK_ERROR(infinirtFreeAsync(ptr, stream));
        // Use device sync to ensure free completes
        infinirtDeviceSynchronize();
        return;
    }

    // Get block info
    MemoryBlock block = it->second;
    SPDLOG_DEBUG("[ALLOCATOR] deallocate: Found in active_blocks_ - ptr={}, size={}, stream={}",
                 block.ptr, block.size, block.stream);

    active_blocks_.erase(it);
    SPDLOG_DEBUG("[ALLOCATOR] deallocate: Removed from active_blocks_ - ptr={}, active_blocks_count={}",
                 static_cast<void *>(ptr), active_blocks_.size());

    // Return to cache instead of freeing immediately (PyTorch-style)
    returnToCache(block.ptr, block.size, block.stream);

    // Cleanup cache if it's too large
    if (cached_memory_size_ > MAX_CACHE_SIZE) {
        cleanupCache();
    }
}

void DeviceCachingAllocator::returnToCache(void *ptr, size_t size, infinirtStream_t stream) {
    // Cache all blocks regardless of size (PyTorch-style)
    // This avoids double-free issues when the same pointer is quickly reused
    // Small blocks will be freed during cache cleanup if cache gets too large
    SPDLOG_DEBUG("[ALLOCATOR] returnToCache: Adding to cache - ptr={}, size={}, stream={}, cached_memory_size_={}",
                 ptr, size, stream, cached_memory_size_);
    free_blocks_[size].emplace_back(ptr, size, stream);
    cached_memory_size_ += size;
    SPDLOG_DEBUG("[ALLOCATOR] returnToCache: Added to cache - ptr={}, size={}, stream={}, cached_memory_size_={}, free_blocks_[{}].size()={}",
                 ptr, size, stream, cached_memory_size_, size, free_blocks_[size].size());
}

bool DeviceCachingAllocator::tryGetFromCache(size_t size, MemoryBlock &out_block) {
    // Try exact size match first
    auto it = free_blocks_.find(size);
    if (it != free_blocks_.end() && !it->second.empty()) {
        out_block = it->second.front();
        SPDLOG_DEBUG("[ALLOCATOR] tryGetFromCache: Exact match found - requested_size={}, block_ptr={}, block_size={}, block_stream={}",
                     size, out_block.ptr, out_block.size, out_block.stream);
        it->second.pop_front();
        if (it->second.empty()) {
            free_blocks_.erase(it);
        }
        cached_memory_size_ -= out_block.size;
        SPDLOG_DEBUG("[ALLOCATOR] tryGetFromCache: Removed from cache - ptr={}, cached_memory_size_={}",
                     out_block.ptr, cached_memory_size_);
        return true;
    }

    // Try to find a larger block we can use (simple first-fit)
    // This helps reduce fragmentation
    size_t best_size = SIZE_MAX;
    std::unordered_map<size_t, std::list<MemoryBlock>>::iterator best_it = free_blocks_.end();

    for (auto map_it = free_blocks_.begin(); map_it != free_blocks_.end(); ++map_it) {
        if (map_it->first >= size && map_it->first < best_size && !map_it->second.empty()) {
            best_size = map_it->first;
            best_it = map_it;
        }
    }

    if (best_it != free_blocks_.end()) {
        out_block = best_it->second.front();
        SPDLOG_DEBUG("[ALLOCATOR] tryGetFromCache: Larger block found - requested_size={}, block_ptr={}, block_size={}, block_stream={}",
                     size, out_block.ptr, out_block.size, out_block.stream);
        best_it->second.pop_front();
        if (best_it->second.empty()) {
            free_blocks_.erase(best_it);
        }
        cached_memory_size_ -= out_block.size;
        SPDLOG_DEBUG("[ALLOCATOR] tryGetFromCache: Removed from cache - ptr={}, cached_memory_size_={}",
                     out_block.ptr, cached_memory_size_);
        return true;
    }

    SPDLOG_DEBUG("[ALLOCATOR] tryGetFromCache: Cache miss - requested_size={}, cached_memory_size_={}",
                 size, cached_memory_size_);
    return false; // Cache miss
}

void DeviceCachingAllocator::cleanupCache() {
    // Cleanup strategy: free half of the cached blocks (oldest first)
    // This keeps some cache for performance while preventing unbounded growth

    size_t target_size = MAX_CACHE_SIZE / 2;
    size_t freed_size = 0;

    // Collect blocks to free (iterate through all sizes)
    std::vector<MemoryBlock> blocks_to_free;

    for (auto &[size, blocks] : free_blocks_) {
        while (!blocks.empty() && freed_size < target_size) {
            MemoryBlock block = blocks.front();
            blocks.pop_front();
            blocks_to_free.push_back(block);
            freed_size += block.size;
        }

        // Remove empty lists
        if (blocks.empty()) {
            // Will be cleaned up in next iteration or when map is accessed
        }
    }

    // Actually free the blocks (synchronize streams first)
    for (auto &block : blocks_to_free) {
        SPDLOG_DEBUG("[ALLOCATOR] cleanupCache: Freeing block - ptr={}, size={}, stream={}",
                     block.ptr, block.size, block.stream);
        // Synchronize stream to ensure all operations using this memory complete
        infinirtStreamSynchronize(block.stream);
        // Queue the free operation
        INFINICORE_CHECK_ERROR(infinirtFreeAsync(block.ptr, block.stream));
        // Synchronize again to ensure the free completes
        infinirtStreamSynchronize(block.stream);
        cached_memory_size_ -= block.size;
        SPDLOG_DEBUG("[ALLOCATOR] cleanupCache: Freed block - ptr={}, cached_memory_size_={}",
                     block.ptr, cached_memory_size_);
    }

    // Clean up empty entries
    for (auto it = free_blocks_.begin(); it != free_blocks_.end();) {
        if (it->second.empty()) {
            it = free_blocks_.erase(it);
        } else {
            ++it;
        }
    }

    if (!blocks_to_free.empty()) {
        SPDLOG_DEBUG("[ALLOCATOR] DeviceCachingAllocator: Cleaned up {} blocks, freed {} bytes",
                     blocks_to_free.size(), freed_size);
    }
}

} // namespace infinicore
