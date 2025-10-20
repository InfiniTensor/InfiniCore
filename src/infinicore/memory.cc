#include "infinicore/memory.hpp"

#include "spdlog/spdlog.h"
namespace infinicore {

Memory::Memory(std::byte *data,
               size_t size,
               Device device,
               Memory::Deleter deleter,
               bool pin_memory)
    : data_{data}, size_{size}, device_{device}, deleter_{deleter}, is_pinned_(pin_memory) {
    // Register this memory allocation in the pool
    MemoryPool::instance().registerMemory(data, size);
    spdlog::debug("Memory::Memory() created for memory={} at Device: {}",
                  static_cast<void *>(data), device.toString());
}

Memory::~Memory() {
    if (data_ && deleter_) {
        spdlog::debug("Memory::~Memory() called for memory={} at Device: {}",
                      static_cast<void *>(data_), device_.toString());
        // Use memory pool to manage reference counting and actual freeing
        MemoryPool::instance().releaseMemory(data_, deleter_);
    }
}

Memory::Memory(const Memory &other)
    : data_{other.data_}, size_{other.size_}, device_{other.device_},
      deleter_{other.deleter_}, is_pinned_{other.is_pinned_} {
    if (data_) {
        MemoryPool::instance().addRef(data_);
        spdlog::debug("Memory::Memory(const Memory&) copy constructor called for memory={}",
                      static_cast<void *>(data_));
    }
}

Memory &Memory::operator=(const Memory &other) {
    if (this != &other) {
        // Release current memory if it exists
        if (data_ && deleter_) {
            MemoryPool::instance().releaseMemory(data_, deleter_);
        }

        // Copy from other
        data_ = other.data_;
        size_ = other.size_;
        device_ = other.device_;
        deleter_ = other.deleter_;
        is_pinned_ = other.is_pinned_;

        // Add reference to new memory
        if (data_) {
            MemoryPool::instance().addRef(data_);
        }

        spdlog::debug("Memory::operator=(const Memory&) copy assignment called for memory={}",
                      static_cast<void *>(data_));
    }
    return *this;
}

Memory::Memory(Memory &&other) noexcept
    : data_{other.data_}, size_{other.size_}, device_{other.device_},
      deleter_{std::move(other.deleter_)}, is_pinned_{other.is_pinned_} {
    // Clear the moved-from object to prevent double-free
    other.data_ = nullptr;
    other.deleter_ = nullptr;
    spdlog::debug("Memory::Memory(Memory&&) move constructor called");
}

Memory &Memory::operator=(Memory &&other) noexcept {
    if (this != &other) {
        // Release current memory if it exists
        if (data_ && deleter_) {
            MemoryPool::instance().releaseMemory(data_, deleter_);
        }

        // Move from other
        data_ = other.data_;
        size_ = other.size_;
        device_ = other.device_;
        deleter_ = std::move(other.deleter_);
        is_pinned_ = other.is_pinned_;

        // Clear the moved-from object
        other.data_ = nullptr;
        other.deleter_ = nullptr;

        spdlog::debug("Memory::operator=(Memory&&) move assignment called");
    }
    return *this;
}

std::byte *Memory::data() const {
    return data_;
}

Device Memory::device() const {
    return device_;
}

size_t Memory::size() const {
    return size_;
}

bool Memory::is_pinned() const {
    return is_pinned_;
}
} // namespace infinicore
