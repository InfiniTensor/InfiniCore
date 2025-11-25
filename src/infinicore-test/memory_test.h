#ifndef __INFINICORE_MEMORY_TEST_H__
#define __INFINICORE_MEMORY_TEST_H__

#include "../infinicore/context/allocators/memory_allocator.hpp"
#include "test_runner.h"
#include <atomic>
#include <cassert>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

namespace infinicore::test {

// Mock allocator for testing exception safety
class MockAllocator : public infinicore::MemoryAllocator {
public:
    MockAllocator(bool should_throw = false, size_t max_allocations = SIZE_MAX)
        : should_throw_(should_throw), max_allocations_(max_allocations),
          allocation_count_(0), total_allocated_(0) {}

    std::byte *allocate(size_t size) override {
        if (should_throw_) {
            throw std::runtime_error("Mock allocation failure");
        }
        if (allocation_count_ >= max_allocations_) {
            throw std::runtime_error("Mock allocation limit exceeded");
        }
        allocation_count_++;
        total_allocated_ += size;
        return static_cast<std::byte *>(std::malloc(size));
    }

    void deallocate(std::byte *ptr) override {
        if (ptr) {
            std::free(ptr);
        }
    }

    size_t getAllocationCount() const { return allocation_count_; }
    size_t getTotalAllocated() const { return total_allocated_; }

private:
    bool should_throw_;
    size_t max_allocations_;
    std::atomic<size_t> allocation_count_;
    std::atomic<size_t> total_allocated_;
};

// Memory leak detector
class MemoryLeakDetector {
public:
    static MemoryLeakDetector &instance() {
        static MemoryLeakDetector detector;
        return detector;
    }

    void recordAllocation(void *ptr, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        allocations_[ptr] = size;
        total_allocated_ += size;
    }

    void recordDeallocation(void *ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            total_allocated_ -= it->second;
            allocations_.erase(it);
        }
    }

    size_t getLeakedMemory() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return total_allocated_;
    }

    size_t getLeakCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return allocations_.size();
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        allocations_.clear();
        total_allocated_ = 0;
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<void *, size_t> allocations_;
    size_t total_allocated_ = 0;
};

// Test categories
class BasicMemoryTest : public TestFramework {
public:
    TestResult run() override;
    std::string getName() const override { return "BasicMemoryTest"; }
};

class ConcurrencyTest : public TestFramework {
public:
    TestResult run() override;
    std::string getName() const override { return "ConcurrencyTest"; }

private:
    TestResult testConcurrentAllocations();
    TestResult testConcurrentDeviceSwitching();
    TestResult testMemoryAllocationRace();
};

class ExceptionSafetyTest : public TestFramework {
public:
    TestResult run() override;
    std::string getName() const override { return "ExceptionSafetyTest"; }

private:
    TestResult testAllocationFailure();
    TestResult testDeallocationException();
    TestResult testContextSwitchException();
};

class MemoryLeakTest : public TestFramework {
public:
    TestResult run() override;
    std::string getName() const override { return "MemoryLeakTest"; }

private:
    TestResult testBasicLeakDetection();
    TestResult testCrossDeviceLeakDetection();
    TestResult testExceptionLeakDetection();
};

class PerformanceTest : public TestFramework {
public:
    TestResult run() override;
    std::string getName() const override { return "PerformanceTest"; }

private:
    TestResult testAllocationPerformance();
    TestResult testConcurrentPerformance();
    TestResult testMemoryCopyPerformance();
};

class StressTest : public TestFramework {
public:
    TestResult run() override;
    std::string getName() const override { return "StressTest"; }

private:
    TestResult testHighFrequencyAllocations();
    TestResult testLargeMemoryAllocations();
    TestResult testCrossDeviceStress();
};

class DeviceSwitchTest : public TestFramework {
public:
    TestResult run() override;
    std::string getName() const override { return "DeviceSwitchTest"; }

private:
    TestResult testD2HWithDeviceSwitch();
    TestResult testD2HWithNonContiguousTensor();
};

} // namespace infinicore::test

#endif // __INFINICORE_MEMORY_TEST_H__
