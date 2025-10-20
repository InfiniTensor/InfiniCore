#ifndef __INFINICORE_MEMORY_TEST_H__
#define __INFINICORE_MEMORY_TEST_H__

#include "../infinicore/context/allocators/memory_allocator.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <exception>
#include <future>
#include <infinicore.hpp>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <spdlog/spdlog.h>
#include <thread>
#include <unordered_map>
#include <vector>

namespace infinicore::test {

// Test result structure
struct TestResult {
    std::string test_name;
    bool passed;
    std::string error_message;
    std::chrono::microseconds duration;

    TestResult(const std::string &name, bool pass, const std::string &error = "",
               std::chrono::microseconds dur = std::chrono::microseconds(0))
        : test_name(name), passed(pass), error_message(error), duration(dur) {}
};

// Test framework base class
class MemoryTestFramework {
public:
    virtual ~MemoryTestFramework() = default;
    virtual TestResult run() = 0;
    virtual std::string getName() const = 0;

protected:
    void logTestStart(const std::string &test_name) {
        std::cout << "[TEST] Starting: " << test_name << std::endl;
    }

    void logTestResult(const TestResult &result) {
        std::cout << "[TEST] " << (result.passed ? "PASSED" : "FAILED")
                  << ": " << result.test_name;
        if (!result.passed && !result.error_message.empty()) {
            std::cout << " - " << result.error_message;
        }
        std::cout << " (Duration: " << result.duration.count() << "μs)" << std::endl;
    }

    template <typename Func>
    TestResult measureTime(const std::string &test_name, Func &&func) {
        auto start = std::chrono::high_resolution_clock::now();
        try {
            bool result = func();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            return TestResult(test_name, result, "", duration);
        } catch (const std::exception &e) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            return TestResult(test_name, false, e.what(), duration);
        }
    }
};

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
class BasicMemoryTest : public MemoryTestFramework {
public:
    TestResult run() override;
    std::string getName() const override { return "BasicMemoryTest"; }
};

class ConcurrencyTest : public MemoryTestFramework {
public:
    ConcurrencyTest(int num_threads = 4) : num_threads_(num_threads) {}
    TestResult run() override;
    std::string getName() const override { return "ConcurrencyTest"; }

private:
    int num_threads_;
    TestResult testConcurrentAllocations();
    TestResult testConcurrentDeviceSwitching();
    TestResult testMemoryAllocationRace();
};

class ExceptionSafetyTest : public MemoryTestFramework {
public:
    TestResult run() override;
    std::string getName() const override { return "ExceptionSafetyTest"; }

private:
    TestResult testAllocationFailure();
    TestResult testDeallocationException();
    TestResult testContextSwitchException();
};

class MemoryLeakTest : public MemoryTestFramework {
public:
    TestResult run() override;
    std::string getName() const override { return "MemoryLeakTest"; }

private:
    TestResult testBasicLeakDetection();
    TestResult testCrossDeviceLeakDetection();
    TestResult testExceptionLeakDetection();
};

class PerformanceTest : public MemoryTestFramework {
public:
    TestResult run() override;
    std::string getName() const override { return "PerformanceTest"; }

private:
    TestResult testAllocationPerformance();
    TestResult testConcurrentPerformance();
    TestResult testMemoryCopyPerformance();
};

class StressTest : public MemoryTestFramework {
public:
    StressTest(int iterations = 1000) : iterations_(iterations) {}
    TestResult run() override;
    std::string getName() const override { return "StressTest"; }

private:
    int iterations_;
    TestResult testHighFrequencyAllocations();
    TestResult testLargeMemoryAllocations();
    TestResult testCrossDeviceStress();
};

// Test runner
class MemoryTestRunner {
public:
    void addTest(std::unique_ptr<MemoryTestFramework> test) {
        tests_.push_back(std::move(test));
    }

    std::vector<TestResult> runAllTests() {
        std::vector<TestResult> results;

        std::cout << "==============================================\n"
                  << "InfiniCore Memory Management Test Suite\n"
                  << "==============================================" << std::endl;

        for (auto &test : tests_) {
            logTestStart(test->getName());
            TestResult result = test->run();
            logTestResult(result);
            results.push_back(result);
        }

        printSummary(results);
        return results;
    }

private:
    std::vector<std::unique_ptr<MemoryTestFramework>> tests_;

    void logTestStart(const std::string &test_name) {
        std::cout << "\n[SUITE] Running: " << test_name << std::endl;
    }

    void logTestResult(const TestResult &result) {
        std::cout << "[SUITE] " << (result.passed ? "PASSED" : "FAILED")
                  << ": " << result.test_name << std::endl;
    }

    void printSummary(const std::vector<TestResult> &results) {
        size_t passed = 0, failed = 0;
        std::chrono::microseconds total_time(0);

        for (const auto &result : results) {
            if (result.passed) {
                passed++;
            } else {
                failed++;
            }
            total_time += result.duration;
        }

        std::cout << "\n==============================================\n"
                  << "Test Summary\n"
                  << "==============================================\n"
                  << "Total Tests: " << results.size() << "\n"
                  << "Passed: " << passed << "\n"
                  << "Failed: " << failed << "\n"
                  << "Total Time: " << total_time.count() << "μs\n"
                  << "==============================================" << std::endl;
    }
};

} // namespace infinicore::test

#endif // __INFINICORE_MEMORY_TEST_H__
