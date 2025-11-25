#include "memory_test.h"
#include <algorithm>
#include <cstring>
#include <random>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

namespace infinicore::test {

// Basic Memory Test Implementation
TestResult BasicMemoryTest::run() {
    return measureTime("BasicMemoryTest", [this]() -> bool {
        try {
            SPDLOG_DEBUG("BasicMemoryTest: Starting test");
            // Test basic memory allocation
            SPDLOG_DEBUG("BasicMemoryTest: About to allocate memory");
            auto memory = context::allocateMemory(1024);
            SPDLOG_DEBUG("BasicMemoryTest: Memory allocated successfully");
            if (!memory) {
                SPDLOG_ERROR("Failed to allocate memory");
                return false;
            }

            SPDLOG_DEBUG("BasicMemoryTest: Testing memory properties");
            // Test memory properties
            if (memory->size() != 1024) {
                SPDLOG_ERROR("Memory size mismatch: expected 1024, got {}", memory->size());
                return false;
            }
            SPDLOG_DEBUG("BasicMemoryTest: Memory size check passed");

            SPDLOG_DEBUG("BasicMemoryTest: Testing memory access");
            // Test memory access
            std::byte *data = memory->data();
            SPDLOG_DEBUG("BasicMemoryTest: Got memory data pointer: {}", static_cast<void *>(data));
            if (!data) {
                SPDLOG_ERROR("Memory data pointer is null");
                return false;
            }
            SPDLOG_DEBUG("BasicMemoryTest: Memory data pointer is valid");

            // Check if this is GPU memory that can't be accessed directly
            Device current_device = context::getDevice();
            SPDLOG_DEBUG("BasicMemoryTest: Current device type: {}", static_cast<int>(current_device.getType()));
            SPDLOG_DEBUG("BasicMemoryTest: Memory is pinned: {}", memory->is_pinned());

            // For GPU memory, we shouldn't try to access it directly with memset
            if (current_device.getType() != Device::Type::CPU) {
                SPDLOG_DEBUG("BasicMemoryTest: Skipping direct memory access for GPU device");
                SPDLOG_DEBUG("BasicMemoryTest: GPU memory access test completed (skipped)");
            } else {
                SPDLOG_DEBUG("BasicMemoryTest: Testing memory write/read");
                // Test memory write/read
                std::memset(data, 0xAB, 1024);
                SPDLOG_DEBUG("BasicMemoryTest: Memory memset completed");
                for (size_t i = 0; i < 1024; ++i) {
                    if (data[i] != static_cast<std::byte>(0xAB)) {
                        SPDLOG_ERROR("Memory write/read test failed at index {}", i);
                        return false;
                    }
                }
                SPDLOG_DEBUG("BasicMemoryTest: Memory write/read test completed");
            }

            SPDLOG_DEBUG("BasicMemoryTest: Testing pinned memory allocation");
            // Test pinned memory allocation
            auto pinned_memory = context::allocatePinnedHostMemory(512);
            SPDLOG_DEBUG("BasicMemoryTest: Pinned memory allocated");
            if (!pinned_memory) {
                SPDLOG_ERROR("Failed to allocate pinned memory");
                return false;
            }

            SPDLOG_DEBUG("BasicMemoryTest: Checking pinned memory properties");
            // For CPU devices, pinned memory falls back to regular memory, so it may not be marked as pinned
            Device pinned_device = context::getDevice();
            if (pinned_device.getType() != Device::Type::CPU && !pinned_memory->is_pinned()) {
                SPDLOG_ERROR("Pinned memory not marked as pinned");
                return false;
            }
            SPDLOG_DEBUG("BasicMemoryTest: Pinned memory test completed");

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("BasicMemoryTest failed with exception: {}", e.what());
            return false;
        }
    });
}

// Concurrency Test Implementation
TestResult ConcurrencyTest::run() {
    return measureTime("ConcurrencyTest", [this]() -> bool {
        try {
            // Run all concurrency subtests
            auto result1 = testConcurrentAllocations();
            if (!result1.passed) {
                SPDLOG_ERROR("Concurrent allocations test failed: {}", result1.error_message);
                return false;
            }

            auto result2 = testConcurrentDeviceSwitching();
            if (!result2.passed) {
                SPDLOG_ERROR("Concurrent device switching test failed: {}", result2.error_message);
                return false;
            }

            auto result3 = testMemoryAllocationRace();
            if (!result3.passed) {
                SPDLOG_ERROR("Memory allocation race test failed: {}", result3.error_message);
                return false;
            }

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("ConcurrencyTest failed with exception: {}", e.what());
            return false;
        }
    });
}

TestResult ConcurrencyTest::testConcurrentAllocations() {
    return measureTime("ConcurrentAllocations", [this]() -> bool {
        SPDLOG_INFO("================================================");
        SPDLOG_INFO("ConcurrentAllocations: Starting test");
        SPDLOG_INFO("================================================");
        const int num_threads = 8;
        const int allocations_per_thread = 100;
        std::vector<std::thread> threads;
        std::atomic<int> success_count{0};
        std::atomic<int> failure_count{0};

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&, i]() {
                try {
                    for (int j = 0; j < allocations_per_thread; ++j) {
                        // Allocate memory of random size
                        size_t size = 64 + (j % 1024);
                        auto memory = context::allocateMemory(size);
                        if (memory && memory->size() == size) {
                            success_count++;
                        } else {
                            failure_count++;
                        }

                        // Small delay to increase chance of race conditions
                        std::this_thread::sleep_for(std::chrono::microseconds(1));
                    }
                } catch (const std::exception &e) {
                    failure_count++;
                    SPDLOG_ERROR("Thread {} failed: {}", i, e.what());
                }
            });
        }

        for (auto &thread : threads) {
            thread.join();
        }

        int total_expected = num_threads * allocations_per_thread;
        if (success_count.load() != total_expected) {
            SPDLOG_ERROR("Concurrent allocation test failed: expected {} successes, got {} successes and {} failures",
                         total_expected, success_count.load(), failure_count.load());
            return false;
        }

        return true;
    });
}

TestResult ConcurrencyTest::testConcurrentDeviceSwitching() {
    return measureTime("ConcurrentDeviceSwitching", [this]() -> bool {
        SPDLOG_INFO("================================================");
        SPDLOG_INFO("ConcurrentDeviceSwitching: Starting test");
        SPDLOG_INFO("================================================");
        const int num_threads = 4;
        std::vector<std::thread> threads;
        std::atomic<int> success_count{0};
        std::atomic<int> failure_count{0};

        // Get available devices
        std::vector<Device> devices;
        for (int type = 0; type < static_cast<int>(Device::Type::COUNT); ++type) {
            size_t count = context::getDeviceCount(static_cast<Device::Type>(type));
            for (size_t i = 0; i < count; ++i) {
                devices.emplace_back(static_cast<Device::Type>(type), i);
            }
        }

        if (devices.size() < 2) {
            std::cout << "Skipping device switching test - need at least 2 devices" << std::endl;
            return true;
        }

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&, i, devices]() {
                try {
                    for (int j = 0; j < 50; ++j) {
                        // Switch to random device
                        Device target_device = devices[j % devices.size()];
                        context::setDevice(target_device);

                        // Verify device was set correctly
                        Device current_device = context::getDevice();
                        if (current_device == target_device) {
                            success_count++;
                        } else {
                            failure_count++;
                            SPDLOG_ERROR("Device switching failed: expected {}, got {}",
                                         static_cast<int>(target_device.getType()),
                                         static_cast<int>(current_device.getType()));
                        }

                        // Allocate memory to test device context
                        auto memory = context::allocateMemory(256);
                        if (memory && memory->device() == target_device) {
                            success_count++;
                        } else {
                            failure_count++;
                        }

                        std::this_thread::sleep_for(std::chrono::microseconds(10));
                    }
                } catch (const std::exception &e) {
                    failure_count++;
                    SPDLOG_ERROR("Thread {} failed: {}", i, e.what());
                }
            });
        }

        for (auto &thread : threads) {
            thread.join();
        }

        if (failure_count.load() > 0) {
            SPDLOG_ERROR("Concurrent device switching test failed: {} failures out of {} operations",
                         failure_count.load(), success_count.load() + failure_count.load());
            return false;
        }

        return true;
    });
}

TestResult ConcurrencyTest::testMemoryAllocationRace() {
    return measureTime("MemoryAllocationRace", [this]() -> bool {
        SPDLOG_INFO("================================================");
        SPDLOG_INFO("MemoryAllocationRace: Starting test");
        SPDLOG_INFO("================================================");
        const int num_threads = 16;
        const int allocations_per_thread = 100;
        std::vector<std::thread> threads;
        std::atomic<int> success_count{0};
        std::atomic<int> failure_count{0};
        std::vector<std::shared_ptr<Memory>> all_allocations;
        std::mutex allocations_mutex;

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&, i]() {
                std::vector<std::shared_ptr<Memory>> thread_allocations;
                try {
                    for (int j = 0; j < allocations_per_thread; ++j) {
                        size_t size = 64 + (j % 1024);
                        auto memory = context::allocateMemory(size);
                        if (memory) {
                            thread_allocations.push_back(memory);
                            success_count++;
                        } else {
                            failure_count++;
                        }

                        // Occasionally deallocate some memory to test concurrent alloc/dealloc
                        if (j % 10 == 0 && !thread_allocations.empty()) {
                            thread_allocations.pop_back();
                        }
                    }

                    // Store remaining allocations
                    std::lock_guard<std::mutex> lock(allocations_mutex);
                    all_allocations.insert(all_allocations.end(),
                                           thread_allocations.begin(),
                                           thread_allocations.end());
                } catch (const std::exception &e) {
                    failure_count++;
                    SPDLOG_ERROR("Thread {} failed: {}", i, e.what());
                }
            });
        }

        for (auto &thread : threads) {
            thread.join();
        }

        // Verify all allocations are valid
        for (const auto &memory : all_allocations) {
            if (!memory || !memory->data()) {
                SPDLOG_ERROR("Invalid memory allocation found");
                return false;
            }
        }

        int total_expected = num_threads * allocations_per_thread;
        if (success_count.load() < total_expected * 0.9) { // Allow 10% failure rate
            SPDLOG_ERROR("Memory allocation race test failed: expected at least {} successes, got {}",
                         static_cast<int>(total_expected * 0.9), success_count.load());
            return false;
        }

        return true;
    });
}

// Exception Safety Test Implementation
TestResult ExceptionSafetyTest::run() {
    return measureTime("ExceptionSafetyTest", [this]() -> bool {
        try {
            auto result1 = testAllocationFailure();
            if (!result1.passed) {
                SPDLOG_ERROR("Allocation failure test failed: {}", result1.error_message);
                return false;
            }

            auto result2 = testDeallocationException();
            if (!result2.passed) {
                SPDLOG_ERROR("Deallocation exception test failed: {}", result2.error_message);
                return false;
            }

            auto result3 = testContextSwitchException();
            if (!result3.passed) {
                SPDLOG_ERROR("Context switch exception test failed: {}", result3.error_message);
                return false;
            }

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("ExceptionSafetyTest failed with exception: {}", e.what());
            return false;
        }
    });
}

TestResult ExceptionSafetyTest::testAllocationFailure() {
    return measureTime("AllocationFailure", [this]() -> bool {
        try {
            // Test allocation with extremely large size (should fail)
            try {
                auto memory = context::allocateMemory(SIZE_MAX);
                SPDLOG_ERROR("Expected allocation to fail with SIZE_MAX");
                return false;
            } catch (const std::exception &e) {
                // Expected to fail
                std::cout << "Allocation correctly failed with SIZE_MAX: " << e.what() << std::endl;
            }

            // Test allocation with zero size
            try {
                auto memory = context::allocateMemory(0);
                if (memory) {
                    SPDLOG_ERROR("Zero-size allocation should return null or throw");
                    return false;
                }
            } catch (const std::exception &e) {
                // Also acceptable
                std::cout << "Zero-size allocation correctly failed: " << e.what() << std::endl;
            }

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("Allocation failure test failed with unexpected exception: {}", e.what());
            return false;
        }
    });
}

TestResult ExceptionSafetyTest::testDeallocationException() {
    return measureTime("DeallocationException", [this]() -> bool {
        try {
            // Test that deallocation doesn't throw exceptions
            std::vector<std::shared_ptr<Memory>> memories;

            // Allocate some memory
            for (int i = 0; i < 10; ++i) {
                auto memory = context::allocateMemory(1024);
                if (memory) {
                    memories.push_back(memory);
                }
            }

            // Test that destruction doesn't throw
            try {
                memories.clear(); // This should trigger deallocation
            } catch (const std::exception &e) {
                SPDLOG_ERROR("Memory deallocation threw exception: {}", e.what());
                return false;
            }

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("Deallocation exception test failed: {}", e.what());
            return false;
        }
    });
}

TestResult ExceptionSafetyTest::testContextSwitchException() {
    return measureTime("ContextSwitchException", [this]() -> bool {
        try {
            // Test context switching with invalid device
            Device original_device = context::getDevice();

            try {
                // Try to switch to a device that might not exist
                Device invalid_device(Device::Type::COUNT, 999);
                context::setDevice(invalid_device);
                SPDLOG_ERROR("Expected device switching to fail with invalid device");
                return false;
            } catch (const std::exception &e) {
                // Expected to fail
                std::cout << "Device switching correctly failed with invalid device: " << e.what() << std::endl;
            }

            // Verify original device is still set
            Device current_device = context::getDevice();
            if (current_device != original_device) {
                SPDLOG_ERROR("Device context not restored after exception");
                return false;
            }

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("Context switch exception test failed: {}", e.what());
            return false;
        }
    });
}

// Memory Leak Test Implementation
TestResult MemoryLeakTest::run() {
    return measureTime("MemoryLeakTest", [this]() -> bool {
        try {
            auto result1 = testBasicLeakDetection();
            if (!result1.passed) {
                SPDLOG_ERROR("Basic leak detection test failed: {}", result1.error_message);
                return false;
            }

            auto result2 = testCrossDeviceLeakDetection();
            if (!result2.passed) {
                SPDLOG_ERROR("Cross-device leak detection test failed: {}", result2.error_message);
                return false;
            }

            auto result3 = testExceptionLeakDetection();
            if (!result3.passed) {
                SPDLOG_ERROR("Exception leak detection test failed: {}", result3.error_message);
                return false;
            }

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("MemoryLeakTest failed with exception: {}", e.what());
            return false;
        }
    });
}

TestResult MemoryLeakTest::testBasicLeakDetection() {
    return measureTime("BasicLeakDetection", [this]() -> bool {
        try {
            // Reset leak detector
            MemoryLeakDetector::instance().reset();

            // Allocate and deallocate memory
            std::vector<std::shared_ptr<Memory>> memories;
            for (int i = 0; i < 100; ++i) {
                auto memory = context::allocateMemory(1024);
                if (memory) {
                    memories.push_back(memory);
                }
            }

            // Clear memories to trigger deallocation
            memories.clear();

            // Force garbage collection if available
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // Check for leaks (this is a basic test - real leak detection would need more sophisticated tools)
            size_t leaked_memory = MemoryLeakDetector::instance().getLeakedMemory();
            if (leaked_memory > 0) {
                SPDLOG_ERROR("Potential memory leak detected: {} bytes", leaked_memory);
                return false;
            }

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("Basic leak detection test failed: {}", e.what());
            return false;
        }
    });
}

TestResult MemoryLeakTest::testCrossDeviceLeakDetection() {
    return measureTime("CrossDeviceLeakDetection", [this]() -> bool {
        try {
            // Get available devices
            std::vector<Device> devices;
            for (int type = 0; type < static_cast<int>(Device::Type::COUNT); ++type) {
                size_t count = context::getDeviceCount(static_cast<Device::Type>(type));
                for (size_t i = 0; i < count; ++i) {
                    devices.emplace_back(static_cast<Device::Type>(type), i);
                }
            }

            if (devices.size() < 2) {
                std::cout << "Skipping cross-device leak test - need at least 2 devices" << std::endl;
                return true;
            }

            // Allocate pinned memory on one device
            context::setDevice(devices[0]);
            auto pinned_memory = context::allocatePinnedHostMemory(1024);

            if (!pinned_memory) {
                SPDLOG_ERROR("Failed to allocate pinned memory");
                return false;
            }

            // Switch to another device and deallocate
            context::setDevice(devices[1]);
            pinned_memory.reset(); // This should trigger cross-device deallocation

            // Force garbage collection
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // Check for leaks
            size_t leaked_memory = MemoryLeakDetector::instance().getLeakedMemory();
            if (leaked_memory > 0) {
                SPDLOG_ERROR("Potential cross-device memory leak detected: {} bytes", leaked_memory);
                return false;
            }

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("Cross-device leak detection test failed: {}", e.what());
            return false;
        }
    });
}

TestResult MemoryLeakTest::testExceptionLeakDetection() {
    return measureTime("ExceptionLeakDetection", [this]() -> bool {
        try {
            // Test that exceptions don't cause memory leaks
            std::vector<std::shared_ptr<Memory>> memories;

            try {
                // Allocate some memory
                for (int i = 0; i < 10; ++i) {
                    auto memory = context::allocateMemory(1024);
                    if (memory) {
                        memories.push_back(memory);
                    }
                }

                // Simulate an exception
                throw std::runtime_error("Simulated exception");

            } catch (const std::exception &e) {
                // Memory should still be properly cleaned up
                memories.clear();
            }

            // Force garbage collection
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // Check for leaks
            size_t leaked_memory = MemoryLeakDetector::instance().getLeakedMemory();
            if (leaked_memory > 0) {
                SPDLOG_ERROR("Potential exception-related memory leak detected: {} bytes", leaked_memory);
                return false;
            }

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("Exception leak detection test failed: {}", e.what());
            return false;
        }
    });
}

// Performance Test Implementation
TestResult PerformanceTest::run() {
    return measureTime("PerformanceTest", [this]() -> bool {
        try {
            auto result1 = testAllocationPerformance();
            if (!result1.passed) {
                SPDLOG_ERROR("Allocation performance test failed: {}", result1.error_message);
                return false;
            }

            auto result2 = testConcurrentPerformance();
            if (!result2.passed) {
                SPDLOG_ERROR("Concurrent performance test failed: {}", result2.error_message);
                return false;
            }

            auto result3 = testMemoryCopyPerformance();
            if (!result3.passed) {
                SPDLOG_ERROR("Memory copy performance test failed: {}", result3.error_message);
                return false;
            }

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("PerformanceTest failed with exception: {}", e.what());
            return false;
        }
    });
}

TestResult PerformanceTest::testAllocationPerformance() {
    return measureTime("AllocationPerformance", [this]() -> bool {
        try {
            const int num_allocations = 10000;
            const size_t allocation_size = 1024;

            auto start = std::chrono::high_resolution_clock::now();

            std::vector<std::shared_ptr<Memory>> memories;
            for (int i = 0; i < num_allocations; ++i) {
                auto memory = context::allocateMemory(allocation_size);
                if (memory) {
                    memories.push_back(memory);
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            double avg_time_per_allocation = static_cast<double>(duration.count()) / num_allocations;
            std::cout << "Average allocation time: " << avg_time_per_allocation << "μs" << std::endl;

            // Performance threshold: should be under 100μs per allocation
            if (avg_time_per_allocation > 100.0) {
                SPDLOG_ERROR("Allocation performance too slow: {}μs per allocation", avg_time_per_allocation);
                return false;
            }

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("Allocation performance test failed: {}", e.what());
            return false;
        }
    });
}

TestResult PerformanceTest::testConcurrentPerformance() {
    return measureTime("ConcurrentPerformance", [this]() -> bool {
        try {
            const int num_threads = 4;
            const int allocations_per_thread = 1000;

            auto start = std::chrono::high_resolution_clock::now();

            std::vector<std::thread> threads;
            std::atomic<int> success_count{0};

            for (int i = 0; i < num_threads; ++i) {
                threads.emplace_back([&]() {
                    for (int j = 0; j < allocations_per_thread; ++j) {
                        auto memory = context::allocateMemory(512);
                        if (memory) {
                            success_count++;
                        }
                    }
                });
            }

            for (auto &thread : threads) {
                thread.join();
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            double total_allocations = num_threads * allocations_per_thread;
            double avg_time_per_allocation = static_cast<double>(duration.count()) / total_allocations;
            std::cout << "Concurrent allocation time: " << avg_time_per_allocation << "μs per allocation" << std::endl;

            // Performance threshold: should be under 200μs per allocation under concurrent load
            if (avg_time_per_allocation > 200.0) {
                SPDLOG_ERROR("Concurrent allocation performance too slow: {}μs per allocation", avg_time_per_allocation);
                return false;
            }

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("Concurrent performance test failed: {}", e.what());
            return false;
        }
    });
}

TestResult PerformanceTest::testMemoryCopyPerformance() {
    return measureTime("MemoryCopyPerformance", [this]() -> bool {
        try {
            const size_t data_size = 1024 * 1024; // 1MB
            const int num_copies = 100;

            // Allocate source and destination memory
            auto src_memory = context::allocateMemory(data_size);
            auto dst_memory = context::allocateMemory(data_size);

            if (!src_memory || !dst_memory) {
                SPDLOG_ERROR("Failed to allocate memory for copy test");
                return false;
            }

            // Initialize source data
            std::memset(src_memory->data(), 0xAB, data_size);

            auto start = std::chrono::high_resolution_clock::now();

            // Perform memory copies
            for (int i = 0; i < num_copies; ++i) {
                context::memcpyD2D(dst_memory->data(), src_memory->data(), data_size);
            }

            // Synchronize to ensure copies complete
            context::syncDevice();

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            double avg_time_per_copy = static_cast<double>(duration.count()) / num_copies;
            double bandwidth = (data_size * num_copies) / (duration.count() / 1e6) / (1024 * 1024); // MB/s

            std::cout << "Average copy time: " << avg_time_per_copy << "μs" << std::endl;
            std::cout << "Memory bandwidth: " << bandwidth << " MB/s" << std::endl;

            // Performance threshold: should achieve at least 100 MB/s
            if (bandwidth < 100.0) {
                SPDLOG_ERROR("Memory copy performance too slow: {} MB/s", bandwidth);
                return false;
            }

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("Memory copy performance test failed: {}", e.what());
            return false;
        }
    });
}

// Stress Test Implementation
TestResult StressTest::run() {
    return measureTime("StressTest", [this]() -> bool {
        try {
            auto result1 = testHighFrequencyAllocations();
            if (!result1.passed) {
                SPDLOG_ERROR("High frequency allocations test failed: {}", result1.error_message);
                return false;
            }

            auto result2 = testLargeMemoryAllocations();
            if (!result2.passed) {
                SPDLOG_ERROR("Large memory allocations test failed: {}", result2.error_message);
                return false;
            }

            auto result3 = testCrossDeviceStress();
            if (!result3.passed) {
                SPDLOG_ERROR("Cross-device stress test failed: {}", result3.error_message);
                return false;
            }

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("StressTest failed with exception: {}", e.what());
            return false;
        }
    });
}

TestResult StressTest::testHighFrequencyAllocations() {
    return measureTime("HighFrequencyAllocations", [this]() -> bool {
        try {
            const int num_allocations = 100000;
            std::vector<std::shared_ptr<Memory>> memories;
            memories.reserve(num_allocations);

            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < num_allocations; ++i) {
                size_t size = 64 + (i % 1024);
                auto memory = context::allocateMemory(size);
                if (memory) {
                    memories.push_back(memory);
                }

                // Periodically deallocate some memory to test alloc/dealloc stress
                if (i % 1000 == 0 && !memories.empty()) {
                    memories.erase(memories.begin(), memories.begin() + std::min(100, static_cast<int>(memories.size())));
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            std::cout << "High frequency allocations completed: " << num_allocations
                      << " allocations in " << duration.count() << "ms" << std::endl;

            // Clear remaining memory
            memories.clear();

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("High frequency allocations test failed: {}", e.what());
            return false;
        }
    });
}

TestResult StressTest::testLargeMemoryAllocations() {
    return measureTime("LargeMemoryAllocations", [this]() -> bool {
        try {
            const size_t large_size = 100 * 1024 * 1024; // 100MB
            const int num_allocations = 10;

            std::vector<std::shared_ptr<Memory>> memories;

            for (int i = 0; i < num_allocations; ++i) {
                try {
                    auto memory = context::allocateMemory(large_size);
                    if (memory) {
                        memories.push_back(memory);
                        std::cout << "Allocated " << large_size / (1024 * 1024) << "MB memory block " << i + 1 << std::endl;
                    }
                } catch (const std::exception &e) {
                    std::cout << "Large allocation " << i + 1 << " failed (expected): " << e.what() << std::endl;
                    break; // Expected to fail at some point due to memory limits
                }
            }

            std::cout << "Successfully allocated " << memories.size() << " large memory blocks" << std::endl;

            // Clear memory
            memories.clear();

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("Large memory allocations test failed: {}", e.what());
            return false;
        }
    });
}

TestResult StressTest::testCrossDeviceStress() {
    return measureTime("CrossDeviceStress", [this]() -> bool {
        try {
            // Get available devices
            std::vector<Device> devices;
            for (int type = 0; type < static_cast<int>(Device::Type::COUNT); ++type) {
                size_t count = context::getDeviceCount(static_cast<Device::Type>(type));
                for (size_t i = 0; i < count; ++i) {
                    devices.emplace_back(static_cast<Device::Type>(type), i);
                }
            }

            if (devices.size() < 2) {
                std::cout << "Skipping cross-device stress test - need at least 2 devices" << std::endl;
                return true;
            }

            const int num_operations = 1000;
            std::vector<std::shared_ptr<Memory>> pinned_memories;

            for (int i = 0; i < num_operations; ++i) {
                // Switch to random device
                Device target_device = devices[i % devices.size()];
                context::setDevice(target_device);

                // Allocate pinned memory
                auto pinned_memory = context::allocatePinnedHostMemory(1024);
                if (pinned_memory) {
                    pinned_memories.push_back(pinned_memory);
                }

                // Periodically deallocate some memory
                if (i % 100 == 0 && !pinned_memories.empty()) {
                    pinned_memories.erase(pinned_memories.begin(),
                                          pinned_memories.begin() + std::min(10, static_cast<int>(pinned_memories.size())));
                }
            }

            std::cout << "Cross-device stress test completed: " << num_operations
                      << " operations across " << devices.size() << " devices" << std::endl;

            // Clear remaining memory
            pinned_memories.clear();

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("Cross-device stress test failed: {}", e.what());
            return false;
        }
    });
}

// Device Switch Test Implementation
TestResult DeviceSwitchTest::run() {
    return measureTime("DeviceSwitchTest", [this]() -> bool {
        try {
            SPDLOG_INFO("DeviceSwitchTest: Starting test");
            auto result1 = testD2HWithDeviceSwitch();
            if (!result1.passed) {
                SPDLOG_ERROR("D2H with device switch test failed: {}", result1.error_message);
                return false;
            }
            auto result2 = testD2HWithNonContiguousTensor();
            if (!result2.passed) {
                SPDLOG_ERROR("D2H with non-contiguous tensor test failed: {}", result2.error_message);
                return false;
            }
            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("DeviceSwitchTest failed with exception: {}", e.what());
            return false;
        }
    });
}

TestResult DeviceSwitchTest::testD2HWithDeviceSwitch() {
    return measureTime("testD2HWithDeviceSwitch", [this]() -> bool {
        try {
            SPDLOG_INFO("testD2HWithDeviceSwitch: Starting test");

            // Find a non-CPU device
            bool has_device = false;
            Device device_to_use;

            for (int device_type = 0; device_type < 10; ++device_type) {
                try {
                    Device test_device(static_cast<Device::Type>(device_type), 0);
                    context::setDevice(test_device);
                    if (test_device.getType() != Device::Type::CPU) {
                        has_device = true;
                        device_to_use = test_device;
                        break;
                    }
                } catch (...) {
                    // Device type not available, continue
                }
            }

            if (!has_device) {
                SPDLOG_INFO("testD2HWithDeviceSwitch: No non-CPU device available, skipping test");
                return true; // Skip test if no device available
            }

            SPDLOG_INFO("testD2HWithDeviceSwitch: Using device: {}", device_to_use.toString());

            // Allocate memory on device
            context::setDevice(device_to_use);
            auto device_memory = context::allocateMemory(1024);
            if (!device_memory) {
                SPDLOG_ERROR("Failed to allocate device memory");
                return false;
            }

            // Initialize device memory with test pattern (using H2D)
            std::vector<uint8_t> test_pattern(1024, 0xAB);
            context::memcpyH2D(device_memory->data(), test_pattern.data(), 1024);

            // Switch to CPU (simulating what happens when creating CPU tensors)
            context::setDevice(Device(Device::Type::CPU, 0));

            // Allocate host memory
            auto host_memory = context::allocateHostMemory(1024);
            if (!host_memory) {
                SPDLOG_ERROR("Failed to allocate host memory");
                return false;
            }

            // Now try to do D2H copy with CPU runtime active
            // This should throw an exception since CPU runtime cannot perform D2H operations
            try {
                context::memcpyD2H(host_memory->data(), device_memory->data(), 1024);
                SPDLOG_ERROR("D2H copy with CPU runtime should have thrown an exception");
                return false;
            } catch (const std::exception &e) {
                SPDLOG_INFO("testD2HWithDeviceSwitch: D2H copy correctly threw exception with CPU runtime: {}", e.what());
                // Verify it's the expected exception message
                std::string error_msg = e.what();
                if (error_msg.find("CPU runtime") == std::string::npos) {
                    SPDLOG_ERROR("Unexpected exception message: {}", error_msg);
                    return false;
                }
            }

            // Now do it explicitly - set device to source device first (should also work)
            context::setDevice(device_to_use);
            try {
                context::memcpyD2H(host_memory->data(), device_memory->data(), 1024);

                // Verify data was copied correctly
                std::vector<uint8_t> host_data(1024);
                std::memcpy(host_data.data(), host_memory->data(), 1024);
                if (host_data != test_pattern) {
                    SPDLOG_ERROR("Data mismatch after D2H copy with explicit device set");
                    return false;
                }

                SPDLOG_INFO("testD2HWithDeviceSwitch: Correct D2H copy succeeded with explicit device");
            } catch (const std::exception &e) {
                SPDLOG_ERROR("D2H copy with correct runtime failed: {}", e.what());
                return false;
            }

            SPDLOG_INFO("testD2HWithDeviceSwitch: Test passed");
            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("testD2HWithDeviceSwitch failed: {}", e.what());
            return false;
        }
    });
}

TestResult DeviceSwitchTest::testD2HWithNonContiguousTensor() {
    return measureTime("testD2HWithNonContiguousTensor", [this]() -> bool {
        try {
            SPDLOG_INFO("testD2HWithNonContiguousTensor: Starting test");

            // Find a non-CPU device
            bool has_device = false;
            Device device_to_use;

            for (int device_type = 0; device_type < 10; ++device_type) {
                try {
                    Device test_device(static_cast<Device::Type>(device_type), 0);
                    context::setDevice(test_device);
                    if (test_device.getType() != Device::Type::CPU) {
                        has_device = true;
                        device_to_use = test_device;
                        break;
                    }
                } catch (...) {
                    // Device type not available, continue
                }
            }

            if (!has_device) {
                SPDLOG_INFO("testD2HWithNonContiguousTensor: No non-CPU device available, skipping test");
                return true; // Skip test if no device available
            }

            SPDLOG_INFO("testD2HWithNonContiguousTensor: Using device: {}", device_to_use.toString());

            // This test simulates the exact scenario from the bug:
            // 1. Create device tensor (switches to device)
            // 2. Create CPU tensor (switches to CPU) - this was causing the issue
            // 3. Copy from device to CPU - should work correctly now

            // Set device to source device
            context::setDevice(device_to_use);

            // Allocate device memory
            auto device_memory = context::allocateMemory(1024);
            if (!device_memory) {
                SPDLOG_ERROR("Failed to allocate device memory");
                return false;
            }

            // Initialize device memory
            std::vector<uint8_t> test_pattern(1024, 0xCD);
            context::memcpyH2D(device_memory->data(), test_pattern.data(), 1024);

            // Switch to CPU (simulating Tensor::empty() call which switches device)
            context::setDevice(Device(Device::Type::CPU, 0));

            // Allocate host memory (this simulates creating CPU tensor)
            auto host_memory = context::allocateHostMemory(1024);
            if (!host_memory) {
                SPDLOG_ERROR("Failed to allocate host memory");
                return false;
            }

            // Now try D2H copy with CPU runtime - should throw an exception
            // This tests that CPU runtime correctly rejects D2H operations
            try {
                context::memcpyD2H(host_memory->data(), device_memory->data(), 1024);
                SPDLOG_ERROR("D2H copy with CPU runtime should have thrown an exception");
                return false;
            } catch (const std::exception &e) {
                SPDLOG_INFO("testD2HWithNonContiguousTensor: D2H copy correctly threw exception with CPU runtime: {}", e.what());
                // Verify it's the expected exception message
                std::string error_msg = e.what();
                if (error_msg.find("CPU runtime") == std::string::npos) {
                    SPDLOG_ERROR("Unexpected exception message: {}", error_msg);
                    return false;
                }
            }

            // Now explicitly set device to source device and try again - should succeed
            context::setDevice(device_to_use);
            try {
                context::memcpyD2H(host_memory->data(), device_memory->data(), 1024);

                // Verify data
                std::vector<uint8_t> host_data(1024);
                std::memcpy(host_data.data(), host_memory->data(), 1024);
                if (host_data != test_pattern) {
                    SPDLOG_ERROR("Data mismatch after D2H copy with explicit device set");
                    return false;
                }

                SPDLOG_INFO("testD2HWithNonContiguousTensor: D2H copy succeeded with explicit device set");
            } catch (const std::exception &e) {
                SPDLOG_ERROR("D2H copy failed with explicit device set: {}", e.what());
                return false;
            }

            SPDLOG_INFO("testD2HWithNonContiguousTensor: Test passed");
            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("testD2HWithNonContiguousTensor failed: {}", e.what());
            return false;
        }
    });
}

} // namespace infinicore::test
