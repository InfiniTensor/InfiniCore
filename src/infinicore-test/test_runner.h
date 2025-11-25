#ifndef __INFINICORE_TEST_RUNNER_H__
#define __INFINICORE_TEST_RUNNER_H__

#include <chrono>
#include <cmath>
#include <exception>
#include <infinicore.hpp>
#include <iostream>
#include <memory>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>
#include <vector>

namespace infinicore::test {

// ============================================================================
// Common Test Utilities
// ============================================================================

/**
 * @brief Compare two InfiniCore tensors elementwise with tolerance
 *
 * Compares two tensors for approximate equality, useful for testing numerical
 * computations where exact equality is not expected due to floating-point arithmetic.
 *
 * @param actual The actual tensor result
 * @param expected The expected tensor result
 * @param rtol Relative tolerance (default: 1e-5)
 * @param atol Absolute tolerance (default: 1e-5)
 * @return true if tensors are approximately equal, false otherwise
 *
 * @note Currently only supports F32 dtype
 * @note Tensors are automatically moved to CPU for comparison
 * @note Reports up to 10 mismatches with detailed coordinates
 */
inline bool tensorsAllClose(const infinicore::Tensor &actual,
                            const infinicore::Tensor &expected,
                            double rtol = 1e-5,
                            double atol = 1e-5) {
    if (actual->shape() != expected->shape()) {
        SPDLOG_ERROR("Shape mismatch: actual vs expected");
        return false;
    }

    auto cpu = infinicore::Device(infinicore::Device::Type::CPU, 0);
    auto a_cpu = actual->to(cpu);
    a_cpu = a_cpu->contiguous();
    auto b_cpu = expected->to(cpu);
    b_cpu = b_cpu->contiguous();

    if (a_cpu->dtype() != b_cpu->dtype()) {
        SPDLOG_ERROR("DType mismatch");
        return false;
    }

    // Only support F32 in this test
    if (a_cpu->dtype() != infinicore::DataType::F32) {
        SPDLOG_ERROR("Unsupported dtype for comparison; only F32 supported in test");
        return false;
    }

    size_t n = a_cpu->numel();
    const auto &shape = a_cpu->shape();

    // Precompute strides for index -> coords mapping
    std::vector<size_t> stride(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }

    const float *ap = reinterpret_cast<const float *>(a_cpu->data());
    const float *bp = reinterpret_cast<const float *>(b_cpu->data());
    size_t max_diff_index = 0;
    float max_diff = 0.0f;
    size_t num_fail_reported = 0;

    for (size_t i = 0; i < n; ++i) {
        float av = ap[i];
        float bv = bp[i];
        float diff = std::fabs(av - bv);
        if (diff > static_cast<float>(atol + rtol * std::fabs(bv))) {
            if (diff > max_diff) {
                max_diff = diff;
                max_diff_index = i;
            }
            if (num_fail_reported < 10) {
                // Convert linear index to coordinates
                std::vector<size_t> coords(shape.size(), 0);
                size_t t = i;
                for (size_t d = 0; d < shape.size(); ++d) {
                    coords[d] = t / stride[d];
                    t -= coords[d] * stride[d];
                }
                std::stringstream ss;
                ss << "[";
                for (size_t d = 0; d < coords.size(); ++d) {
                    ss << coords[d] << (d + 1 < coords.size() ? "," : "]");
                }
                double tol = atol + rtol * std::fabs(bv);
                SPDLOG_ERROR("Mismatch at index {} coords {}: actual={} expected={} diff={} tol={}",
                             i, ss.str(), av, bv, diff, tol);
                num_fail_reported++;
            }
        }
    }

    if (num_fail_reported > 0) {
        // Report summary with max diff
        std::vector<size_t> coords(shape.size(), 0);
        size_t t = max_diff_index;
        for (size_t d = 0; d < shape.size(); ++d) {
            coords[d] = t / stride[d];
            t -= coords[d] * stride[d];
        }
        std::stringstream ss;
        ss << "[";
        for (size_t d = 0; d < coords.size(); ++d) {
            ss << coords[d] << (d + 1 < coords.size() ? "," : "]");
        }
        SPDLOG_ERROR("Max diff {} at linear index {} coords {}", max_diff, max_diff_index, ss.str());
        return false;
    }

    return true;
}

// ============================================================================
// Test Framework Classes
// ============================================================================

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
class TestFramework {
public:
    virtual ~TestFramework() = default;
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

// Test runner
class InfiniCoreTestRunner {
public:
    void addTest(std::unique_ptr<TestFramework> test) {
        tests_.push_back(std::move(test));
    }

    std::vector<TestResult> runAllTests() {
        std::vector<TestResult> results;

        std::cout << "==============================================\n"
                  << "InfiniCore Test Suite\n"
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
    std::vector<std::unique_ptr<TestFramework>> tests_;

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
        std::vector<TestResult> failed_tests;

        for (const auto &result : results) {
            if (result.passed) {
                passed++;
            } else {
                failed++;
                failed_tests.push_back(result);
            }
            total_time += result.duration;
        }

        // Print list of failed tests if any
        if (!failed_tests.empty()) {
            std::cout << "\n==============================================\n"
                      << "❌ FAILED TESTS\n"
                      << "==============================================" << std::endl;
            for (const auto &test : failed_tests) {
                std::cout << "  • " << test.test_name;
                if (!test.error_message.empty()) {
                    std::cout << "\n    Error: " << test.error_message;
                }
                std::cout << "\n    Duration: " << test.duration.count() << "μs" << std::endl;
            }
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

#endif // __INFINICORE_TEST_RUNNER_H__
