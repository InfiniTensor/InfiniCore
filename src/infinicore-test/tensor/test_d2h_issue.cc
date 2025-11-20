#include "test_d2h_issue.h"
#include "infinicore/tensor.hpp"
#include <spdlog/spdlog.h>

namespace infinicore::test {

TestResult D2HIssueTest::run() {
    return measureTime("D2HIssueTest", [this]() -> bool {
        try {
            SPDLOG_INFO("D2HIssueTest: Starting test - reproducing D2H segfault issue from Python .to() call");

            // Simulate the exact scenario from InfiniLM test:
            // 1. Create a tensor on GPU device (NVIDIA)
            // 2. Call .to(CPU) on the GPU tensor (simulating Python tensor.to(cpu_device))
            // 3. This internally calls TensorImpl::to() which creates a CPU tensor and calls copy_from
            // 4. The segfault occurs during the D2H copy if memcpyD2H uses CPU runtime

            // Find a non-CPU device (prefer NVIDIA to match the actual scenario)
            Device device_to_use;
            bool has_device = false;

            // First try NVIDIA (most common case)
            for (int device_type = 1; device_type < 10; ++device_type) {
                try {
                    Device test_device(static_cast<Device::Type>(device_type), 0);
                    if (test_device.getType() != Device::Type::CPU) {
                        context::setDevice(test_device);
                        has_device = true;
                        device_to_use = test_device;
                        break;
                    }
                } catch (...) {
                    // Device type not available, continue
                }
            }

            if (!has_device) {
                SPDLOG_INFO("D2HIssueTest: No non-CPU device available, skipping test");
                return true;
            }

            SPDLOG_INFO("D2HIssueTest: Using device: {}", device_to_use.toString());

            // Step 1: Create tensor on GPU device (simulating a GPU tensor from model)
            context::setDevice(device_to_use);
            Tensor gpu_tensor = Tensor::zeros({10, 20}, DataType::F32, device_to_use);
            SPDLOG_INFO("D2HIssueTest: Created GPU tensor on {}", device_to_use.toString());

            // Verify the tensor is on GPU
            if (gpu_tensor->device() != device_to_use) {
                SPDLOG_ERROR("D2HIssueTest: GPU tensor device mismatch");
                return false;
            }

            // Step 2: Simulate Python .to(CPU) call
            // This is the exact scenario that causes the segfault:
            // - GPU tensor exists
            // - Python calls tensor.to(cpu_device)
            // - This calls TensorImpl::to(CPU) which:
            //   a) Creates a CPU tensor (switches context to CPU)
            //   b) Calls copy_from to copy from GPU to CPU (D2H)
            //   c) The segfault occurs if memcpyD2H uses CPU runtime
            Device cpu_device(Device::Type::CPU, 0);
            SPDLOG_INFO("D2HIssueTest: Current context device before .to() call: {}", context::getDevice().toString());
            SPDLOG_INFO("D2HIssueTest: Calling gpu_tensor->to(CPU) - this simulates Python tensor.to(cpu_device)");

            // This is the critical call that reproduces the issue
            Tensor cpu_result = gpu_tensor->to(cpu_device);
            SPDLOG_INFO("D2HIssueTest: gpu_tensor->to(CPU) completed successfully");

            // Verify the result is on CPU
            if (cpu_result->device() != cpu_device) {
                SPDLOG_ERROR("D2HIssueTest: Result tensor device mismatch - expected CPU, got {}",
                             cpu_result->device().toString());
                return false;
            }

            SPDLOG_INFO("D2HIssueTest: Result tensor is on CPU as expected");
            SPDLOG_INFO("D2HIssueTest: Current context device after .to() call: {}", context::getDevice().toString());

            SPDLOG_INFO("D2HIssueTest: Test passed");
            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("D2HIssueTest failed with exception: {}", e.what());
            return false;
        }
    });
}

} // namespace infinicore::test
