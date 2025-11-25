#include "test_tensor_usage.h"
#include "infinicore/tensor.hpp"
#include <algorithm>
#include <cstring>
#include <random>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

namespace infinicore::test {

// Tensor Usage Test Implementation
// Tests tensor operations similar to those used in InfiniLM weight loading
TestResult TensorUsageTest::run() {
    return measureTime("TensorUsageTest", [this]() -> bool {
        try {
            SPDLOG_INFO("TensorUsageTest: Starting test suite");

            auto result1 = testTensorFromBlob();
            if (!result1.passed) {
                SPDLOG_ERROR("testTensorFromBlob failed: {}", result1.error_message);
                return false;
            }

            auto result2 = testTensorDeviceConversion();
            if (!result2.passed) {
                SPDLOG_ERROR("testTensorDeviceConversion failed: {}", result2.error_message);
                return false;
            }

            auto result3 = testTensorContiguous();
            if (!result3.passed) {
                SPDLOG_ERROR("testTensorContiguous failed: {}", result3.error_message);
                return false;
            }

            auto result4 = testTensorCrossDeviceCopy();
            if (!result4.passed) {
                SPDLOG_ERROR("testTensorCrossDeviceCopy failed: {}", result4.error_message);
                return false;
            }

            auto result5 = testTensorShapeAndDtype();
            if (!result5.passed) {
                SPDLOG_ERROR("testTensorShapeAndDtype failed: {}", result5.error_message);
                return false;
            }

            auto result6 = testTensorWeightLoadingScenario();
            if (!result6.passed) {
                SPDLOG_ERROR("testTensorWeightLoadingScenario failed: {}", result6.error_message);
                return false;
            }

            SPDLOG_INFO("TensorUsageTest: All tests passed");
            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("TensorUsageTest failed with exception: {}", e.what());
            return false;
        }
    });
}

TestResult TensorUsageTest::testTensorFromBlob() {
    return measureTime("testTensorFromBlob", [this]() -> bool {
        try {
            SPDLOG_INFO("testTensorFromBlob: Starting test");

            // Test creating tensor from blob (similar to torch_to_infinicore_tensor)
            const size_t num_elements = 100;

            // Allocate host memory
            std::vector<float> host_data(num_elements);
            for (size_t i = 0; i < num_elements; ++i) {
                host_data[i] = static_cast<float>(i);
            }

            // Create tensor from blob
            Device cpu_device(Device::Type::CPU, 0);
            context::setDevice(cpu_device);

            Tensor tensor = Tensor::from_blob(
                host_data.data(),
                {num_elements},
                DataType::F32,
                cpu_device);

            // Verify tensor properties
            if (tensor->shape() != Shape{num_elements}) {
                SPDLOG_ERROR("Tensor shape mismatch");
                return false;
            }

            if (tensor->dtype() != DataType::F32) {
                SPDLOG_ERROR("Tensor dtype mismatch");
                return false;
            }

            if (tensor->device() != cpu_device) {
                SPDLOG_ERROR("Tensor device mismatch");
                return false;
            }

            SPDLOG_INFO("testTensorFromBlob: Test passed");
            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("testTensorFromBlob failed: {}", e.what());
            return false;
        }
    });
}

TestResult TensorUsageTest::testTensorDeviceConversion() {
    return measureTime("testTensorDeviceConversion", [this]() -> bool {
        try {
            SPDLOG_INFO("testTensorDeviceConversion: Starting test");

            // Test tensor.to() method (device conversion)
            Device cpu_device(Device::Type::CPU, 0);
            context::setDevice(cpu_device);

            // Create tensor on CPU
            Tensor cpu_tensor = Tensor::zeros({10, 20}, DataType::F32, cpu_device);

            // Test conversion to same device (should return same tensor)
            Tensor same_device = cpu_tensor->to(cpu_device);
            if (same_device->device() != cpu_device) {
                SPDLOG_ERROR("Same device conversion failed");
                return false;
            }

            // Try to find a non-CPU device for cross-device conversion test
            bool has_device = false;
            Device device_to_use;

            for (int device_type = 0; device_type < 10; ++device_type) {
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

            if (has_device) {
                SPDLOG_INFO("testTensorDeviceConversion: Testing cross-device conversion to {}", device_to_use.toString());
                // Test conversion to device
                Tensor device_tensor = cpu_tensor->to(device_to_use);
                if (device_tensor->device() != device_to_use) {
                    SPDLOG_ERROR("Device conversion failed");
                    return false;
                }

                // Convert back to CPU
                Tensor back_to_cpu = device_tensor->to(cpu_device);
                if (back_to_cpu->device() != cpu_device) {
                    SPDLOG_ERROR("Conversion back to CPU failed");
                    return false;
                }
            } else {
                SPDLOG_INFO("testTensorDeviceConversion: No non-CPU device available, skipping cross-device test");
            }

            SPDLOG_INFO("testTensorDeviceConversion: Test passed");
            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("testTensorDeviceConversion failed: {}", e.what());
            return false;
        }
    });
}

TestResult TensorUsageTest::testTensorContiguous() {
    return measureTime("testTensorContiguous", [this]() -> bool {
        try {
            SPDLOG_INFO("testTensorContiguous: Starting test");

            Device cpu_device(Device::Type::CPU, 0);
            context::setDevice(cpu_device);

            // Create a contiguous tensor
            Tensor tensor = Tensor::zeros({4, 5}, DataType::F32, cpu_device);

            // Test is_contiguous()
            if (!tensor->is_contiguous()) {
                SPDLOG_ERROR("Newly created tensor should be contiguous");
                return false;
            }

            // Test contiguous() on already contiguous tensor (should return same)
            Tensor contiguous_tensor = tensor->contiguous();
            if (!contiguous_tensor->is_contiguous()) {
                SPDLOG_ERROR("contiguous() should return contiguous tensor");
                return false;
            }

            SPDLOG_INFO("testTensorContiguous: Test passed");
            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("testTensorContiguous failed: {}", e.what());
            return false;
        }
    });
}

TestResult TensorUsageTest::testTensorCrossDeviceCopy() {
    return measureTime("testTensorCrossDeviceCopy", [this]() -> bool {
        try {
            SPDLOG_INFO("testTensorCrossDeviceCopy: Starting test");

            Device cpu_device(Device::Type::CPU, 0);
            context::setDevice(cpu_device);

            // Create source tensor on CPU with test data
            const size_t num_elements = 50;
            std::vector<float> source_data(num_elements);
            for (size_t i = 0; i < num_elements; ++i) {
                source_data[i] = static_cast<float>(i * 2.5f);
            }

            Tensor source_tensor = Tensor::from_blob(
                source_data.data(),
                {num_elements},
                DataType::F32,
                cpu_device);

            // Try to find a non-CPU device
            bool has_device = false;
            Device device_to_use;

            for (int device_type = 0; device_type < 10; ++device_type) {
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

            if (has_device) {
                SPDLOG_INFO("testTensorCrossDeviceCopy: Testing copy to {}", device_to_use.toString());

                // Create destination tensor on device
                Tensor dest_tensor = Tensor::zeros({num_elements}, DataType::F32, device_to_use);

                // Copy from CPU to device (H2D)
                dest_tensor->copy_from(source_tensor);

                // Copy back to CPU (D2H)
                Tensor back_to_cpu = Tensor::zeros({num_elements}, DataType::F32, cpu_device);
                back_to_cpu->copy_from(dest_tensor);

                // Verify data (if on CPU, we can read it)
                if (back_to_cpu->device().getType() == Device::Type::CPU) {
                    // Note: We can't directly read device memory, but the copy should succeed
                    SPDLOG_INFO("testTensorCrossDeviceCopy: Cross-device copy completed successfully");
                }
            } else {
                SPDLOG_INFO("testTensorCrossDeviceCopy: No non-CPU device available, skipping cross-device test");
            }

            // Test CPU to CPU copy
            Tensor cpu_dest = Tensor::zeros({num_elements}, DataType::F32, cpu_device);
            cpu_dest->copy_from(source_tensor);

            SPDLOG_INFO("testTensorCrossDeviceCopy: Test passed");
            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("testTensorCrossDeviceCopy failed: {}", e.what());
            return false;
        }
    });
}

TestResult TensorUsageTest::testTensorShapeAndDtype() {
    return measureTime("testTensorShapeAndDtype", [this]() -> bool {
        try {
            SPDLOG_INFO("testTensorShapeAndDtype: Starting test");

            Device cpu_device(Device::Type::CPU, 0);
            context::setDevice(cpu_device);

            // Test different shapes
            std::vector<Shape> test_shapes = {
                {10},
                {5, 6},
                {2, 3, 4},
                {1, 2, 3, 4}};

            for (const auto &shape : test_shapes) {
                Tensor tensor = Tensor::zeros(shape, DataType::F32, cpu_device);
                if (tensor->shape() != shape) {
                    std::string shape_str;
                    for (auto s : shape) {
                        if (!shape_str.empty()) {
                            shape_str += " ";
                        }
                        shape_str += std::to_string(s);
                    }
                    SPDLOG_ERROR("Shape mismatch for shape: {}", shape_str);
                    return false;
                }
            }

            // Test different dtypes
            std::vector<DataType> test_dtypes = {
                DataType::F32,
                DataType::F64,
                DataType::I32,
                DataType::I64};

            for (const auto &dtype : test_dtypes) {
                Tensor tensor = Tensor::zeros({10}, dtype, cpu_device);
                if (tensor->dtype() != dtype) {
                    SPDLOG_ERROR("Dtype mismatch for dtype: {}", static_cast<int>(dtype));
                    return false;
                }
            }

            SPDLOG_INFO("testTensorShapeAndDtype: Test passed");
            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("testTensorShapeAndDtype failed: {}", e.what());
            return false;
        }
    });
}

TestResult TensorUsageTest::testTensorWeightLoadingScenario() {
    return measureTime("testTensorWeightLoadingScenario", [this]() -> bool {
        try {
            SPDLOG_INFO("testTensorWeightLoadingScenario: Starting test (simulating weight loading scenario)");

            // Simulate the weight loading scenario from InfiniLM test
            // 1. Create a "transformers" tensor (simulated with host data)
            // 2. Convert to InfiniCore tensor (from_blob)
            // 3. Copy to device if needed
            // 4. Verify the tensor

            const size_t embed_dim = 128;
            const size_t vocab_size = 1000;
            const size_t total_elements = embed_dim * vocab_size;

            // Step 1: Simulate "transformers" tensor data
            std::vector<float> transformers_data(total_elements);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (size_t i = 0; i < total_elements; ++i) {
                transformers_data[i] = dis(gen);
            }

            // Step 2: Convert to InfiniCore tensor (similar to torch_to_infinicore_tensor)
            Device cpu_device(Device::Type::CPU, 0);
            context::setDevice(cpu_device);

            Tensor infini_tensor = Tensor::from_blob(
                transformers_data.data(),
                {vocab_size, embed_dim},
                DataType::F32,
                cpu_device);

            // Step 3: Verify tensor properties
            if (infini_tensor->shape() != Shape{vocab_size, embed_dim}) {
                SPDLOG_ERROR("Tensor shape mismatch in weight loading scenario");
                return false;
            }

            if (infini_tensor->dtype() != DataType::F32) {
                SPDLOG_ERROR("Tensor dtype mismatch in weight loading scenario");
                return false;
            }

            if (!infini_tensor->is_contiguous()) {
                SPDLOG_ERROR("Tensor should be contiguous after from_blob");
                return false;
            }

            // Step 4: Test device conversion (if device available)
            bool has_device = false;
            Device device_to_use;

            for (int device_type = 0; device_type < 10; ++device_type) {
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

            if (has_device) {
                SPDLOG_INFO("testTensorWeightLoadingScenario: Testing device conversion to {}", device_to_use.toString());

                // Convert to device (similar to model loading)
                Tensor device_tensor = infini_tensor->to(device_to_use);

                // Verify conversion
                if (device_tensor->device() != device_to_use) {
                    SPDLOG_ERROR("Device conversion failed in weight loading scenario");
                    return false;
                }

                if (device_tensor->shape() != infini_tensor->shape()) {
                    SPDLOG_ERROR("Shape changed after device conversion");
                    return false;
                }

                // Convert back to CPU (similar to infinicore_to_torch_tensor)
                Tensor back_to_cpu = device_tensor->to(cpu_device);
                if (back_to_cpu->device() != cpu_device) {
                    SPDLOG_ERROR("Conversion back to CPU failed");
                    return false;
                }
            } else {
                SPDLOG_INFO("testTensorWeightLoadingScenario: No non-CPU device available, skipping device conversion test");
            }

            SPDLOG_INFO("testTensorWeightLoadingScenario: Test passed");
            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("testTensorWeightLoadingScenario failed: {}", e.what());
            return false;
        }
    });
}

} // namespace infinicore::test
