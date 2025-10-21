#include "test_tensor_destructor.h"

namespace infinicore::test {

// Test 1: Basic tensor creation and destruction
TestResult TensorDestructorTest::testBasicTensorDestruction() {
    return measureTime("BasicTensorDestruction", [this]() {
        {
            // Create a tensor in a scope to test automatic destruction
            auto tensor = Tensor::empty({2, 3}, DataType::F32, Device::Type::CPU);

            // Verify tensor was created successfully
            if (!tensor.operator->()) {
                return false;
            }
            if (tensor->shape().size() != 2) {
                return false;
            }
            if (tensor->shape()[0] != 2) {
                return false;
            }
            if (tensor->shape()[1] != 3) {
                return false;
            }
            if (tensor->dtype() != DataType::F32) {
                return false;
            }

            std::cout << "Tensor created successfully with shape: ";
            for (auto dim : tensor->shape()) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
        }

        // Tensor should be destroyed when it goes out of scope
        // This should trigger the TensorMetaData destructor
        std::cout << "Tensor destroyed successfully - destructor called" << std::endl;
        return true;
    });
}

// Test 2: Multiple tensor creation and destruction
TestResult TensorDestructorTest::testMultipleTensorDestruction() {
    return measureTime("MultipleTensorDestruction", [this]() {
        std::vector<Tensor> tensors;

        // Create multiple tensors with different shapes and types
        tensors.push_back(Tensor::empty({1, 2, 3}, DataType::F32, Device::Type::CPU));
        tensors.push_back(Tensor::empty({4, 5}, DataType::F64, Device::Type::CPU));
        tensors.push_back(Tensor::zeros({2, 2, 2}, DataType::I32, Device::Type::CPU));
        tensors.push_back(Tensor::ones({3, 4}, DataType::F16, Device::Type::CPU));

        // Verify all tensors were created
        if (tensors.size() != 4) {
            return false;
        }
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (!tensors[i].operator->()) {
                return false;
            }
            std::cout << "Tensor " << i << " created with shape: ";
            for (auto dim : tensors[i]->shape()) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "All " << tensors.size() << " tensors created successfully" << std::endl;

        // All tensors will be destroyed when the vector goes out of scope
        // This should trigger all TensorMetaData destructors
        return true;
    });
}

// Test 3: Different data types
TestResult TensorDestructorTest::testDifferentDataTypes() {
    return measureTime("DifferentDataTypes", [this]() {
        std::vector<std::pair<DataType, std::string>> data_types = {
            {DataType::F32, "F32"},
            {DataType::F64, "F64"},
            {DataType::F16, "F16"},
            {DataType::I32, "I32"},
            {DataType::I64, "I64"},
            {DataType::I8, "I8"},
            {DataType::U8, "U8"},
            {DataType::BOOL, "BOOL"}};

        for (const auto &[dtype, name] : data_types) {
            {
                auto tensor = Tensor::empty({2, 2}, dtype, Device::Type::CPU);
                if (!tensor.operator->()) {
                    return false;
                }
                if (tensor->dtype() != dtype) {
                    return false;
                }
                std::cout << "Created tensor with data type: " << name << std::endl;
            }
            std::cout << "Destroyed tensor with data type: " << name << std::endl;
        }

        return true;
    });
}

// Test 4: Different shapes
TestResult TensorDestructorTest::testDifferentShapes() {
    return measureTime("DifferentShapes", [this]() {
        std::vector<Shape> shapes = {
            {1},             // 1D
            {2, 3},          // 2D
            {4, 5, 6},       // 3D
            {1, 2, 3, 4},    // 4D
            {2, 3, 4, 5, 6}, // 5D
            {1000},          // Large 1D
            {100, 100},      // Large 2D
            {10, 10, 10, 10} // Large 4D
        };

        for (const auto &shape : shapes) {
            {
                auto tensor = Tensor::empty(shape, DataType::F32, Device::Type::CPU);
                if (!tensor.operator->()) {
                    return false;
                }
                if (tensor->shape() != shape) {
                    return false;
                }
                std::cout << "Created tensor with shape: ";
                for (auto dim : shape) {
                    std::cout << dim << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "Destroyed tensor with shape: ";
            for (auto dim : shape) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
        }

        return true;
    });
}

// Test 5: Tensor from blob
TestResult TensorDestructorTest::testTensorFromBlob() {
    return measureTime("TensorFromBlob", [this]() {
        // Create a blob of data
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

        {
            // Create tensor from blob
            auto tensor = Tensor::from_blob(data.data(), {2, 3}, DataType::F32, Device::Type::CPU);
            if (!tensor.operator->()) {
                return false;
            }
            if (tensor->shape() != Shape({2, 3})) {
                return false;
            }
            if (tensor->dtype() != DataType::F32) {
                return false;
            }

            std::cout << "Created tensor from blob with shape: ";
            for (auto dim : tensor->shape()) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Destroyed tensor from blob successfully" << std::endl;
        return true;
    });
}

// Test 6: Strided tensor
TestResult TensorDestructorTest::testStridedTensor() {
    return measureTime("StridedTensor", [this]() {
        {
            // Create a strided tensor
            auto tensor = Tensor::empty({4, 4}, DataType::F32, Device::Type::CPU);
            if (!tensor.operator->()) {
                return false;
            }

            // Create a narrowed view
            std::vector<TensorSliceParams> slices = {
                {0, 0, 2}, // dimension 0: start at 0, length 2
                {1, 0, 2}  // dimension 1: start at 0, length 2
            };
            auto strided_tensor = tensor->narrow(slices);
            if (!strided_tensor.operator->()) {
                return false;
            }

            std::cout << "Created strided tensor with shape: ";
            for (auto dim : strided_tensor->shape()) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Destroyed strided tensor successfully" << std::endl;
        return true;
    });
}

// Test 7: Memory leak detection
TestResult TensorDestructorTest::testMemoryLeakDetection() {
    return measureTime("MemoryLeakDetection", [this]() {
        // Reset memory leak detector
        MemoryLeakDetector::instance().reset();

        size_t initial_leaks = MemoryLeakDetector::instance().getLeakCount();

        // Create and destroy many tensors
        for (int i = 0; i < 100; ++i) {
            {
                auto tensor = Tensor::empty({10, 10}, DataType::F32, Device::Type::CPU);
                if (!tensor.operator->()) {
                    return false;
                }
            }
        }

        size_t final_leaks = MemoryLeakDetector::instance().getLeakCount();

        std::cout << "Initial leaks: " << initial_leaks << std::endl;
        std::cout << "Final leaks: " << final_leaks << std::endl;

        // Should not have more leaks than we started with
        return final_leaks <= initial_leaks;
    });
}

// Test 8: Tensor copy destruction
TestResult TensorDestructorTest::testTensorCopyDestruction() {
    return measureTime("TensorCopyDestruction", [this]() {
        {
            auto original_tensor = Tensor::empty({3, 3}, DataType::F32, Device::Type::CPU);
            if (!original_tensor.operator->()) {
                return false;
            }

            // Create a copy (using assignment operator)
            auto copied_tensor = original_tensor;
            if (!copied_tensor.operator->()) {
                return false;
            }

            std::cout << "Created original and copied tensors" << std::endl;
            std::cout << "Original tensor shape: ";
            for (auto dim : original_tensor->shape()) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
            std::cout << "Copied tensor shape: ";
            for (auto dim : copied_tensor->shape()) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Destroyed original and copied tensors successfully" << std::endl;
        return true;
    });
}

// Main test runner
TestResult TensorDestructorTest::run() {
    std::vector<TestResult> results;

    std::cout << "==============================================\n"
              << "Tensor Destructor Test Suite\n"
              << "==============================================" << std::endl;

    // Run all tests
    results.push_back(testBasicTensorDestruction());
    results.push_back(testMultipleTensorDestruction());
    results.push_back(testDifferentDataTypes());
    results.push_back(testDifferentShapes());
    results.push_back(testTensorFromBlob());
    results.push_back(testStridedTensor());
    results.push_back(testMemoryLeakDetection());
    results.push_back(testTensorCopyDestruction());

    // Check if all tests passed
    bool all_passed = true;
    for (const auto &result : results) {
        if (!result.passed) {
            all_passed = false;
            break;
        }
    }

    return TestResult("TensorDestructorTest", all_passed,
                      all_passed ? "" : "Some tensor destructor tests failed");
}

} // namespace infinicore::test
