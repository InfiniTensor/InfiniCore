#ifndef __INFINICORE_TEST_TENSOR_DESTRUCTOR_H__
#define __INFINICORE_TEST_TENSOR_DESTRUCTOR_H__

#include "infinicore/context/context.hpp"
#include "infinicore/tensor.hpp"
#include "memory_test.h"
#include <iostream>
#include <memory>
#include <vector>

namespace infinicore::test {

class TensorDestructorTest : public MemoryTestFramework {
public:
    TestResult run() override;
    std::string getName() const override { return "TensorDestructorTest"; }

private:
    TestResult testBasicTensorDestruction();
    TestResult testMultipleTensorDestruction();
    TestResult testDifferentDataTypes();
    TestResult testDifferentShapes();
    TestResult testTensorFromBlob();
    TestResult testStridedTensor();
    TestResult testMemoryLeakDetection();
    TestResult testTensorCopyDestruction();
};

} // namespace infinicore::test

#endif // __INFINICORE_TEST_TENSOR_DESTRUCTOR_H__
