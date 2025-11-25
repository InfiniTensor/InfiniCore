#ifndef __INFINICORE_TEST_TENSOR_USAGE_H__
#define __INFINICORE_TEST_TENSOR_USAGE_H__

#include "../memory_test.h"
#include "../test_runner.h"
#include "infinicore/context/context.hpp"
#include "infinicore/tensor.hpp"
#include <vector>

namespace infinicore::test {

class TensorUsageTest : public TestFramework {
public:
    TestResult run() override;
    std::string getName() const override { return "TensorUsageTest"; }

private:
    TestResult testTensorFromBlob();
    TestResult testTensorDeviceConversion();
    TestResult testTensorContiguous();
    TestResult testTensorCrossDeviceCopy();
    TestResult testTensorShapeAndDtype();
    TestResult testTensorWeightLoadingScenario();
};

} // namespace infinicore::test

#endif // __INFINICORE_TEST_TENSOR_USAGE_H__
