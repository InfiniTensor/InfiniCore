#ifndef __INFINICORE_TEST_SAFETENSORS_LOADING_H__
#define __INFINICORE_TEST_SAFETENSORS_LOADING_H__

#include "../test_runner.h"
#include "infinicore/tensor.hpp"
#include <string>

// Forward declaration to avoid pybind11 dependency in test
namespace infinicore::safetensors {
    infinicore::Tensor load_tensor(const std::string& file_path, const std::string& tensor_name);
}

namespace infinicore::test {

class SafetensorsLoaderTest : public TestFramework {
public:
    TestResult run() override;
    std::string getName() const override { return "SafetensorsLoaderTest"; }

private:
    TestResult testSafetensorsLoadTensorInterface();
};

} // namespace infinicore::test

#endif // __INFINICORE_TEST_SAFETENSORS_LOADING_H__
