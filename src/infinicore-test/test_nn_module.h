#ifndef __INFINICORE_TEST_NN_MODULE_H__
#define __INFINICORE_TEST_NN_MODULE_H__

#include "infinicore/device.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/parameter.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "test_runner.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <sys/stat.h>
#include <vector>

namespace infinicore::test {

// Simple test module that mimics torch.nn.Linear
class MockLinearModule : public infinicore::nn::Module {
public:
    // Declare parameters using macros (torch-like style)
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(bias);

    MockLinearModule(int input_size, int output_size, const infinicore::Device &device,
                     Size tp_dim = 0, Size tp_rank = 0, Size tp_size = 1)
        : input_size_(input_size), output_size_(output_size), device_(device),
          tp_dim_(tp_dim), tp_rank_(tp_rank), tp_size_(tp_size) {
        // Initialize parameters using macros
        INFINICORE_NN_PARAMETER_INIT(weight,
                                     ({static_cast<size_t>(output_size), static_cast<size_t>(input_size)},
                                      infinicore::DataType::F32,
                                      device,
                                      tp_dim_,
                                      tp_rank_,
                                      tp_size_));
        INFINICORE_NN_PARAMETER_INIT(bias,
                                     ({static_cast<size_t>(output_size)},
                                      infinicore::DataType::F32,
                                      device,
                                      0,
                                      tp_dim == 0 ? tp_rank_ : 0,
                                      tp_dim == 0 ? tp_size_ : 1));
    }

    // Simple forward pass (conceptual - would need actual matrix operations)
    infinicore::Tensor forward(const infinicore::Tensor &input) {
        // This is a placeholder - in a real implementation, you'd do matrix multiplication
        // For testing purposes, we'll just return the input
        return input;
    }

    infinicore::Tensor get_weight() const {
        auto state_dict = this->state_dict();
        auto it = state_dict.find("weight");
        if (it != state_dict.end()) {
            return it->second;
        }
        throw std::runtime_error("Weight parameter not found");
    }

    infinicore::Tensor get_bias() const {
        auto state_dict = this->state_dict();
        auto it = state_dict.find("bias");
        if (it != state_dict.end()) {
            return it->second;
        }
        throw std::runtime_error("Bias parameter not found");
    }

private:
    int input_size_;
    int output_size_;
    infinicore::Device device_;

    Size tp_dim_;
    Size tp_rank_;
    Size tp_size_;
};

class NNModuleTest : public TestFramework {
public:
    TestResult run() override;
    std::string getName() const override { return "NNModuleTest"; }

private:
    TestResult testBasicModuleCreation();      // Merged: creation, parameters, state_dict, load_state_dict
    TestResult testTensorParallelParameters(); // Module with tensor parallel parameters
    TestResult testLoadStateDict();            // Advanced: hierarchical modules
    TestResult testModuleHierarchy();          // Demonstrates proper hierarchical construction pattern
    TestResult testParameterLoading();         // Test blob parameter loading
    TestResult testModuleLinear();             // Comprehensive Linear module test
    TestResult testModuleEmbedding();          // Embedding module test
    TestResult testModuleRMSNorm();            // RMSNorm module test
    TestResult testModuleRoPE();               // RoPE module test
    TestResult testDtypeAssertion();           // Test dtype assertions when loading parameters
    TestResult testTinyLlamaConstruction();    // Comprehensive: construction + weight loading + validation
};

} // namespace infinicore::test

#endif // __INFINICORE_TEST_NN_MODULE_H__
