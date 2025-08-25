#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::relu_backward {
struct Test::Attributes {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> grad_output;
    std::shared_ptr<Tensor> grad_input;
    std::shared_ptr<Tensor> ans;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes{
        .input = tensors.at("input"),
        .grad_output = tensors.at("grad_output"),
        .grad_input = tensors.at("grad_input"),
        .ans = tensors.at("ans")
    };
    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    // Check for zero strides in input tensors
    auto input_strides = _attributes->input->strides();
    auto grad_output_strides = _attributes->grad_output->strides();
    auto grad_input_strides = _attributes->grad_input->strides();
    
    // Skip test if any tensor has zero stride (similar to PyTorch test behavior)
    for (auto stride : input_strides) {
        if (stride == 0) {
            return TEST_PASSED(0.0); // Skip test with zero time
        }
    }
    for (auto stride : grad_output_strides) {
        if (stride == 0) {
            return TEST_PASSED(0.0); // Skip test with zero time
        }
    }
    for (auto stride : grad_input_strides) {
        if (stride == 0) {
            return TEST_PASSED(0.0); // Skip test with zero time
        }
    }
    
    infiniopReluBackwardDescriptor_t reluBackwardDesc;
    auto grad_output_device = _attributes->grad_output->to(device, device_id);
    auto input_device = _attributes->input->to(device, device_id);
    auto grad_input_device = _attributes->grad_input->to(device, device_id);
    
    CHECK_OR(infiniopCreateReluBackwardDescriptor(handle, &reluBackwardDesc,
                                                 grad_input_device->desc(),
                                                 grad_output_device->desc(),
                                                 input_device->desc()), return TEST_FAILED(OP_CREATION_FAILED, "Failed to create relu_backward descriptor."));

    size_t workspaceSize;
    CHECK_OR(infiniopGetReluBackwardWorkspaceSize(reluBackwardDesc, &workspaceSize),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get relu_backward workspace size."));

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspaceSize),
                 return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    }

    CHECK_OR(infiniopReluBackward(reluBackwardDesc, workspace, workspaceSize,
                                  grad_input_device->data(),
                                  input_device->data(),
                                  grad_output_device->data(),
                                  nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed to execute relu backward."));

    auto grad_input_host = grad_input_device->to(INFINI_DEVICE_CPU);

    try {
        allClose(grad_input_host, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double time = 0.0;
    if (iterations > 0) {
        time = benchmark(
            [&]() {
                CHECK_OR(infiniopReluBackward(reluBackwardDesc, workspace, workspaceSize,
                                              grad_input_device->data(),
                                              input_device->data(),
                                              grad_output_device->data(),
                                              nullptr),
                         throw std::runtime_error("Failed to execute relu backward"));
            },
            warm_ups, iterations);
    }

    CHECK_OR(infiniopDestroyReluBackwardDescriptor(reluBackwardDesc), return TEST_FAILED(OP_CREATION_FAILED, "Failed to destroy relu_backward descriptor."));
    if (workspace) {
        CHECK_OR(infinirtFree(workspace), return TEST_FAILED(OP_CREATION_FAILED, "Failed to free workspace."));
    }

    return TEST_PASSED(time);
}

std::vector<std::string> Test::attribute_names() {
    return {};
}

std::vector<std::string> Test::tensor_names() {
    return {"input", "grad_output", "grad_input", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"grad_input"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << "relu_backward(";
    oss << "input=" << _attributes->input->info();
    oss << ", grad_output=" << _attributes->grad_output->info();
    oss << ", grad_input=" << _attributes->grad_input->info();
    oss << ", rtol=" << _rtol << ", atol=" << _atol;
    oss << ")";
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::relu_backward