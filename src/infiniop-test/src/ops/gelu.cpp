#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::gelu {
struct Test::Attributes {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> output;
    std::shared_ptr<Tensor> ans;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes{
        .input = tensors.at("input"),
        .output = tensors.at("output"),
        .ans = tensors.at("ans")
    };
    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    infiniopGeluDescriptor_t geluDesc;
    auto input_device = _attributes->input->to(device, device_id);
    auto output_device = _attributes->output->to(device, device_id);
    
    CHECK_OR(infiniopCreateGeluDescriptor(handle, &geluDesc,
                                         output_device->desc(),
                                         input_device->desc()), return TEST_FAILED(OP_CREATION_FAILED, "Failed to create gelu descriptor."));

    size_t workspaceSize;
    CHECK_OR(infiniopGetGeluWorkspaceSize(geluDesc, &workspaceSize),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get gelu workspace size."));

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspaceSize),
                 return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    }

    CHECK_OR(infiniopGelu(geluDesc, workspace, workspaceSize,
                          output_device->data(),
                          input_device->data(),
                          nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed to execute gelu."));

    auto output_host = output_device->to(INFINI_DEVICE_CPU);

    try {
        allClose(output_host, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double time = 0.0;
    if (iterations > 0) {
        time = benchmark(
            [&]() {
                CHECK_OR(infiniopGelu(geluDesc, workspace, workspaceSize,
                                      output_device->data(),
                                      input_device->data(),
                                      nullptr),
                         throw std::runtime_error("Failed to execute gelu"));
            },
            warm_ups, iterations);
    }

    CHECK_OR(infiniopDestroyGeluDescriptor(geluDesc), return TEST_FAILED(OP_CREATION_FAILED, "Failed to destroy gelu descriptor."));
    if (workspace) {
        CHECK_OR(infinirtFree(workspace), return TEST_FAILED(OP_CREATION_FAILED, "Failed to free workspace."));
    }

    return TEST_PASSED(time);
}

std::vector<std::string> Test::attribute_names() {
    return {};
}

std::vector<std::string> Test::tensor_names() {
    return {"input", "output", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"output"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << "gelu(";
    oss << "input=" << _attributes->input->info();
    oss << ", output=" << _attributes->output->info();
    oss << ", rtol=" << _rtol << ", atol=" << _atol;
    oss << ")";
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::gelu