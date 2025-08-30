#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::sigmoid_backward {
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
    test->_attributes = new Attributes();
    if (tensors.find("input") == tensors.end()
        || tensors.find("grad_output") == tensors.end()
        || tensors.find("grad_input") == tensors.end()
        || tensors.find("ans") == tensors.end()) {
        throw std::runtime_error("Invalid Test");
    }

    test->_attributes->input = tensors["input"];
    test->_attributes->grad_output = tensors["grad_output"];
    test->_attributes->grad_input = tensors["grad_input"];
    test->_attributes->ans = tensors["ans"];

    auto elemType = test->_attributes->input->ggml_type();
    if (elemType == GGML_TYPE_BF16) {
        test->_rtol = 1e-2;
        test->_atol = 1e-2;
    }
    if (elemType == GGML_TYPE_F16) {
        test->_rtol = 1e-3;
        test->_atol = 1e-3;
    }
    if (elemType == GGML_TYPE_F32) {
        test->_rtol = 1e-6;
        test->_atol = 1e-6;
    }

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    infiniopSigmoidBackwardDescriptor_t op_desc;
    auto input = _attributes->input->to(device, device_id);
    auto grad_output = _attributes->grad_output->to(device, device_id);
    auto grad_input = _attributes->grad_input->to(device, device_id);
    CHECK_OR(infiniopCreateSigmoidBackwardDescriptor(handle, &op_desc,
                                                     grad_input->desc(),
                                                     input->desc(),
                                                     grad_output->desc()),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create op descriptor."));
    size_t workspace_size;
    CHECK_OR(infiniopGetSigmoidBackwardWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));
    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    CHECK_OR(infiniopSigmoidBackward(op_desc, workspace, workspace_size,
                                     grad_input->data(),
                                     input->data(),
                                     grad_output->data(),
                                     nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    try {
        allClose(grad_input, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopSigmoidBackward(
                op_desc, workspace, workspace_size,
                grad_input->data(),
                input->data(),
                grad_output->data(),
                nullptr);
        },
        warm_ups, iterations);

    return TEST_PASSED(elapsed_time);
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
    oss << op_name() << std::endl;
    oss << "- input: " << _attributes->input->info() << std::endl;
    oss << "- grad_output: " << _attributes->grad_output->info() << std::endl;
    oss << "- grad_input: " << _attributes->grad_input->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}
} // namespace infiniop_test::sigmoid_backward
