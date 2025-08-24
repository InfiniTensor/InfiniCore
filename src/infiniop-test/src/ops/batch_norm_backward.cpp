#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::batch_norm_backward {
struct Test::Attributes {
    std::shared_ptr<Tensor> grad_input;
    std::shared_ptr<Tensor> grad_weight;
    std::shared_ptr<Tensor> grad_bias;
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> grad_output;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> running_mean;
    std::shared_ptr<Tensor> running_var;
    std::shared_ptr<Tensor> ans_grad_input;
    std::shared_ptr<Tensor> ans_grad_weight;
    std::shared_ptr<Tensor> ans_grad_bias;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    if (tensors.find("grad_input") == tensors.end()
        || tensors.find("grad_weight") == tensors.end()
        || tensors.find("grad_bias") == tensors.end()
        || tensors.find("input") == tensors.end()
        || tensors.find("grad_output") == tensors.end()
        || tensors.find("weight") == tensors.end()
        || tensors.find("running_mean") == tensors.end()
        || tensors.find("running_var") == tensors.end()
        || tensors.find("ans_grad_input") == tensors.end()
        || tensors.find("ans_grad_weight") == tensors.end()
        || tensors.find("ans_grad_bias") == tensors.end()) {
        throw std::runtime_error("Invalid Test");
    }
    test->_attributes->grad_input = tensors["grad_input"];
    test->_attributes->grad_weight = tensors["grad_weight"];
    test->_attributes->grad_bias = tensors["grad_bias"];
    test->_attributes->input = tensors["input"];
    test->_attributes->grad_output = tensors["grad_output"];
    test->_attributes->weight = tensors["weight"];
    test->_attributes->running_mean = tensors["running_mean"];
    test->_attributes->running_var = tensors["running_var"];
    test->_attributes->ans_grad_input = tensors["ans_grad_input"];
    test->_attributes->ans_grad_weight = tensors["ans_grad_weight"];
    test->_attributes->ans_grad_bias = tensors["ans_grad_bias"];

    return test;
}
std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    infiniopBatchNormBackwardDescriptor_t op_desc;
    auto grad_input = _attributes->grad_input->to(device, device_id);
    auto grad_weight = _attributes->grad_weight->to(device, device_id);
    auto grad_bias = _attributes->grad_bias->to(device, device_id);
    auto input = _attributes->input->to(device, device_id);
    auto grad_output = _attributes->grad_output->to(device, device_id);
    auto weight = _attributes->weight->to(device, device_id);
    auto running_mean = _attributes->running_mean->to(device, device_id);
    auto running_var = _attributes->running_var->to(device, device_id);
    CHECK_OR(infiniopCreateBatchNormBackwardDescriptor(handle, &op_desc,
            grad_input->desc(),
            grad_weight->desc(),
            grad_bias->desc(),
            input->desc(),
            grad_output->desc(),
            weight->desc(),
            running_mean->desc(),
            running_var->desc()
        ),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create op descriptor."));
    size_t workspace_size;
    CHECK_OR(infiniopGetBatchNormBackwardWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));
    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    CHECK_OR(infiniopBatchNormBackward(op_desc, workspace, workspace_size,
                    grad_input->data(),
                    grad_weight->data(),
                    grad_bias->data(),
                    input->data(),
                    grad_output->data(),
                    weight->data(),
                    running_mean->data(),
                    running_var->data(),
                    nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    try {
        allClose(grad_input, _attributes->ans_grad_input, _rtol, _atol);
        allClose(grad_weight, _attributes->ans_grad_weight, _rtol, _atol);
        allClose(grad_bias, _attributes->ans_grad_bias, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopBatchNormBackward(
                op_desc, workspace, workspace_size,
                grad_input->data(),
                grad_weight->data(),
                grad_bias->data(),
                input->data(),
                grad_output->data(),
                weight->data(),
                running_mean->data(),
                running_var->data(),
                nullptr);
        },
        warm_ups, iterations);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {};
}

std::vector<std::string> Test::tensor_names() {
    return {"grad_input", "grad_weight", "grad_bias", "input", "grad_output", "weight", "running_mean", "running_var", "ans_grad_input", "ans_grad_weight", "ans_grad_bias"};
}

std::vector<std::string> Test::output_names() {
    return {"grad_input", "grad_weight", "grad_bias"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- grad_input: " << _attributes->grad_input->info() << std::endl;
    oss << "- grad_weight: " << _attributes->grad_weight->info() << std::endl;
    oss << "- grad_bias: " << _attributes->grad_bias->info() << std::endl;
    oss << "- input: " << _attributes->input->info() << std::endl;
    oss << "- grad_output: " << _attributes->grad_output->info() << std::endl;
    oss << "- weight: " << _attributes->weight->info() << std::endl;
    oss << "- running_mean: " << _attributes->running_mean->info() << std::endl;
    oss << "- running_var: " << _attributes->running_var->info() << std::endl;

    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::batch_norm_backward
