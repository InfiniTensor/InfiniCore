#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::layer_norm_backward {
struct Test::Attributes {
    bool bias_exist;
    std::shared_ptr<Tensor> grad_input;
    std::shared_ptr<Tensor> grad_weight;
    std::shared_ptr<Tensor> grad_bias;
    std::shared_ptr<Tensor> grad_output;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> input_standardization;
    std::shared_ptr<Tensor> input_std_deviation;
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
        || tensors.find("grad_output") == tensors.end()
        || tensors.find("weight") == tensors.end()
        || tensors.find("input_standardization") == tensors.end()
        || tensors.find("input_std_deviation") == tensors.end()
        || tensors.find("ans_grad_input") == tensors.end()
        || tensors.find("ans_grad_weight") == tensors.end()
        || tensors.find("ans_grad_bias") == tensors.end()) {
        throw std::runtime_error("Invalid Test");
    }
    test->_attributes->grad_input = tensors["grad_input"];
    test->_attributes->grad_weight = tensors["grad_weight"];
    test->_attributes->grad_bias = tensors["grad_bias"];
    test->_attributes->grad_output = tensors["grad_output"];
    test->_attributes->weight = tensors["weight"];
    test->_attributes->input_standardization = tensors["input_standardization"];
    test->_attributes->input_std_deviation = tensors["input_std_deviation"];
    test->_attributes->ans_grad_input = tensors["ans_grad_input"];
    test->_attributes->ans_grad_weight = tensors["ans_grad_weight"];
    test->_attributes->ans_grad_bias = tensors["ans_grad_bias"];
    test->_attributes->bias_exist = *reinterpret_cast<bool *>(attributes["bias_exist"].data());

    return test;
}
std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    infiniopLayerNormBackwardDescriptor_t op_desc;
    auto grad_input = _attributes->grad_input->to(device, device_id);
    auto grad_weight = _attributes->grad_weight->to(device, device_id);
    auto grad_bias = _attributes->grad_bias->to(device, device_id);
    auto grad_output = _attributes->grad_output->to(device, device_id);
    auto weight = _attributes->weight->to(device, device_id);
    auto input_standardization = _attributes->input_standardization->to(device, device_id);
    auto input_std_deviation = _attributes->input_std_deviation->to(device, device_id);
    CHECK_OR(infiniopCreateLayerNormBackwardDescriptor(handle, &op_desc,
            grad_input->desc(),
            grad_weight->desc(),
            (_attributes->bias_exist) ? (grad_bias->desc()) : nullptr,
            grad_output->desc(),
            weight->desc(),
            input_standardization->desc(),
            input_std_deviation->desc()
        ),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create op descriptor."));
    size_t workspace_size;
    CHECK_OR(infiniopGetLayerNormBackwardWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));
    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    CHECK_OR(infiniopLayerNormBackward(op_desc, workspace, workspace_size,
                    grad_input->data(),
                    grad_weight->data(),
                    (_attributes->bias_exist) ? (grad_bias->data()) : nullptr,
                    grad_output->data(),
                    weight->data(),
                    input_standardization->data(),
                    input_std_deviation->data(),
                    nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    try {
        allClose(grad_input, _attributes->ans_grad_input, _rtol, _atol);
        allClose(grad_weight, _attributes->ans_grad_weight, _rtol, _atol);
        if (_attributes->bias_exist)
            allClose(grad_bias, _attributes->ans_grad_bias, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopLayerNormBackward(
                op_desc, workspace, workspace_size,
                grad_input->data(),
                grad_weight->data(),
                (_attributes->bias_exist) ? (grad_bias->data()) : nullptr,
                grad_output->data(),
                weight->data(),
                input_standardization->data(),
                input_std_deviation->data(),
                nullptr);
        },
        warm_ups, iterations);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"bias_exist"};
}

std::vector<std::string> Test::tensor_names() {
    return {"grad_input", "grad_weight", "grad_bias", "grad_output", "weight", "input_standardization", "input_std_deviation", "ans_grad_input", "ans_grad_weight", "ans_grad_bias"};
}

std::vector<std::string> Test::output_names() {
    return {"grad_input", "grad_weight", "grad_bias"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- grad_input: " << _attributes->grad_input->info() << std::endl;
    oss << "- grad_weight: " << _attributes->grad_weight->info() << std::endl;
    oss << "- grad_bias: " << (_attributes->bias_exist ? _attributes->grad_bias->info() : "null") << std::endl;
    oss << "- grad_output: " << _attributes->grad_output->info() << std::endl;
    oss << "- weight: " << _attributes->weight->info() << std::endl;
    oss << "- input_standardization: " << _attributes->input_standardization->info() << std::endl;
    oss << "- input_std_deviation: " << _attributes->input_std_deviation->info() << std::endl;

    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::layer_norm_backward
