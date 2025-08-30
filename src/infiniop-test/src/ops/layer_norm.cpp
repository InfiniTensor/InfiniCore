#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::layer_norm {
struct Test::Attributes {
    bool bias_exist;
    std::shared_ptr<Tensor> output;
    std::shared_ptr<Tensor> input_standardization;
    std::shared_ptr<Tensor> input_std_deviation;
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    float eps;
    std::shared_ptr<Tensor> ans_output;
    std::shared_ptr<Tensor> ans_input_standardization;
    std::shared_ptr<Tensor> ans_input_std_deviation;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    if (tensors.find("output") == tensors.end()
        || tensors.find("input_standardization") == tensors.end()
        || tensors.find("input_std_deviation") == tensors.end()
        || tensors.find("input") == tensors.end()
        || tensors.find("weight") == tensors.end()
        || tensors.find("bias") == tensors.end()
        || tensors.find("ans_output") == tensors.end()
        || tensors.find("ans_input_standardization") == tensors.end()
        || tensors.find("ans_input_std_deviation") == tensors.end()) {
        throw std::runtime_error("Invalid Test");
    }
    test->_attributes->output = tensors["output"];
    test->_attributes->input_standardization = tensors["input_standardization"];
    test->_attributes->input_std_deviation = tensors["input_std_deviation"];
    test->_attributes->input = tensors["input"];
    test->_attributes->weight = tensors["weight"];
    test->_attributes->bias = tensors["bias"];
    test->_attributes->ans_output = tensors["ans_output"];
    test->_attributes->ans_input_standardization = tensors["ans_input_standardization"];
    test->_attributes->ans_input_std_deviation = tensors["ans_input_std_deviation"];
    test->_attributes->eps = *reinterpret_cast<float*>(attributes["eps"].data());
    test->_attributes->bias_exist = *reinterpret_cast<bool *>(attributes["bias_exist"].data());
    return test;
}
std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    infiniopLayerNormDescriptor_t op_desc;
    auto output = _attributes->output->to(device, device_id);
    auto input_standardization = _attributes->input_standardization->to(device, device_id);
    auto input_std_deviation = _attributes->input_std_deviation->to(device, device_id);
    auto input = _attributes->input->to(device, device_id);
    auto weight = _attributes->weight->to(device, device_id);
    auto bias = _attributes->bias->to(device, device_id);
    auto eps = _attributes->eps;
    CHECK_OR(infiniopCreateLayerNormDescriptor(handle, &op_desc,
            output->desc(),
            input_standardization->desc(),
            input_std_deviation->desc(),
            input->desc(),
            weight->desc(),
            (_attributes->bias_exist) ? bias->desc() : nullptr,
            eps
        ),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create op descriptor."));
    size_t workspace_size;
    CHECK_OR(infiniopGetLayerNormWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));
    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    CHECK_OR(infiniopLayerNorm(op_desc, workspace, workspace_size,
                    output->data(),
                    input_standardization->data(),
                    input_std_deviation->data(),
                    input->data(),
                    weight->data(),
                    (_attributes->bias_exist) ? bias->data() : nullptr,
                    nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    try {
        allClose(output, _attributes->ans_output, _rtol, _atol);
        allClose(input_standardization, _attributes->ans_input_standardization, _rtol, _atol);
        allClose(input_std_deviation, _attributes->ans_input_std_deviation, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopLayerNorm(
                op_desc, workspace, workspace_size,
                output->data(),
                input_standardization->data(),
                input_std_deviation->data(),
                input->data(),
                weight->data(),
                (_attributes->bias_exist) ? bias->data() : nullptr,
                nullptr);
        },
        warm_ups, iterations);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"bias_exist", "eps"};
}

std::vector<std::string> Test::tensor_names() {
    return {"output", "input_standardization", "input_std_deviation", "input", "weight", "bias", "ans_output", "ans_input_standardization", "ans_input_std_deviation"};
}

std::vector<std::string> Test::output_names() {
    return {"output", "input_standardization", "input_std_deviation"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- output: " << _attributes->output->info() << std::endl;
    oss << "- input_standardization: " << _attributes->input_standardization->info() << std::endl;
    oss << "- input_std_deviation: " << _attributes->input_std_deviation->info() << std::endl;
    oss << "- input: " << _attributes->input->info() << std::endl;
    oss << "- weight: " << _attributes->weight->info() << std::endl;
    oss << "- bias: " << (_attributes->bias_exist ? _attributes->bias->info() : "null") << std::endl;
    oss << "- eps: " << _attributes->eps << std::endl;

    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::layer_norm
