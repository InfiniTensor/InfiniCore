#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::batch_norm {
struct Test::Attributes {
    std::shared_ptr<Tensor> output;
    std::shared_ptr<Tensor> running_mean;
    std::shared_ptr<Tensor> running_var;
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    float momentum;
    float eps;
    std::shared_ptr<Tensor> ans_output;
    std::shared_ptr<Tensor> ans_running_mean;
    std::shared_ptr<Tensor> ans_running_var;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    if (tensors.find("output") == tensors.end()
        || tensors.find("running_mean") == tensors.end()
        || tensors.find("running_var") == tensors.end()
        || tensors.find("input") == tensors.end()
        || tensors.find("weight") == tensors.end()
        || tensors.find("bias") == tensors.end()
        || tensors.find("ans_output") == tensors.end()
        || tensors.find("ans_running_mean") == tensors.end()
        || tensors.find("ans_running_var") == tensors.end()) {
        throw std::runtime_error("Invalid Test");
    }
    test->_attributes->output = tensors["output"];
    test->_attributes->running_mean = tensors["running_mean"];
    test->_attributes->running_var = tensors["running_var"];
    test->_attributes->input = tensors["input"];
    test->_attributes->weight = tensors["weight"];
    test->_attributes->bias = tensors["bias"];
    test->_attributes->ans_output = tensors["ans_output"];
    test->_attributes->ans_running_mean = tensors["ans_running_mean"];
    test->_attributes->ans_running_var = tensors["ans_running_var"];
    test->_attributes->momentum = *reinterpret_cast<float*>(attributes["momentum"].data());
    test->_attributes->eps = *reinterpret_cast<float*>(attributes["eps"].data());

    return test;
}
std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    infiniopBatchNormDescriptor_t op_desc;
    auto output = _attributes->output->to(device, device_id);
    auto running_mean = _attributes->running_mean->to(device, device_id);
    auto running_var = _attributes->running_var->to(device, device_id);
    auto input = _attributes->input->to(device, device_id);
    auto weight = _attributes->weight->to(device, device_id);
    auto bias = _attributes->bias->to(device, device_id);
    auto momentum = _attributes->momentum;
    auto eps = _attributes->eps;
    CHECK_OR(infiniopCreateBatchNormDescriptor(handle, &op_desc,
            output->desc(),
            running_mean->desc(),
            running_var->desc(),
            input->desc(),
            weight->desc(),
            bias->desc(),
            momentum,
            eps
        ),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create op descriptor."));
    size_t workspace_size;
    CHECK_OR(infiniopGetBatchNormWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));
    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    CHECK_OR(infiniopBatchNorm(op_desc, workspace, workspace_size,
                    output->data(),
                    running_mean->data(),
                    running_var->data(),
                    input->data(),
                    weight->data(),
                    bias->data(),
                    nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    try {
        allClose(output, _attributes->ans_output, _rtol, _atol);
        allClose(running_mean, _attributes->ans_running_mean, _rtol, _atol);
        allClose(running_var, _attributes->ans_running_var, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopBatchNorm(
                op_desc, workspace, workspace_size,
                output->data(),
                running_mean->data(),
                running_var->data(),
                input->data(),
                weight->data(),
                bias->data(),
                nullptr);
        },
        warm_ups, iterations);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"momentum", "eps"};
}

std::vector<std::string> Test::tensor_names() {
    return {"output", "running_mean", "running_var", "input", "weight", "bias", "ans_output", "ans_running_mean", "ans_running_var"};
}

std::vector<std::string> Test::output_names() {
    return {"output", "running_mean", "running_var"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- output: " << _attributes->output->info() << std::endl;
    oss << "- running_mean: " << _attributes->running_mean->info() << std::endl;
    oss << "- running_var: " << _attributes->running_var->info() << std::endl;
    oss << "- input: " << _attributes->input->info() << std::endl;
    oss << "- weight: " << _attributes->weight->info() << std::endl;
    oss << "- bias: " << _attributes->bias->info() << std::endl;
    oss << "- momentum: " << _attributes->momentum << std::endl;
    oss << "- eps: " << _attributes->eps << std::endl;

    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::batch_norm
