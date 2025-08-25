#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::cross_entropy_loss_backward {
struct Test::Attributes {
    std::shared_ptr<Tensor> probs;
    std::shared_ptr<Tensor> target;
    std::shared_ptr<Tensor> grad_logits;
    std::shared_ptr<Tensor> ans;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    if (tensors.find("probs") == tensors.end()
        || tensors.find("target") == tensors.end()
        || tensors.find("grad_logits") == tensors.end()
        || tensors.find("ans") == tensors.end()) {
        throw std::runtime_error("Invalid Test");
    }

    test->_attributes->probs = tensors["probs"];
    test->_attributes->target = tensors["target"];
    test->_attributes->grad_logits = tensors["grad_logits"];
    test->_attributes->ans = tensors["ans"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    infiniopCrossEntropyLossBackwardDescriptor_t op_desc;
    auto probs = _attributes->probs->to(device, device_id);
    auto target = _attributes->target->to(device, device_id);
    auto grad_logits = _attributes->grad_logits->to(device, device_id);
    CHECK_OR(infiniopCreateCrossEntropyLossBackwardDescriptor(handle, &op_desc,
                                                              grad_logits->desc(),
                                                              probs->desc(),
                                                              target->desc()),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create op descriptor."));
    size_t workspace_size;
    CHECK_OR(infiniopGetCrossEntropyLossBackwardWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));
    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    CHECK_OR(infiniopCrossEntropyLossBackward(op_desc, workspace, workspace_size,
                                              grad_logits->data(),
                                              probs->data(),
                                              target->data(),
                                              nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    try {
        allClose(grad_logits, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopCrossEntropyLossBackward(
                op_desc, workspace, workspace_size,
                grad_logits->data(),
                probs->data(),
                target->data(),
                nullptr);
        },
        warm_ups, iterations);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {};
}

std::vector<std::string> Test::tensor_names() {
    return {"probs", "target", "grad_logits", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"grad_logits"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- probs: " << _attributes->probs->info() << std::endl;
    oss << "- target: " << _attributes->target->info() << std::endl;
    oss << "- grad_logits: " << _attributes->grad_logits->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::cross_entropy_loss_backward
