#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::cross_entropy_loss_backward {

struct Test::Attributes {
    std::shared_ptr<Tensor> probs;       // 概率 (softmax 输出)
    std::shared_ptr<Tensor> target;      // one-hot 标签 (与 logits 同形状)
    std::shared_ptr<Tensor> grad_logits; // 输出: dL/dlogits
    std::shared_ptr<Tensor> ans;         // 参考结果
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol, bool equal_nan) {

    auto test = std::shared_ptr<Test>(new Test(rtol, atol, equal_nan));
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
    infiniopHandle_t handle, infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations) {

    infiniopCrossEntropyLossBackwardDescriptor_t op_desc;

    auto probs = _attributes->probs->to(device, device_id);
    auto target = _attributes->target->to(device, device_id);
    auto grad_logits = _attributes->grad_logits->to(device, device_id);

    CHECK_OR(infiniopCreateCrossEntropyLossBackwardDescriptor(
                 handle, &op_desc,
                 /*dst     */ grad_logits->desc(),
                 /*probs   */ probs->desc(),
                 /*target  */ target->desc()),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create cross_entropy_loss_backward descriptor."));

    size_t workspace_size = 0;
    CHECK_OR(infiniopGetCrossEntropyLossBackwardWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));

    void *workspace = nullptr;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));

    CHECK_OR(infiniopCrossEntropyLossBackward(
                 op_desc, workspace, workspace_size,
                 /*dst     */ grad_logits->data(),
                 /*probs   */ probs->data(),
                 /*target  */ target->data(),
                 /*stream  */ nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    try {
        // 浮点比较；如混合精度，建议设置合理的 rtol/atol
        allClose(grad_logits, _attributes->ans, _rtol, _atol, _equal_nan);
    } catch (const std::exception &e) {
        infiniopDestroyCrossEntropyLossBackwardDescriptor(op_desc);
        infinirtFree(workspace);
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = benchmark(
        [=]() {
            infiniopCrossEntropyLossBackward(
                op_desc, workspace, workspace_size,
                grad_logits->data(),
                probs->data(),
                target->data(),
                nullptr);
        },
        warm_ups, iterations);

    infiniopDestroyCrossEntropyLossBackwardDescriptor(op_desc);
    infinirtFree(workspace);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() { return {}; }

std::vector<std::string> Test::tensor_names() {
    return {"probs", "target", "grad_logits", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"grad_logits"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- probs:       " << _attributes->probs->info() << std::endl;
    oss << "- target:      " << _attributes->target->info() << std::endl;
    oss << "- grad_logits: " << _attributes->grad_logits->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol
        << ", equal_nan=" << _equal_nan << std::endl;
    return oss.str();
}

Test::~Test() { delete _attributes; }

} // namespace infiniop_test::cross_entropy_loss_backward
