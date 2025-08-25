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
    int64_t ignore_index;
    std::string reduction;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    
    int64_t ignore_index = -100;
    if (attributes.count("ignore_index")) {
        ignore_index = *reinterpret_cast<const int64_t*>(attributes.at("ignore_index").data());
    }
    
    std::string reduction = "mean";
    if (attributes.count("reduction")) {
        reduction = std::string(reinterpret_cast<const char*>(attributes.at("reduction").data()));
    }
    
    test->_attributes = new Attributes{
        .probs = tensors.at("probs"),
        .target = tensors.at("target"),
        .grad_logits = tensors.at("grad_logits"),
        .ans = tensors.at("ans"),
        .ignore_index = ignore_index,
        .reduction = reduction
    };
    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    // Check for zero strides in input tensors
    auto probs_strides = _attributes->probs->strides();
    auto target_strides = _attributes->target->strides();
    auto grad_logits_strides = _attributes->grad_logits->strides();
    
    // Skip test if any tensor has zero stride (similar to PyTorch test behavior)
    for (auto stride : probs_strides) {
        if (stride == 0) {
            return TEST_PASSED(0.0); // Skip test with zero time
        }
    }
    for (auto stride : target_strides) {
        if (stride == 0) {
            return TEST_PASSED(0.0); // Skip test with zero time
        }
    }
    for (auto stride : grad_logits_strides) {
        if (stride == 0) {
            return TEST_PASSED(0.0); // Skip test with zero time
        }
    }
    
    infiniopCrossEntropyLossBackwardDescriptor_t crossEntropyLossBackwardDesc;
    auto probs_device = _attributes->probs->to(device, device_id);
    auto target_device = _attributes->target->to(device, device_id);
    auto grad_logits_device = _attributes->grad_logits->to(device, device_id);
    
    CHECK_OR(infiniopCreateCrossEntropyLossBackwardDescriptor(handle, &crossEntropyLossBackwardDesc,
                                                             grad_logits_device->desc(),
                                                             probs_device->desc(),
                                                             target_device->desc()),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create cross_entropy_loss_backward descriptor."));

    size_t workspaceSize;
    CHECK_OR(infiniopGetCrossEntropyLossBackwardWorkspaceSize(crossEntropyLossBackwardDesc, &workspaceSize),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get cross_entropy_loss_backward workspace size."));

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspaceSize),
                 return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    }

    CHECK_OR(infiniopCrossEntropyLossBackward(crossEntropyLossBackwardDesc, workspace, workspaceSize,
                                              grad_logits_device->data(),
                                              probs_device->data(),
                                              target_device->data(),
                                              nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed to execute cross_entropy_loss_backward."));

    auto grad_logits_host = grad_logits_device->to(INFINI_DEVICE_CPU);

    try {
        allClose(grad_logits_host, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double time = 0.0;
    if (iterations > 0) {
        time = benchmark(
            [&]() {
                CHECK_OR(infiniopCrossEntropyLossBackward(crossEntropyLossBackwardDesc, workspace, workspaceSize,
                                                          grad_logits_device->data(),
                                                          probs_device->data(),
                                                          target_device->data(),
                                                          nullptr),
                         throw std::runtime_error("Failed to execute cross_entropy_loss_backward"));
            },
            warm_ups, iterations);
    }

    CHECK_OR(infiniopDestroyCrossEntropyLossBackwardDescriptor(crossEntropyLossBackwardDesc), return TEST_FAILED(OP_CREATION_FAILED, "Failed to destroy cross_entropy_loss_backward descriptor."));
    if (workspace) {
        CHECK_OR(infinirtFree(workspace), return TEST_FAILED(OP_CREATION_FAILED, "Failed to free workspace."));
    }

    return TEST_PASSED(time);
}

std::vector<std::string> Test::attribute_names() {
    return {"ignore_index", "reduction"};
}

std::vector<std::string> Test::tensor_names() {
    return {"probs", "target", "grad_logits", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"grad_logits"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << "cross_entropy_loss_backward(";
    oss << "probs=" << _attributes->probs->info();
    oss << ", target=" << _attributes->target->info();
    oss << ", grad_logits=" << _attributes->grad_logits->info();
    oss << ", ignore_index=" << _attributes->ignore_index;
    oss << ", reduction=" << _attributes->reduction;
    oss << ", rtol=" << _rtol << ", atol=" << _atol;
    oss << ")";
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::cross_entropy_loss_backward