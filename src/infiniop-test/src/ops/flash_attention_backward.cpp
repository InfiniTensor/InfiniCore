#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::flash_attention_backward {
struct Test::Attributes {
    int mask_type;
    std::shared_ptr<Tensor> q;
    std::shared_ptr<Tensor> k;
    std::shared_ptr<Tensor> v;
    std::shared_ptr<Tensor> mask;
    std::shared_ptr<Tensor> grad_out;
    std::shared_ptr<Tensor> grad_q;
    std::shared_ptr<Tensor> grad_k;
    std::shared_ptr<Tensor> grad_v;
    std::shared_ptr<Tensor> ans_grad_q;
    std::shared_ptr<Tensor> ans_grad_k;
    std::shared_ptr<Tensor> ans_grad_v;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {

    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();

    if (attributes.find("mask_type") == attributes.end()
        || tensors.find("q") == tensors.end()
        || tensors.find("k") == tensors.end()
        || tensors.find("v") == tensors.end()
        || tensors.find("grad_out") == tensors.end()
        || tensors.find("grad_q") == tensors.end()
        || tensors.find("grad_k") == tensors.end()
        || tensors.find("grad_v") == tensors.end()
        || tensors.find("ans_grad_q") == tensors.end()
        || tensors.find("ans_grad_k") == tensors.end()
        || tensors.find("ans_grad_v") == tensors.end()) {
        throw std::runtime_error("Invalid Test: Missing attributes or tensors");
    }

    if (tensors.find("mask") == tensors.end()) {
        test->_attributes->mask = nullptr;
    } else {
        test->_attributes->mask = tensors["mask"];
    }

    test->_attributes->mask_type = *reinterpret_cast<int *>(attributes["mask_type"].data());

    test->_attributes->q = tensors["q"];
    test->_attributes->k = tensors["k"];
    test->_attributes->v = tensors["v"];
    test->_attributes->grad_out = tensors["grad_out"];
    test->_attributes->grad_q = tensors["grad_q"];
    test->_attributes->grad_k = tensors["grad_k"];
    test->_attributes->grad_v = tensors["grad_v"];
    test->_attributes->ans_grad_q = tensors["ans_grad_q"];
    test->_attributes->ans_grad_k = tensors["ans_grad_k"];
    test->_attributes->ans_grad_v = tensors["ans_grad_v"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations) {

    infiniopFlashAttentionBackwardDescriptor_t op_desc;
    infiniopAttentionMaskType_t mask_type = static_cast<infiniopAttentionMaskType_t>(_attributes->mask_type);
    CHECK_OR(infiniopCreateFlashAttentionBackwardDescriptor(
                 handle, &op_desc,
                 _attributes->grad_q->desc(),
                 _attributes->grad_k->desc(),
                 _attributes->grad_v->desc(),
                 _attributes->q->desc(),
                 _attributes->k->desc(),
                 _attributes->v->desc(),
                 _attributes->grad_out->desc(),
                 _attributes->mask->desc(),
                 mask_type),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create FlashAttentionBackward descriptor"));

    auto grad_q = _attributes->grad_q->to(device, device_id);
    auto grad_k = _attributes->grad_k->to(device, device_id);
    auto grad_v = _attributes->grad_v->to(device, device_id);
    auto q = _attributes->q->to(device, device_id);
    auto k = _attributes->k->to(device, device_id);
    auto v = _attributes->v->to(device, device_id);
    auto grad_out = _attributes->grad_out->to(device, device_id);
    auto mask = _attributes->mask ? _attributes->mask->to(device, device_id) : nullptr;

    size_t workspace_size;
    CHECK_OR(infiniopGetFlashAttentionBackwardWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size"));
    void *workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspace_size),
                 return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace"));
    }

    CHECK_OR(infiniopFlashAttentionBackward(op_desc,
                                            workspace, workspace_size,
                                            grad_q->data(),
                                            grad_k->data(),
                                            grad_v->data(),
                                            q->data(),
                                            k->data(),
                                            v->data(),
                                            grad_out->data(),
                                            mask ? mask->data() : nullptr,
                                            nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed to execute FlashAttentionBackward"));

    try {
        allClose(grad_q, _attributes->ans_grad_q, _rtol, _atol);
        allClose(grad_k, _attributes->ans_grad_k, _rtol, _atol);
        allClose(grad_v, _attributes->ans_grad_v, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0;

    elapsed_time = benchmark(
        [=]() {
            infiniopFlashAttentionBackward(op_desc,
                                           workspace, workspace_size,
                                           grad_q->data(),
                                           grad_k->data(),
                                           grad_v->data(),
                                           q->data(),
                                           k->data(),
                                           v->data(),
                                           grad_out->data(),
                                           mask ? mask->data() : nullptr,
                                           nullptr);
        },
        warm_ups, iterations);

    if (workspace != nullptr) {
        infinirtFree(workspace);
    }

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"mask_type"};
}

std::vector<std::string> Test::tensor_names() {
    return {"grad_q", "grad_k", "grad_v", "q", "k", "v", "grad_out", "mask",
            "ans_grad_q", "ans_grad_k", "ans_grad_v"};
}

std::vector<std::string> Test::output_names() {
    return {"grad_q", "grad_k", "grad_v"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- masktype=" << static_cast<infiniopAttentionMaskType_t>(_attributes->mask_type) << std::endl;
    oss << "- q: " << _attributes->q->info() << std::endl;
    oss << "- k: " << _attributes->k->info() << std::endl;
    oss << "- v: " << _attributes->v->info() << std::endl;
    oss << "- grad_out: " << _attributes->grad_out->info() << std::endl;
    oss << "- mask: " << (_attributes->mask ? _attributes->mask->info() : "none") << std::endl;
    oss << "- grad_q: " << _attributes->grad_q->info() << std::endl;
    oss << "- grad_k: " << _attributes->grad_k->info() << std::endl;
    oss << "- grad_v: " << _attributes->grad_v->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::flash_attention_backward
