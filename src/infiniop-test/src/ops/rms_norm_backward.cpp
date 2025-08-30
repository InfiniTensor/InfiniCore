#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::rms_norm_backward {
struct Test::Attributes {
    std::shared_ptr<Tensor> grad_x;
    std::shared_ptr<Tensor> grad_w;
    std::shared_ptr<Tensor> grad_y;
    std::shared_ptr<Tensor> x;
    std::shared_ptr<Tensor> w;
    std::shared_ptr<Tensor> ans_grad_x;
    std::shared_ptr<Tensor> ans_grad_w;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    if (tensors.find("grad_x") == tensors.end()
        || tensors.find("grad_w") == tensors.end()
        || tensors.find("grad_y") == tensors.end()
        || tensors.find("x") == tensors.end()
        || tensors.find("w") == tensors.end()
        || tensors.find("ans_grad_x") == tensors.end()
        || tensors.find("ans_grad_w") == tensors.end()) {
        throw std::runtime_error("Invalid Test");
    }
    test->_attributes->grad_x = tensors["grad_x"];
    test->_attributes->grad_w = tensors["grad_w"];
    test->_attributes->grad_y = tensors["grad_y"];
    test->_attributes->x = tensors["x"];
    test->_attributes->w = tensors["w"];
    test->_attributes->ans_grad_x = tensors["ans_grad_x"];
    test->_attributes->ans_grad_w = tensors["ans_grad_w"];

    return test;
}
std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    infiniopRMSNormBackwardDescriptor_t op_desc;
    auto grad_x = _attributes->grad_x->to(device, device_id);
    auto grad_w = _attributes->grad_w->to(device, device_id);
    auto grad_y = _attributes->grad_y->to(device, device_id);
    auto x = _attributes->x->to(device, device_id);
    auto w = _attributes->w->to(device, device_id);
    CHECK_OR(infiniopCreateRMSNormBackwardDescriptor(handle, &op_desc,
            grad_x->desc(),
            grad_w->desc(),
            grad_y->desc(),
            x->desc(),
            w->desc()
        ),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create op descriptor."));
    size_t workspace_size;
    CHECK_OR(infiniopGetRMSNormBackwardWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));
    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    CHECK_OR(infiniopRMSNormBackward(op_desc, workspace, workspace_size,
                    grad_x->data(),
                    grad_w->data(),
                    grad_y->data(),
                    x->data(),
                    w->data(),
                    nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    try {
        allClose(grad_x, _attributes->ans_grad_x, _rtol, _atol);
        allClose(grad_w, _attributes->ans_grad_w, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopRMSNormBackward(
                op_desc, workspace, workspace_size,
                grad_x->data(),
                grad_w->data(),
                grad_y->data(),
                x->data(),
                w->data(),
                nullptr);
        },
        warm_ups, iterations);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {};
}

std::vector<std::string> Test::tensor_names() {
    return {"grad_x", "grad_w", "grad_y", "x", "w", "ans_grad_x", "ans_grad_w"};
}

std::vector<std::string> Test::output_names() {
    return {"grad_x", "grad_w"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- grad_x: " << _attributes->grad_x->info() << std::endl;
    oss << "- grad_w: " << _attributes->grad_w->info() << std::endl;
    oss << "- grad_y: " << _attributes->grad_y->info() << std::endl;
    oss << "- x: " << _attributes->x->info() << std::endl;
    oss << "- w: " << _attributes->w->info() << std::endl;

    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::rms_norm_backward
