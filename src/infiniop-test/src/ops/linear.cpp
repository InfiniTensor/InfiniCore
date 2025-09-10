#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::linear {
struct Test::Attributes {
    bool bias_exist;
    std::shared_ptr<Tensor> y;
    std::shared_ptr<Tensor> x;
    std::shared_ptr<Tensor> w;
    std::shared_ptr<Tensor> b;
    std::shared_ptr<Tensor> ans;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol, bool equal_nan) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol, equal_nan));
    test->_attributes = new Attributes();
    if (tensors.find("y") == tensors.end()
        || tensors.find("x") == tensors.end()
        || tensors.find("w") == tensors.end()
        || tensors.find("b") == tensors.end()
        || tensors.find("ans") == tensors.end()) {
        throw std::runtime_error("Invalid Test");
    }
    test->_attributes->y = tensors["y"];
    test->_attributes->x = tensors["x"];
    test->_attributes->w = tensors["w"];
    test->_attributes->b = tensors["b"];
    test->_attributes->ans = tensors["ans"];
    test->_attributes->bias_exist = *reinterpret_cast<bool *>(attributes["bias_exist"].data());

    return test;
}
std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    infiniopLinearDescriptor_t op_desc;
    auto y = _attributes->y->to(device, device_id);
    auto x = _attributes->x->to(device, device_id);
    auto w = _attributes->w->to(device, device_id);
    auto b = _attributes->b->to(device, device_id);
    CHECK_OR(infiniopCreateLinearDescriptor(handle, &op_desc,
            y->desc(),
            x->desc(),
            w->desc(),
            (_attributes->bias_exist) ? (b->desc() ) : nullptr
        ),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create op descriptor."));
    size_t workspace_size;
    CHECK_OR(infiniopGetLinearWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));
    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    CHECK_OR(infiniopLinear(op_desc, workspace, workspace_size,
                    y->data(),
                    x->data(),
                    w->data(),
                    (_attributes->bias_exist) ? b->data() : nullptr,
                    nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    try {
        allClose(y, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopLinear(
                op_desc, workspace, workspace_size,
                y->data(),
                x->data(),
                w->data(),
                (_attributes->bias_exist) ? b->data() : nullptr,
                nullptr);
        },
        warm_ups, iterations);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"bias_exist"};
}

std::vector<std::string> Test::tensor_names() {
    return {"y", "x", "w", "b", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"y"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- y: " << _attributes->y->info() << std::endl;
    oss << "- x: " << _attributes->x->info() << std::endl;
    oss << "- w: " << _attributes->w->info() << std::endl;
    oss << "- b: " << (_attributes->bias_exist ? _attributes->b->info() : "null") << std::endl;

    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::linear
