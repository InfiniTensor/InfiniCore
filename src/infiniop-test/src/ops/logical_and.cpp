#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::logical_and {
struct Test::Attributes {
    std::shared_ptr<Tensor> a;
    std::shared_ptr<Tensor> b;
    std::shared_ptr<Tensor> c;
    std::shared_ptr<Tensor> ans;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes{
        .a = tensors.at("a"),
        .b = tensors.at("b"),
        .c = tensors.at("c"),
        .ans = tensors.at("ans")
    };
    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    infiniopAndDescriptor_t andDesc;
    auto a_device = _attributes->a->to(device, device_id);
    auto b_device = _attributes->b->to(device, device_id);
    auto c_device = _attributes->c->to(device, device_id);
    
    CHECK_OR(infiniopCreateAndDescriptor(handle, &andDesc,
                                         c_device->desc(),
                                         a_device->desc(),
                                         b_device->desc()), return TEST_FAILED(OP_CREATION_FAILED, "Failed to create and descriptor."));

    size_t workspaceSize;
    CHECK_OR(infiniopGetAndWorkspaceSize(andDesc, &workspaceSize),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get and workspace size."));

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspaceSize),
                 return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    }

    CHECK_OR(infiniopAnd(andDesc, workspace, workspaceSize,
                         c_device->data(),
                         a_device->data(),
                         b_device->data(),
                         nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed to execute and."));

    auto c_host = c_device->to(INFINI_DEVICE_CPU);

    try {
        allClose(c_host, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double time = 0.0;
    if (iterations > 0) {
        time = benchmark(
            [&]() {
                CHECK_OR(infiniopAnd(andDesc, workspace, workspaceSize,
                                     c_device->data(),
                                     a_device->data(),
                                     b_device->data(),
                                     nullptr),
                         throw std::runtime_error("Failed to execute and"));
            },
            warm_ups, iterations);
    }

    CHECK_OR(infiniopDestroyAndDescriptor(andDesc), return TEST_FAILED(OP_CREATION_FAILED, "Failed to destroy and descriptor."));
    if (workspace) {
        CHECK_OR(infinirtFree(workspace), return TEST_FAILED(OP_CREATION_FAILED, "Failed to free workspace."));
    }

    return TEST_PASSED(time);
}

std::vector<std::string> Test::attribute_names() {
    return {};
}

std::vector<std::string> Test::tensor_names() {
    return {"a", "b", "c", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"c"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << "logical_and(";
    oss << "a=" << _attributes->a->info();
    oss << ", b=" << _attributes->b->info();
    oss << ", c=" << _attributes->c->info();
    oss << ", rtol=" << _rtol << ", atol=" << _atol;
    oss << ")";
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::logical_and