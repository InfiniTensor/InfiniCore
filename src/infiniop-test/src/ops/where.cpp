#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::where {
struct Test::Attributes {
    std::shared_ptr<Tensor> a;
    std::shared_ptr<Tensor> b;
    std::shared_ptr<Tensor> condition;
    std::shared_ptr<Tensor> c;
    std::shared_ptr<Tensor> ans;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    if (tensors.find("a") == tensors.end()
        || tensors.find("b") == tensors.end()
        || tensors.find("condition") == tensors.end()
        || tensors.find("c") == tensors.end()
        || tensors.find("ans") == tensors.end()) {
        throw std::runtime_error("Invalid Test");
    }

    test->_attributes->a = tensors["a"];
    test->_attributes->b = tensors["b"];
    test->_attributes->condition = tensors["condition"];
    test->_attributes->c = tensors["c"];
    test->_attributes->ans = tensors["ans"];

    auto elemType = test->_attributes->a->ggml_type();
    if (elemType == GGML_TYPE_I8) {
        test->_rtol = 1e-5;
        test->_atol = 1e-5;
    }
    if (elemType == GGML_TYPE_I16) {
        test->_rtol = 1e-5;
        test->_atol = 1e-5;
    }
    if (elemType == GGML_TYPE_I32) {
        test->_rtol = 1e-5;
        test->_atol = 1e-5;
    }
    if (elemType == GGML_TYPE_I64) {
        test->_rtol = 1e-5;
        test->_atol = 1e-5;
    }
    if (elemType == GGML_TYPE_F16) {
        test->_rtol = 1e-7;
        test->_atol = 1e-7;
    }
    if (elemType == GGML_TYPE_F32) {
        test->_rtol = 1e-7;
        test->_atol = 1e-7;
    }
    if (elemType == GGML_TYPE_F64) {
        test->_rtol = 1e-7;
        test->_atol = 1e-7;
    }
    if (elemType == GGML_TYPE_BF16) {
        test->_rtol = 1e-5;
        test->_atol = 1e-5;
    }

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    infiniopWhereDescriptor_t op_desc;
    auto a = _attributes->a->to(device, device_id);
    auto b = _attributes->b->to(device, device_id);
    auto condition = _attributes->condition->to(device, device_id);
    auto c = _attributes->c->to(device, device_id);
    CHECK_OR(infiniopCreateWhereDescriptor(handle, &op_desc,
                                           c->desc(),
                                           a->desc(),
                                           b->desc(),
                                           condition->desc()),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create op descriptor."));
    size_t workspace_size;
    CHECK_OR(infiniopGetWhereWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));
    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    CHECK_OR(infiniopWhere(op_desc, workspace, workspace_size,
                           c->data(),
                           a->data(),
                           b->data(),
                           condition->data(),
                           nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    try {
        allClose(c, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopWhere(
                op_desc, workspace, workspace_size,
                c->data(),
                a->data(),
                b->data(),
                condition->data(),
                nullptr);
        },
        warm_ups, iterations);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {};
}

std::vector<std::string> Test::tensor_names() {
    return {"a", "b", "condition", "c", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"c"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- a: " << _attributes->a->info() << std::endl;
    oss << "- b: " << _attributes->b->info() << std::endl;
    oss << "- condition: " << _attributes->condition->info() << std::endl;
    oss << "- c: " << _attributes->c->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::where
