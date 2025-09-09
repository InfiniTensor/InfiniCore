#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::where {

struct Test::Attributes {
    std::shared_ptr<Tensor> cond;
    std::shared_ptr<Tensor> a;
    std::shared_ptr<Tensor> b;
    std::shared_ptr<Tensor> out;
    std::shared_ptr<Tensor> ans;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol, bool equal_nan) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol, equal_nan));
    test->_attributes = new Attributes();

    if (tensors.find("condition") == tensors.end()
        || tensors.find("a") == tensors.end()
        || tensors.find("b") == tensors.end()
        || tensors.find("c") == tensors.end()
        || tensors.find("ans") == tensors.end()) {
        throw std::runtime_error("Invalid Test");
    }

    test->_attributes->cond = tensors["condition"];
    test->_attributes->a = tensors["a"];
    test->_attributes->b = tensors["b"];
    test->_attributes->out = tensors["c"];
    test->_attributes->ans = tensors["ans"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations) {

    infiniopWhereDescriptor_t op_desc;

    auto cond = _attributes->cond->to(device, device_id);
    auto a = _attributes->a->to(device, device_id);
    auto b = _attributes->b->to(device, device_id);
    auto out = _attributes->out->to(device, device_id);

    CHECK_OR(infiniopCreateWhereDescriptor(handle, &op_desc,
                                           out->desc(),
                                           cond->desc(),
                                           a->desc(),
                                           b->desc()),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create where descriptor."));

    size_t workspace_size;
    CHECK_OR(infiniopGetWhereWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));

    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));

    CHECK_OR(infiniopWhere(op_desc, workspace, workspace_size,
                           out->data(),
                           cond->data(),
                           a->data(),
                           b->data(),
                           nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    try {
        // where 输出通常与 a/b 同 dtype；若为整型/布尔，建议 rtol=0, atol=0
        allClose(out, _attributes->ans, _rtol, _atol, _equal_nan);
    } catch (const std::exception &e) {
        infiniopDestroyWhereDescriptor(op_desc);
        infinirtFree(workspace);
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.0;
    elapsed_time = benchmark(
        [=]() {
            infiniopWhere(op_desc, workspace, workspace_size,
                          out->data(),
                          cond->data(),
                          a->data(),
                          b->data(),
                          nullptr);
        },
        warm_ups, iterations);

    infiniopDestroyWhereDescriptor(op_desc);
    infinirtFree(workspace);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {};
}

std::vector<std::string> Test::tensor_names() {
    return {"condition", "a", "b", "c", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"c"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- condition: " << _attributes->cond->info() << std::endl;
    oss << "- a: " << _attributes->a->info() << std::endl;
    oss << "- b: " << _attributes->b->info() << std::endl;
    oss << "- out: " << _attributes->out->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << ", equal_nan=" << _equal_nan << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::where
