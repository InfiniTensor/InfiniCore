#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::rms_norm {
struct Test::Attributes {
    float epsilon;
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> ans;
    std::shared_ptr<Tensor> result;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();

    if (attributes.find("epsilon") == attributes.end()
        || tensors.find("input") == tensors.end()
        || tensors.find("weight") == tensors.end()
        || tensors.find("ans") == tensors.end()
        || tensors.find("result") == tensors.end()) {
        throw std::runtime_error("Invalid Test: Missing attributes or tensors");
    }

    test->_attributes->epsilon = *reinterpret_cast<float *>(attributes["epsilon"].data());

    test->_attributes->input = tensors["input"];
    test->_attributes->weight = tensors["weight"];
    test->_attributes->ans = tensors["ans"];
    test->_attributes->result = tensors["result"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations) {

    infiniopRMSNormDescriptor_t op_desc;
    CHECK_OR(infiniopCreateRMSNormDescriptor(handle, &op_desc,
                                             _attributes->result->desc(),
                                             _attributes->input->desc(),
                                             _attributes->weight->desc(),
                                             _attributes->epsilon),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create RMSNorm descriptor"));

    auto input = _attributes->input->to(device, device_id);
    auto weight = _attributes->weight->to(device, device_id);
    auto result = _attributes->result->to(device, device_id);

    size_t workspace_size;
    CHECK_OR(infiniopGetRMSNormWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size"));
    void *workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspace_size),
                 return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace"));
    }

    CHECK_OR(infiniopRMSNorm(op_desc,
                             workspace, workspace_size,
                             result->data(),
                             input->data(),
                             weight->data(),
                             nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "RMSNorm execution failed"));

    try {
        allClose(result, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopRMSNorm(op_desc,
                            workspace, workspace_size,
                            result->data(),
                            input->data(),
                            weight->data(),
                            nullptr);
        },
        warm_ups, iterations);

    if (workspace != nullptr) {
        infinirtFree(workspace);
    }

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"epsilon"};
}

std::vector<std::string> Test::tensor_names() {
    return {"input", "weight", "ans", "result"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- epsilon=" << _attributes->epsilon << std::endl;
    oss << "- input: " << _attributes->input->info() << std::endl;
    oss << "- weight: " << _attributes->weight->info() << std::endl;
    oss << "- result: " << _attributes->result->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::rms_norm
