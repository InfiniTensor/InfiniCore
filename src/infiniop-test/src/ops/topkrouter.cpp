#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::topkrouter {
struct Test::Attributes {
    std::shared_ptr<Tensor> values;
    std::shared_ptr<Tensor> indices;
    std::shared_ptr<Tensor> x;
    std::shared_ptr<Tensor> correction_bias;
    float routed_scaling_factor;
    int topk;
    std::shared_ptr<Tensor> lable_values;
    std::shared_ptr<Tensor> lable_indices;
};

std::shared_ptr<Test> Test::build(std::unordered_map<std::string, std::vector<uint8_t>> attributes,
                                  std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors, double rtol,
                                  double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();

    if (attributes.find("routed_scaling_factor") == attributes.end() || attributes.find("topk") == attributes.end() || tensors.find("values") == tensors.end() || tensors.find("indices") == tensors.end() || tensors.find("x") == tensors.end() || tensors.find("correction_bias") == tensors.end() || tensors.find("lable_values") == tensors.end() || tensors.find("lable_indices") == tensors.end()) {
        throw std::runtime_error("Invalid Test: Missing attributes or tensors");
    }

    test->_attributes->values = tensors["values"];
    test->_attributes->indices = tensors["indices"];
    test->_attributes->x = tensors["x"];
    test->_attributes->correction_bias = tensors["correction_bias"];

    test->_attributes->routed_scaling_factor = *reinterpret_cast<float *>(attributes["routed_scaling_factor"].data());
    test->_attributes->topk = *reinterpret_cast<int *>(attributes["topk"].data());

    test->_attributes->lable_values = tensors["lable_values"];
    test->_attributes->lable_indices = tensors["lable_indices"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(infiniopHandle_t handle, infiniDevice_t device, int device_id,
                                                 size_t warm_ups, size_t iterations) {
    infiniopTopkrouterDescriptor_t op_desc;
    CHECK_OR(infiniopCreateTopkrouterDescriptor(handle, &op_desc, _attributes->x->desc(),
                                                _attributes->correction_bias->desc()),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create Topkrouter descriptor"));

    //
    auto values = _attributes->values->to(device, device_id);
    auto indices = _attributes->indices->to(device, device_id);
    auto x = _attributes->x->to(device, device_id);
    auto correction_bias = _attributes->correction_bias->to(device, device_id);

    float routed_scaling_factor = _attributes->routed_scaling_factor;
    int topk = _attributes->topk;

    size_t workspace_size;
    CHECK_OR(infiniopGetTopkrouterWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size"));
    void *workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspace_size),
                 return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace"));
    }

    CHECK_OR(infiniopTopkrouter(op_desc, workspace, workspace_size, values->data(), indices->data(), x->data(),
                                correction_bias->data(), routed_scaling_factor, topk, nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Topkrouter execution failed"));

    try {
        allClose(values, _attributes->lable_values, _rtol, _atol);
        allClose(indices, _attributes->lable_indices, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopTopkrouter(op_desc, workspace, workspace_size, values->data(), indices->data(), x->data(),
                               correction_bias->data(), routed_scaling_factor, topk, nullptr);
        },
        warm_ups, iterations);

    if (workspace != nullptr) {
        infinirtFree(workspace);
    }

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"routed_scaling_factor", "topk"};
}

std::vector<std::string> Test::tensor_names() {
    return {"values", "indices", "x", "correction_bias", "lable_values", "lable_indices"};
}

std::vector<std::string> Test::output_names() {
    return {"values", "indices"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- routed_scaling_factor=" << _attributes->routed_scaling_factor << std::endl;
    oss << "- topk=" << _attributes->topk << std::endl;

    oss << "- values: " << _attributes->values->info() << std::endl;
    oss << "- indices: " << _attributes->indices->info() << std::endl;
    oss << "- x: " << _attributes->x->info() << std::endl;
    oss << "- correction_bias: " << _attributes->correction_bias->info() << std::endl;

    oss << "- lable_values: " << _attributes->lable_values->info() << std::endl;
    oss << "- lable_indices: " << _attributes->lable_indices->info() << std::endl;

    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::topkrouter
