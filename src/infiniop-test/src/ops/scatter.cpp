#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::scatter {
struct Test::Attributes {
    std::shared_ptr<Tensor> output;
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> index;
    size_t dim;
    std::shared_ptr<Tensor> ans;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    if (tensors.find("output") == tensors.end()
        || tensors.find("input") == tensors.end()
        || tensors.find("index") == tensors.end()
        || tensors.find("ans") == tensors.end()) {
        throw std::runtime_error("Invalid Test");
    }
    test->_attributes->output = tensors["output"];
    test->_attributes->input = tensors["input"];
    test->_attributes->index = tensors["index"];
    test->_attributes->ans = tensors["ans"];
    test->_attributes->dim = *reinterpret_cast<size_t *>(attributes["dim"].data());

    return test;
}
std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    infiniopScatterDescriptor_t op_desc;
    auto output = _attributes->output->to(device, device_id);
    auto input = _attributes->input->to(device, device_id);
    auto index = _attributes->index->to(device, device_id);
    auto dim = _attributes->dim;
    CHECK_OR(infiniopCreateScatterDescriptor(handle, &op_desc,
            output->desc(),
            input->desc(),
            index->desc(),
            dim
        ),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create op descriptor."));
    size_t workspace_size;
    CHECK_OR(infiniopGetScatterWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));
    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    CHECK_OR(infiniopScatter(op_desc, workspace, workspace_size,
                    output->data(),
                    input->data(),
                    index->data(),
                    nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    try {
        allClose(output, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopScatter(
                op_desc, workspace, workspace_size,
                output->data(),
                input->data(),
                index->data(),
                nullptr);
        },
        warm_ups, iterations);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"dim"};
}

std::vector<std::string> Test::tensor_names() {
    return {"output", "input", "index", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"output"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- output: " << _attributes->output->info() << std::endl;
    oss << "- input: " << _attributes->input->info() << std::endl;
    oss << "- index: " << _attributes->index->info() << std::endl;
    oss << "- dim: " << _attributes->dim << std::endl;

    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::scatter
