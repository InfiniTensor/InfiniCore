#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::paged_attention {

// The Test class for the paged_attention operator.
struct Test::Attributes {
    // Paged attention uses tensors for most parameters, but scale is a scalar.
    std::shared_ptr<Tensor> scale;

    // Tensors for the operation.
    std::shared_ptr<Tensor> q;
    std::shared_ptr<Tensor> k_cache;
    std::shared_ptr<Tensor> v_cache;
    std::shared_ptr<Tensor> block_tables;
    std::shared_ptr<Tensor> seq_lens;
    std::shared_ptr<Tensor> alibi_slopes; // Can be null
    std::shared_ptr<Tensor> ans;
    std::shared_ptr<Tensor> out;
};

// Factory method to build a Test object from GGUF data.
std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    if (!check_names(tensors, Test::tensor_names())) {
        throw std::runtime_error("Invalid Test: Missing tensors.");
    }

    test->_attributes->scale = tensors["scale"];
    test->_attributes->q = tensors["q"];
    test->_attributes->k_cache = tensors["k_cache"];
    test->_attributes->v_cache = tensors["v_cache"];
    test->_attributes->block_tables = tensors["block_tables"];
    test->_attributes->seq_lens = tensors["seq_lens"];
    if (tensors.count("alibi_slopes")) {
        test->_attributes->alibi_slopes = tensors["alibi_slopes"];
    } else {
        test->_attributes->alibi_slopes = nullptr;
    }
    test->_attributes->ans = tensors["ans"];
    test->_attributes->out = tensors["out"];

    return test;
}

// Executes the test case.
std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {

    infiniopPagedAttentionDescriptor_t op_desc = nullptr;
    void *workspace = nullptr;

    auto scale_tensor = _attributes->scale->to(device, device_id);
    auto q = _attributes->q->to(device, device_id);
    auto k_cache = _attributes->k_cache->to(device, device_id);
    auto v_cache = _attributes->v_cache->to(device, device_id);
    auto block_tables = _attributes->block_tables->to(device, device_id);
    auto seq_lens = _attributes->seq_lens->to(device, device_id);
    auto out = _attributes->out->to(device, device_id);
    std::shared_ptr<Tensor> alibi_slopes = nullptr;
    if (_attributes->alibi_slopes) {
        alibi_slopes = _attributes->alibi_slopes->to(device, device_id);
    }

    float scale_val = *reinterpret_cast<float *>(scale_tensor->data());

    // Create operator descriptor
    CHECK_OR(infiniopCreatePagedAttentionDescriptor(
                 handle, &op_desc, out->desc(), q->desc(), k_cache->desc(), v_cache->desc(),
                 block_tables->desc(), seq_lens->desc(),
                 alibi_slopes ? alibi_slopes->desc() : nullptr, scale_val),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create op descriptor."));

    // Get workspace size and allocate memory
    size_t workspace_size;
    CHECK_OR(infiniopGetPagedAttentionWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));
    if (workspace_size > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspace_size),
                 return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    }

    // Execute the operator for the first time
    CHECK_OR(infiniopPagedAttention(op_desc, workspace, workspace_size,
                                    out->data(), q->data(), k_cache->data(), v_cache->data(),
                                    block_tables->data(), seq_lens->data(),
                                    alibi_slopes ? alibi_slopes->data() : nullptr, nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    // Verify the result
    try {
        allClose(out, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    // Benchmark the operation
    double elapsed_time = 0.;
    elapsed_time = benchmark(
        [=]() { // Use reference capture to ensure local variables are available
            infiniopPagedAttention(op_desc, workspace, workspace_size,
                                   out->data(), q->data(), k_cache->data(), v_cache->data(),
                                   block_tables->data(), seq_lens->data(),
                                   alibi_slopes ? alibi_slopes->data() : nullptr, nullptr);
        },
        warm_ups, iterations);

    if (op_desc) {
        infiniopDestroyPagedAttentionDescriptor(op_desc);
    }
    if (workspace) {
        infinirtFree(workspace);
    }
    return TEST_PASSED(elapsed_time);
}

// Define expected attribute and tensor names for validation.
std::vector<std::string> Test::attribute_names() { return {}; }
std::vector<std::string> Test::tensor_names() {
    return {"scale", "q", "k_cache", "v_cache", "block_tables", "seq_lens", "ans", "out"};
}
std::vector<std::string> Test::output_names() { return {"out"}; }

// MODIFIED: Added a toString() method for better debugging and logging, mimicking the reference file.
std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- q: " << _attributes->q->info() << std::endl;
    oss << "- k_cache: " << _attributes->k_cache->info() << std::endl;
    oss << "- v_cache: " << _attributes->v_cache->info() << std::endl;
    oss << "- block_tables: " << _attributes->block_tables->info() << std::endl;
    oss << "- seq_lens: " << _attributes->seq_lens->info() << std::endl;
    if (_attributes->alibi_slopes) {
        oss << "- alibi_slopes: " << _attributes->alibi_slopes->info() << std::endl;
    }
    oss << "- out: " << _attributes->out->info() << std::endl;
    oss << "- ans: " << _attributes->ans->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    if (_attributes) {
        delete _attributes;
    }
}

} // namespace infiniop_test::paged_attention
