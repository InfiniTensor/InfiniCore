#include "ops.hpp"
#include "utils.hpp"
#include <cstdio>
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::conv_backward {

struct Test::Attributes {
    std::shared_ptr<Tensor> grad_output;
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;

    std::shared_ptr<Tensor> expected_grad_input;
    std::shared_ptr<Tensor> expected_grad_weight;
    std::shared_ptr<Tensor> expected_grad_bias;

    std::vector<int> stride;
    std::vector<int> padding;
    std::vector<int> dilation;
    int groups;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {

    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();

    if (!check_names(attributes, Test::attribute_names()) || !check_names(tensors, Test::tensor_names())) {
        throw std::runtime_error("Invalid Test");
    }

    test->_attributes->grad_output = tensors["grad_output"];
    test->_attributes->input = tensors["input"];
    test->_attributes->weight = tensors["weight"];

    auto bias_it = tensors.find("bias");
    if (bias_it != tensors.end()) {
        test->_attributes->bias = bias_it->second;
    }

    test->_attributes->expected_grad_input = tensors["grad_input"];
    test->_attributes->expected_grad_weight = tensors["grad_weight"];

    auto grad_bias_it = tensors.find("grad_bias");
    if (grad_bias_it != tensors.end()) {
        test->_attributes->expected_grad_bias = grad_bias_it->second;
    }

    size_t pool_ndim = test->_attributes->input->shape().size() - 2;
    if (pool_ndim == 0) {
        throw std::runtime_error("Input tensor must have at least 3 dimensions (N, C, ...)");
    }

    auto stride_data = attributes["stride"];
    auto padding_data = attributes["padding"];
    auto dilation_data = attributes["dilation"];

    size_t stride_count = stride_data.size() / sizeof(int);
    size_t padding_count = padding_data.size() / sizeof(int);
    size_t dilation_count = dilation_data.size() / sizeof(int);

    if (stride_data.size() % sizeof(int) != 0) {
        throw std::runtime_error("Invalid stride data size");
    }
    const int *stride_ptr = reinterpret_cast<const int *>(stride_data.data());
    if (stride_count == pool_ndim) {
        test->_attributes->stride.clear();
        for (size_t i = 0; i < stride_count; i++) {
            test->_attributes->stride.push_back(static_cast<size_t>(stride_ptr[i]));
        }
    } else {
        test->_attributes->stride.assign(pool_ndim, static_cast<size_t>(stride_ptr[0]));
    }

    if (padding_data.size() % sizeof(int) != 0) {
        throw std::runtime_error("Invalid padding data size");
    }
    const int *padding_ptr = reinterpret_cast<const int *>(padding_data.data());
    if (padding_count == pool_ndim) {
        test->_attributes->padding.clear();
        for (size_t i = 0; i < padding_count; i++) {
            test->_attributes->padding.push_back(static_cast<size_t>(padding_ptr[i]));
        }
    } else {
        test->_attributes->padding.assign(pool_ndim, static_cast<size_t>(padding_ptr[0]));
    }

    if (dilation_data.size() % sizeof(int) != 0) {
        throw std::runtime_error("Invalid dilation data size");
    }
    const int *dilation_ptr = reinterpret_cast<const int *>(dilation_data.data());
    if (dilation_count == pool_ndim) {
        test->_attributes->dilation.clear();
        for (size_t i = 0; i < dilation_count; i++) {
            test->_attributes->dilation.push_back(static_cast<size_t>(dilation_ptr[i]));
        }
    } else {
        test->_attributes->dilation.assign(pool_ndim, static_cast<size_t>(dilation_ptr[0]));
    }

    test->_attributes->groups = *reinterpret_cast<int *>(attributes["groups"].data());

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations) {
    infiniopConvBackwardDescriptor_t op_desc;

    auto grad_output = _attributes->grad_output->to(device, device_id);
    auto input = _attributes->input->to(device, device_id);
    auto weight = _attributes->weight->to(device, device_id);
    auto bias = _attributes->bias ? _attributes->bias->to(device, device_id) : nullptr;

    auto expected_grad_input = _attributes->expected_grad_input;
    auto expected_grad_weight = _attributes->expected_grad_weight;
    auto expected_grad_bias = _attributes->expected_grad_bias;

    auto input_dtype = input->ggml_type();

    auto grad_input_shape = expected_grad_input->shape();
    size_t grad_input_size = 1;
    for (auto dim : grad_input_shape) {
        grad_input_size *= dim;
    }
    grad_input_size *= ggmlTypeSize(input_dtype);

    auto grad_input_memory = std::make_shared<Memory>(grad_input_size, device, device_id);
    std::vector<ptrdiff_t> grad_input_strides(grad_input_shape.size());

    if (grad_input_shape.size() > 0) {
        grad_input_strides[grad_input_shape.size() - 1] = 1;
        for (int i = static_cast<int>(grad_input_shape.size()) - 2; i >= 0; i--) {
            grad_input_strides[i] = grad_input_strides[i + 1] * grad_input_shape[i + 1];
        }
    }

    auto actual_grad_input = std::make_shared<Tensor>(
        grad_input_memory, 0, grad_input_shape, grad_input_strides, input_dtype);

    auto grad_weight_shape = expected_grad_weight->shape();
    size_t grad_weight_size = 1;
    for (auto dim : grad_weight_shape) {
        grad_weight_size *= dim;
    }
    grad_weight_size *= ggmlTypeSize(input_dtype);

    auto grad_weight_memory = std::make_shared<Memory>(grad_weight_size, device, device_id);
    std::vector<ptrdiff_t> grad_weight_strides(grad_weight_shape.size());

    if (grad_weight_shape.size() > 0) {
        grad_weight_strides[grad_weight_shape.size() - 1] = 1;
        for (int i = static_cast<int>(grad_weight_shape.size()) - 2; i >= 0; i--) {
            grad_weight_strides[i] = grad_weight_strides[i + 1] * grad_weight_shape[i + 1];
        }
    }

    auto actual_grad_weight = std::make_shared<Tensor>(
        grad_weight_memory, 0, grad_weight_shape, grad_weight_strides, input_dtype);

    std::shared_ptr<Tensor> actual_grad_bias = nullptr;
    if (bias && expected_grad_bias) {
        auto grad_bias_shape = expected_grad_bias->shape();
        size_t grad_bias_size = 1;
        for (auto dim : grad_bias_shape) {
            grad_bias_size *= dim;
        }
        grad_bias_size *= ggmlTypeSize(input_dtype);

        auto grad_bias_memory = std::make_shared<Memory>(grad_bias_size, device, device_id);
        std::vector<ptrdiff_t> grad_bias_strides(grad_bias_shape.size());

        if (grad_bias_shape.size() > 0) {
            grad_bias_strides[grad_bias_shape.size() - 1] = 1;
            for (int i = static_cast<int>(grad_bias_shape.size()) - 2; i >= 0; i--) {
                grad_bias_strides[i] = grad_bias_strides[i + 1] * grad_bias_shape[i + 1];
            }
        }

        actual_grad_bias = std::make_shared<Tensor>(
            grad_bias_memory, 0, grad_bias_shape, grad_bias_strides, input_dtype);
    }

    void *pads_ptr = _attributes->padding.data();
    void *strides_ptr = _attributes->stride.data();
    void *dilations_ptr = _attributes->dilation.data();

    CHECK_OR(infiniopCreateConvBackwardDescriptor(
                 handle, &op_desc,
                 grad_output->desc(),
                 input->desc(),
                 weight->desc(),
                 bias ? bias->desc() : nullptr,
                 pads_ptr,
                 strides_ptr,
                 dilations_ptr,
                 _attributes->groups),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create conv_backward descriptor."));

    // 获取工作空间大小
    size_t workspace_size;
    CHECK_OR(infiniopGetConvBackwardWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));

    // 分配工作空间
    void *workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspace_size),
                 return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    }

    CHECK_OR(infiniopConvBackward(
                 op_desc, workspace, workspace_size,
                 actual_grad_input->data(),                             // void *grad_input
                 actual_grad_weight->data(),                            // void *grad_weight
                 actual_grad_bias ? actual_grad_bias->data() : nullptr, // void *grad_bias
                 grad_output->data(),                                   // const void *grad_output
                 input->data(),                                         // const void *input
                 weight->data(),                                        // const void *weight
                 nullptr),                                              // void *stream
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during conv_backward execution."));

    // 验证结果
    try {
        allClose(actual_grad_input, expected_grad_input, _rtol, _atol);
        allClose(actual_grad_weight, expected_grad_weight, _rtol, _atol);

        if (actual_grad_bias && expected_grad_bias) {
            allClose(actual_grad_bias, expected_grad_bias, _rtol, _atol);
        }
    } catch (const std::exception &e) {
        if (workspace) {
            infinirtFree(workspace);
        }
        infiniopDestroyConvBackwardDescriptor(op_desc);
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    // 性能测试
    double elapsed_time = benchmark(
        [=]() {
            infiniopConvBackward(
                op_desc, workspace, workspace_size,
                actual_grad_input->data(),
                actual_grad_weight->data(),
                actual_grad_bias ? actual_grad_bias->data() : nullptr,
                grad_output->data(),
                input->data(),
                weight->data(),
                nullptr);
        },
        warm_ups, iterations);

    // 清理资源
    if (workspace) {
        infinirtFree(workspace);
    }
    infiniopDestroyConvBackwardDescriptor(op_desc);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"stride", "padding", "dilation", "groups"};
}

std::vector<std::string> Test::tensor_names() {
    return {"grad_output", "input", "weight", "bias", "grad_input", "grad_weight", "grad_bias"};
}

std::vector<std::string> Test::output_names() {
    return {};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- grad_output: " << _attributes->grad_output->info() << std::endl;
    oss << "- input: " << _attributes->input->info() << std::endl;
    oss << "- weight: " << _attributes->weight->info() << std::endl;
    if (_attributes->bias) {
        oss << "- bias: " << _attributes->bias->info() << std::endl;
    }
    oss << "- expected_grad_input: " << _attributes->expected_grad_input->info() << std::endl;
    oss << "- expected_grad_weight: " << _attributes->expected_grad_weight->info() << std::endl;
    if (_attributes->expected_grad_bias) {
        oss << "- expected_grad_bias: " << _attributes->expected_grad_bias->info() << std::endl;
    }

    oss << "- stride: [";
    for (size_t i = 0; i < _attributes->stride.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << _attributes->stride[i];
    }
    oss << "]" << std::endl;

    oss << "- padding: [";
    for (size_t i = 0; i < _attributes->padding.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << _attributes->padding[i];
    }
    oss << "]" << std::endl;

    oss << "- dilation: [";
    for (size_t i = 0; i < _attributes->dilation.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << _attributes->dilation[i];
    }
    oss << "]" << std::endl;

    oss << "- groups: " << _attributes->groups << std::endl;

    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::conv_backward
