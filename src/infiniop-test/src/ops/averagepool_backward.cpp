// averagepool_backward.cpp
#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::averagepool_backward {

struct Test::Attributes {
    // 张量
    std::shared_ptr<Tensor> input;               // 前向输入 X
    std::shared_ptr<Tensor> grad_output;         // 上游梯度 dY
    std::shared_ptr<Tensor> expected_grad_input; // 期望梯度 dX

    // 平均池化参数
    std::vector<size_t> kernel_size;
    std::vector<size_t> stride;
    std::vector<size_t> padding;
    bool ceil_mode;
};

static void broadcast_or_fill(std::vector<size_t> &dst,
                              const int *src, size_t src_cnt,
                              size_t ndim) {
    dst.clear();
    if (src_cnt == ndim) {
        for (size_t i = 0; i < ndim; ++i) {
            dst.push_back(static_cast<size_t>(src[i]));
        }
    } else {
        // 将单个值广播到所有池化维度
        dst.assign(ndim, static_cast<size_t>(src[0]));
    }
}

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {

    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();

    if (!check_names(attributes, Test::attribute_names()) || !check_names(tensors, Test::tensor_names())) {
        throw std::runtime_error("Invalid Test: missing attributes or tensors");
    }

    test->_attributes->input = tensors["input"];
    test->_attributes->grad_output = tensors["grad_output"];
    test->_attributes->expected_grad_input = tensors["grad_input"];

    // 维度：去掉 N、C 后的空间维度数
    const auto &in_shape = test->_attributes->input->shape();
    if (in_shape.size() < 3) {
        throw std::runtime_error("Input tensor rank must be >= 3 (N, C, ...)");
    }
    size_t pool_ndim = in_shape.size() - 2;

    // --- kernel_size ---
    {
        const auto &buf = attributes["kernel_size"];
        if (buf.size() % sizeof(int) != 0) {
            throw std::runtime_error("Invalid kernel_size data size");
        }
        size_t cnt = buf.size() / sizeof(int);
        const int *p = reinterpret_cast<const int *>(buf.data());
        broadcast_or_fill(test->_attributes->kernel_size, p, cnt, pool_ndim);
    }

    // --- stride ---
    {
        const auto &buf = attributes["stride"];
        if (buf.size() % sizeof(int) != 0) {
            throw std::runtime_error("Invalid stride data size");
        }
        size_t cnt = buf.size() / sizeof(int);
        const int *p = reinterpret_cast<const int *>(buf.data());
        broadcast_or_fill(test->_attributes->stride, p, cnt, pool_ndim);
    }

    // --- padding ---
    {
        const auto &buf = attributes["padding"];
        if (buf.size() % sizeof(int) != 0) {
            throw std::runtime_error("Invalid padding data size");
        }
        size_t cnt = buf.size() / sizeof(int);
        const int *p = reinterpret_cast<const int *>(buf.data());
        broadcast_or_fill(test->_attributes->padding, p, cnt, pool_ndim);
    }

    // --- ceil_mode ---
    {
        const auto &buf = attributes["ceil_mode"];
        if (buf.size() == sizeof(bool)) {
            test->_attributes->ceil_mode = *reinterpret_cast<const bool *>(buf.data());
        } else if (buf.size() == sizeof(uint8_t)) {
            test->_attributes->ceil_mode = (*reinterpret_cast<const uint8_t *>(buf.data()) != 0);
        } else {
            throw std::runtime_error("Invalid ceil_mode data size");
        }
    }

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations) {

    // 把张量放到目标设备
    auto input = _attributes->input->to(device, device_id);             // X
    auto grad_output = _attributes->grad_output->to(device, device_id); // dY
    auto expected_grad_input = _attributes->expected_grad_input;        // 参考 dX

    // 构造实际输出 dX 的张量（形状等于 input，dtype 等于 input）
    const auto &in_shape = input->shape();
    std::vector<ptrdiff_t> in_strides(in_shape.size());
    if (!in_shape.empty()) {
        in_strides.back() = 1;
        for (int i = static_cast<int>(in_shape.size()) - 2; i >= 0; --i) {
            in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
        }
    }
    size_t dx_bytes = ggmlTypeSize(input->ggml_type());
    for (auto d : in_shape) {
        dx_bytes *= d;
    }

    auto dx_mem = std::make_shared<Memory>(dx_bytes, device, device_id);
    auto actual_grad_input = std::make_shared<Tensor>(
        dx_mem, 0, in_shape, in_strides, input->ggml_type());

    // 参数指针
    void *kernel_size_ptr = _attributes->kernel_size.data();
    void *stride_ptr = _attributes->stride.data();
    void *padding_ptr = _attributes->padding.data();

    // --- 创建反向算子描述符 ---
    infiniopAvgPoolBackwardDescriptor_t bwd_desc;
    CHECK_OR(infiniopCreateAvgPoolBackwardDescriptor(
                 handle, &bwd_desc,
                 actual_grad_input->desc(), // grad_input_desc (dX)
                 grad_output->desc(),       // grad_output_desc (dY)
                 input->desc(),             // input_desc (X)
                 kernel_size_ptr,
                 stride_ptr,
                 padding_ptr,
                 _attributes->ceil_mode),
             return TEST_FAILED(OP_CREATION_FAILED,
                                "Failed to create averagepool backward descriptor."));

    // --- 获取工作空间大小 ---
    size_t workspace_size = 0;
    CHECK_OR(infiniopGetAvgPoolBackwardWorkspaceSize(bwd_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED,
                                "Failed to get backward workspace size."));

    void *workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspace_size),
                 return TEST_FAILED(OP_CREATION_FAILED,
                                    "Failed to allocate backward workspace."));
    }

    // --- 执行反向：dX = AvgPoolBackward(dY, X, ...) ---
    CHECK_OR(infiniopAvgPoolBackward(
                 bwd_desc, workspace, workspace_size,
                 actual_grad_input->data(), // dX
                 grad_output->data(),       // dY
                 input->data(),             // X
                 /*stream*/ nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED,
                                "Failed during averagepool backward execution."));

    // --- 校验数值 ---
    try {
        allClose(actual_grad_input, expected_grad_input, _rtol, _atol);
    } catch (const std::exception &e) {
        if (workspace) {
            infinirtFree(workspace);
        }
        infiniopDestroyAvgPoolBackwardDescriptor(bwd_desc);
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    // --- 基准测试 ---
    double elapsed_time = benchmark(
        [=]() {
            infiniopAvgPoolBackward(
                bwd_desc, workspace, workspace_size,
                actual_grad_input->data(),
                grad_output->data(),
                input->data(),
                nullptr);
        },
        warm_ups, iterations);

    // --- 清理 ---
    if (workspace) {
        infinirtFree(workspace);
    }
    infiniopDestroyAvgPoolBackwardDescriptor(bwd_desc);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"kernel_size", "stride", "padding", "ceil_mode"};
}

std::vector<std::string> Test::tensor_names() {
    // 需要的输入张量
    return {"input", "grad_output", "grad_input"};
}

std::vector<std::string> Test::output_names() {
    // 无额外导出
    return {};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << "\n";
    oss << "- input: " << _attributes->input->info() << "\n";
    oss << "- grad_output (dY): " << _attributes->grad_output->info() << "\n";
    oss << "- expected_grad_input: " << _attributes->expected_grad_input->info() << "\n";

    auto dump = [&](const char *name, const std::vector<size_t> &v) {
        oss << "- " << name << ": [";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) {
                oss << ", ";
            }
            oss << v[i];
        }
        oss << "]\n";
    };
    dump("kernel_size", _attributes->kernel_size);
    dump("stride", _attributes->stride);
    dump("padding", _attributes->padding);

    oss << "- ceil_mode: " << (_attributes->ceil_mode ? "true" : "false") << "\n";
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << "\n";
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::averagepool_backward
