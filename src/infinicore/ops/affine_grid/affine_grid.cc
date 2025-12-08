#include "infinicore/ops/affine_grid.hpp"
#include <stdexcept>
#include <vector>
#include <string>

namespace infinicore::op {

// =========================================================
// Dispatcher & Execute
// =========================================================

common::OpDispatcher<AffineGrid::schema> &AffineGrid::dispatcher() {
    static common::OpDispatcher<AffineGrid::schema> dispatcher_;
    return dispatcher_;
};

void AffineGrid::execute(Tensor output, Tensor theta, bool align_corners) {
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No AffineGrid implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, theta, align_corners);
}

// =========================================================
// Wrapper Implementation
// =========================================================

Tensor affine_grid(Tensor theta, const std::vector<int64_t>& size, bool align_corners) {
    // 1. 输入 Theta 维度检查
    // Theta 必须是 (N, 2, 3)
    if (theta->ndim() != 3) {
        throw std::runtime_error("AffineGrid: Theta tensor must be 3D (N, 2, 3).");
    }
    if (theta->shape()[1] != 2 || theta->shape()[2] != 3) {
        throw std::runtime_error("AffineGrid: Theta tensor shape must be (N, 2, 3).");
    }

    // 2. 目标尺寸参数检查
    // size 通常为 (N, C, H, W)，长度必须为 4
    if (size.size() != 4) {
        throw std::runtime_error("AffineGrid: target size length must be 4 (N, C, H, W).");
    }

    // 3. 检查 Batch Size 一致性
    if (static_cast<int64_t>(theta->shape()[0]) != size[0]) {
        throw std::runtime_error("AffineGrid: Theta batch size does not match target size batch.");
    }

    // ===========================================================
    // 【关键修复】确保输入 Theta 是连续的 (Contiguous)
    // ===========================================================
    // 刚才的报错是因为测试用例传入了 stride 不标准的 Tensor。
    // CUDA Kernel 使用 flat index 访问 (ptr + n*6)，所以必须保证内存连续。
    if (!theta->is_contiguous()) {
        theta = theta->contiguous();
    }

    // 4. 推导输出形状
    // 输入 size 为 (N, C, H, W)
    // 输出 Grid 形状为 (N, H, W, 2)
    std::vector<size_t> out_shape;
    out_shape.reserve(4);
    out_shape.push_back(static_cast<size_t>(size[0])); // N
    out_shape.push_back(static_cast<size_t>(size[2])); // H
    out_shape.push_back(static_cast<size_t>(size[3])); // W
    out_shape.push_back(2);                            // (x, y) 坐标

    // 5. 创建输出 Tensor
    // 继承输入 Theta 的数据类型和设备
    auto output = Tensor::empty(out_shape, theta->dtype(), theta->device());

    // 6. 执行计算
    AffineGrid::execute(output, theta, align_corners);

    return output;
}

} // namespace infinicore::op