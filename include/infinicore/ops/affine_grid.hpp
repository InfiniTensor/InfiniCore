#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <vector>

namespace infinicore::op {

class AffineGrid {
public:
    // Schema: execute(Output, Theta, AlignCorners)
    // Output: 生成的网格 (N, H, W, 2)
    // Input: 仿射矩阵 Theta (N, 2, 3)
    // AlignCorners: 对齐方式标志
    using schema = void (*)(Tensor, Tensor, bool);
    static void execute(Tensor output, Tensor theta, bool align_corners);
    static common::OpDispatcher<schema> &dispatcher();
};

// 函数接口
// theta: 仿射变换矩阵，形状通常为 (N, 2, 3) 用于 2D
// size: 目标输出张量的形状 (N, C, H, W)，用于确定网格的维度
// align_corners: 是否将像素视为网格点的中心 (通常默认为 false)
Tensor affine_grid(Tensor theta, const std::vector<int64_t>& size, bool align_corners = false);

} // namespace infinicore::op