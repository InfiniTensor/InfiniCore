template <class Tp, class Ta>
static __device__ void padding(
    Ta *__restrict__ y_,
    int const stride_token_y,
    int const stride_head_y,
    Ta const *__restrict__ x_,
    int const stride_token_x,
    int const stride_head_x,
    Tp const *__restrict__ pos_,
    float const *__restrict__ sin_table,
    float const *__restrict__ cos_table,
    Tp const *__restrict__ rope_section_
) {

    // n = gridDim.y
    // nh_h = gridDim.x
    int nh_l = blockDim.y,
        dh_div_2 = blockDim.x,
        it = blockIdx.y,
        ih_h = blockIdx.x,
        ih_l = threadIdx.y,
        ih = ih_h * nh_l + ih_l,
        i = threadIdx.x;

    // 计算 x 和 y 的位置, 每相距 d_div_2 的两个为一组
    auto x1 = x_ + it * stride_token_x + ih * stride_head_x + i;
    auto x2 = x_ + it * stride_token_x + ih * stride_head_x + i + dh_div_2;
    auto y1 = y_ + it * stride_token_y + ih * stride_head_y + i;
    auto y2 = y_ + it * stride_token_y + ih * stride_head_y + i + dh_div_2;

    // 寻找 i 在 rope_section 中的位置
    int thw = 0;
    for (int j = 0; j < 3; j++) {
        if (i < rope_section_[j]) {
            thw = j;
            break;
        }
    }

    // 获取位置索引
    auto pos = pos_[it * 3 + thw]; // 3 维 pos 的 shape: [n, 3], strides: [3, 1]
    float sin = sin_table[pos * dh_div_2 + i],
          cos = cos_table[pos * dh_div_2 + i],
          a = x1[0],
          b = x2[0];

    // 应用旋转并写入 y
    y1[0] = Ta(a * cos - b * sin);
    y2[0] = Ta(a * sin + b * cos);
}
