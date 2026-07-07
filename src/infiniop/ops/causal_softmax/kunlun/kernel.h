#ifndef __CAUSAL_SOFTMAX_KUNLUN_KERNEL_H__
#define __CAUSAL_SOFTMAX_KUNLUN_KERNEL_H__

#include "../../../devices/kunlun/kunlun_kernel_common.h"
#include "../../../reduce/kunlun/reduce_kunlun.h"

using namespace device::kunlun::kernel;

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ void causalSoftmaxBlock(
    __shared_ptr__ Tdata *y,
    __shared_ptr__ const Tdata *x,
    size_t height,
    size_t width,
    int row_id) {

    // Reduce max for each row and store in shared memory
    __shared__ Tdata max_;
    Tdata max_0 = op::common_kunlun::reduce_op::max<BLOCK_SIZE, Tdata>(x, width - height + 1 + size_t(row_id));
    if (core_id() == 0) {
        storeShared(&max_, max_0);
    }
    sync_cluster();

    Tdata max_val = loadShared(&max_);

    // Elemetwise sub max for each element and apply causal softmax
    for (size_t col = core_id(); col < width; col += BLOCK_SIZE) {
        //   row_id ↓ |<-     width   ->|
        //          0 | * * * ... *     |
        //          1 | * * * ... * *   |
        //          2 | * * * ... * * * |
        //  height: 3  col_id->
        if (width + size_t(row_id) >= col + height) {
            Tdata x_val = loadShared(x + col);
            Tdata y_val;
            if constexpr (std::is_same_v<Tdata, half>) {
                y_val = hexp(x_val - max_val);
            } else if constexpr (std::is_same_v<Tdata, bfloat16_t>) {
                y_val = __float2bfloat16(exp(__bfloat162float(x_val) - __bfloat162float(max_val)));
            } else {
                y_val = exp(x_val - max_val);
            }
            storeShared(y + col, y_val);
        } else {
            storeShared(y + col, Tdata(0));
        }
    }
    sync_cluster();

    // Reduce sum for each row
    __shared__ Tcompute sum_;
    Tcompute sum_0 = op::common_kunlun::reduce_op::sum<BLOCK_SIZE, Tdata, Tcompute>(y, width);
    if (core_id() == 0) {
        storeShared(&sum_, sum_0);
    }
    sync_cluster();

    Tcompute sum_val = loadShared(&sum_);

    // Apply softmax
    for (size_t col = core_id(); col < width; col += BLOCK_SIZE) {
        if (sum_val != 0) {
            Tdata y_val = loadShared(y + col);
            storeShared(y + col, Tdata(Tcompute(y_val) / sum_val));
        } else {
            storeShared(y + col, Tdata(0));
        }
    }
    sync_cluster();
}

#endif
