#ifndef __INFINIOP_BINARY_KUNLUN_H__
#define __INFINIOP_BINARY_KUNLUN_H__

#include "../../devices/kunlun/kunlun_common.h"
#include "../../devices/kunlun/kunlun_type.h"
#include <iostream>
namespace op::kunlun_common {

namespace binary_op {

void host2device(const kunlun_size_t *c_shape, const kunlun_ptrdiff_t *c_strides, const kunlun_size_t *a_shape, const kunlun_ptrdiff_t *a_strides,
                 const kunlun_size_t *b_shape, const kunlun_ptrdiff_t *b_strides,
                 kunlun_size_t *xpu_c_shape, kunlun_ptrdiff_t *xpu_c_strides, kunlun_size_t *xpu_a_shape, kunlun_ptrdiff_t *xpu_a_strides,
                 kunlun_size_t *xpu_b_shape, kunlun_ptrdiff_t *xpu_b_strides,
                 kunlun_size_t ndim);

// Perform binary computation when inputs and the output can have different dtypes
template <typename Tc, typename Ta, typename Tb, typename BinaryOp, typename... Args>
__global__ void calculate(kunlun_size_t c_data_size,
                          kunlun_size_t ndim,
                          bool contiguous,
                          bool broadcasted, Tc *c, const Ta *a, const Tb *b,
                          kunlun_size_t *xpu_c_shape, kunlun_ptrdiff_t *xpu_c_strides, kunlun_size_t *xpu_a_shape, kunlun_ptrdiff_t *xpu_a_strides,
                          kunlun_size_t *xpu_b_shape, kunlun_ptrdiff_t *xpu_b_strides,
                          Args &&...args) {

    kunlun_size_t data_size = c_data_size;
    int cid = core_id();
    int ncores = core_num();
    if (cid >= ncores) {
        return;
    }
    int thread_id = ncores * cluster_id() + cid;
    int nthreads = ncores * cluster_num();

    constexpr int buf_size = 512; // 保证所有内存加起来不超过16kB
    int task_size = buf_size * nthreads;

    __local__ Ta a_local[buf_size];
    __local__ Tb b_local[buf_size];
    __local__ Tc c_local[buf_size];

    int remain = data_size % task_size;
    int repeat = (data_size - remain) / task_size;

    int remain_task = remain % nthreads;
    int step_easy = (remain - remain_task) / nthreads;
    int step_hard = step_easy + 1;
    int step = (thread_id < remain_task ? step_hard : step_easy);
    int ind_start = repeat * task_size + (thread_id < remain_task ? thread_id * step_hard : remain_task * step_hard + (thread_id - remain_task) * step_easy);

    for (int r = 0; r < repeat + (step > 0 ? 1 : 0); r++) {
        int read_len = (r < repeat ? buf_size : step);
        int start = (r < repeat ? r * task_size + thread_id * buf_size : ind_start);
        if (contiguous) {
            GM2LM(a + start, a_local, read_len * sizeof(Ta));
            GM2LM(b + start, b_local, read_len * sizeof(Tb));

            for (int i = 0; i < read_len; i++) {
                c_local[i] = BinaryOp{}(a_local[i], b_local[i], std::forward<Args>(args)...);
            }
            mfence();

            LM2GM(c_local, c + start, read_len * sizeof(Tc));
        } else {
            for (int i = 0; i < read_len; i++) {
                int i_index = i + start;
                int a_index = broadcasted ? op::kunlun_common::indexToReducedOffset(i_index, ndim, xpu_c_strides, xpu_a_strides) : op::kunlun_common::indexToOffset(i_index, ndim, xpu_a_shape, xpu_a_strides);
                int b_index = broadcasted ? op::kunlun_common::indexToReducedOffset(i_index, ndim, xpu_c_strides, xpu_b_strides) : op::kunlun_common::indexToOffset(i_index, ndim, xpu_b_shape, xpu_b_strides);
                int c_index = op::kunlun_common::indexToOffset(i_index, ndim, xpu_c_shape, xpu_c_strides);

                GM2LM(a + a_index, a_local + i, 1 * sizeof(Ta));
                GM2LM(b + b_index, b_local + i, 1 * sizeof(Tb));
                c_local[i] = BinaryOp{}(a_local[i], b_local[i], std::forward<Args>(args)...);
                mfence();

                LM2GM(c_local + i, c + c_index, 1 * sizeof(Tc));
            }
        }
    }
}

// Perform binary computation when all inputs and the output share the same dtype
template <typename Tdata, typename BinaryOp, typename... Args>
__global__ void calculate(kunlun_size_t c_data_size,
                          kunlun_size_t ndim,
                          bool contiguous,
                          bool broadcasted, Tdata *c, const Tdata *a, const Tdata *b,
                          kunlun_size_t *xpu_c_shape, kunlun_ptrdiff_t *xpu_c_strides, kunlun_size_t *xpu_a_shape, kunlun_ptrdiff_t *xpu_a_strides,
                          kunlun_size_t *xpu_b_shape, kunlun_ptrdiff_t *xpu_b_strides,
                          Args &&...args) {

    kunlun_size_t data_size = c_data_size;

    int cid = core_id();
    int ncores = core_num();
    if (cid >= ncores) {
        return;
    }
    int thread_id = ncores * cluster_id() + cid;
    int nthreads = ncores * cluster_num();

    constexpr int buf_size = 512; // 保证所有内存加起来不超过16kB
    int task_size = buf_size * nthreads;

    __local__ Tdata a_local[buf_size];
    __local__ Tdata b_local[buf_size];
    __local__ Tdata c_local[buf_size];

    int remain = data_size % task_size;
    int repeat = (data_size - remain) / task_size;

    int remain_task = remain % nthreads;
    int step_easy = (remain - remain_task) / nthreads;
    int step_hard = step_easy + 1;
    int step = (thread_id < remain_task ? step_hard : step_easy);
    int ind_start = repeat * task_size + (thread_id < remain_task ? thread_id * step_hard : remain_task * step_hard + (thread_id - remain_task) * step_easy);

    for (int r = 0; r < repeat + (step > 0 ? 1 : 0); r++) {
        int read_len = (r < repeat ? buf_size : step);
        int start = (r < repeat ? r * task_size + thread_id * buf_size : ind_start);
        if (contiguous) {
            GM2LM(a + start, a_local, read_len * sizeof(Tdata));
            GM2LM(b + start, b_local, read_len * sizeof(Tdata));

            for (int i = 0; i < read_len; i++) {

                c_local[i] = BinaryOp{}(a_local[i], b_local[i], std::forward<Args>(args)...);
            }
            mfence();

            LM2GM(c_local, c + start, read_len * sizeof(Tdata));
        } else {
            for (int i = 0; i < read_len; i++) {
                int i_index = i + start;
                int a_index = broadcasted ? op::kunlun_common::indexToReducedOffset(i_index, ndim, xpu_c_strides, xpu_a_strides) : op::kunlun_common::indexToOffset(i_index, ndim, xpu_a_shape, xpu_a_strides);
                int b_index = broadcasted ? op::kunlun_common::indexToReducedOffset(i_index, ndim, xpu_c_strides, xpu_b_strides) : op::kunlun_common::indexToOffset(i_index, ndim, xpu_b_shape, xpu_b_strides);
                int c_index = op::kunlun_common::indexToOffset(i_index, ndim, xpu_c_shape, xpu_c_strides);

                GM2LM(a + a_index, a_local + i, 1 * sizeof(Tdata));
                GM2LM(b + b_index, b_local + i, 1 * sizeof(Tdata));
                c_local[i] = BinaryOp{}(a_local[i], b_local[i], std::forward<Args>(args)...);
                mfence();
                LM2GM(c_local + i, c + c_index, 1 * sizeof(Tdata));
            }
        }
    }
}
template <typename Tdata, typename BinaryOp, typename... Args>
void launch_calculate(kunlun_size_t c_data_size,
                      kunlun_size_t ndim,
                      bool contiguous,
                      bool broadcasted, const kunlun_size_t *c_shape, const kunlun_ptrdiff_t *c_strides, const kunlun_size_t *a_shape, const kunlun_ptrdiff_t *a_strides,
                      const kunlun_size_t *b_shape, const kunlun_ptrdiff_t *b_strides, Tdata *c, const Tdata *a, const Tdata *b, XPUStream stream,
                      Args... args) {

    char *workspace;
    int ret = 0;
    ret = xpu_malloc((void **)&workspace, ndim * (3 * sizeof(kunlun_size_t) + 3 * sizeof(long)));
    assert(ret == 0);
    char *tmp_strides = workspace + 3 * ndim * sizeof(kunlun_size_t);
    kunlun_size_t *xpu_c_shape = (kunlun_size_t *)workspace;
    kunlun_size_t *xpu_a_shape = xpu_c_shape + ndim;
    kunlun_size_t *xpu_b_shape = xpu_a_shape + ndim;
    kunlun_ptrdiff_t *xpu_c_strides = (kunlun_ptrdiff_t *)tmp_strides;
    kunlun_ptrdiff_t *xpu_a_strides = xpu_c_strides + ndim;
    kunlun_ptrdiff_t *xpu_b_strides = xpu_a_strides + ndim;

    host2device(c_shape, c_strides, a_shape, a_strides,
                b_shape, b_strides, xpu_c_shape, xpu_c_strides, xpu_a_shape, xpu_a_strides,
                xpu_b_shape, xpu_b_strides, ndim);

    calculate<Tdata, BinaryOp><<<8, 64, stream>>>(c_data_size,
                                                  ndim,
                                                  contiguous,
                                                  broadcasted, c, a, b,
                                                  xpu_c_shape, xpu_c_strides,
                                                  xpu_a_shape, xpu_a_strides,
                                                  xpu_b_shape, xpu_b_strides,
                                                  std::forward<Args>(args)...);
    xpu_wait();
    xpu_free(workspace);
}

template <typename Tc, typename Ta, typename Tb, typename BinaryOp, typename... Args>
void launch_calculate(kunlun_size_t c_data_size,
                      kunlun_size_t ndim,
                      bool contiguous,
                      bool broadcasted, const kunlun_size_t *c_shape, const kunlun_ptrdiff_t *c_strides, const kunlun_size_t *a_shape, const kunlun_ptrdiff_t *a_strides,
                      const kunlun_size_t *b_shape, const kunlun_ptrdiff_t *b_strides, Tc *c, const Ta *a, const Tb *b, XPUStream stream,
                      Args... args) {

    char *workspace;
    int ret = 0;
    ret = xpu_malloc((void **)&workspace, ndim * 3 * (sizeof(kunlun_size_t) + sizeof(kunlun_ptrdiff_t)));
    assert(ret == 0);
    char *tmp_strides = workspace + 3 * ndim * sizeof(kunlun_size_t);
    kunlun_size_t *xpu_c_shape = (kunlun_size_t *)workspace;
    kunlun_size_t *xpu_a_shape = xpu_c_shape + ndim;
    kunlun_size_t *xpu_b_shape = xpu_a_shape + ndim;
    kunlun_ptrdiff_t *xpu_c_strides = (kunlun_ptrdiff_t *)tmp_strides;
    kunlun_ptrdiff_t *xpu_a_strides = xpu_c_strides + ndim;
    kunlun_ptrdiff_t *xpu_b_strides = xpu_a_strides + ndim;
    host2device(c_shape, c_strides, a_shape, a_strides,
                b_shape, b_strides, xpu_c_shape, xpu_c_strides, xpu_a_shape, xpu_a_strides,
                xpu_b_shape, xpu_b_strides, ndim);
    calculate<Tc, Ta, Tb, BinaryOp><<<8, 64, stream>>>(c_data_size,
                                                       ndim,
                                                       contiguous,
                                                       broadcasted, c, a, b,
                                                       xpu_c_shape, xpu_c_strides,
                                                       xpu_a_shape, xpu_a_strides,
                                                       xpu_b_shape, xpu_b_strides,
                                                       std::forward<Args>(args)...);
    xpu_wait();
    xpu_free(workspace);
}

} // namespace binary_op
} // namespace op::kunlun_common

#endif // __INFINIOP_BINARY_KUNLUN_H__
