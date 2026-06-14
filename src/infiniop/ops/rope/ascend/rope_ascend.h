#ifndef __ACLNN_ROPE_H__
#define __ACLNN_ROPE_H__

#include "../rope.h"

extern "C" infiniStatus_t rope_kernel_launch(
    void *y,
    void *x,
    void *pos,
    void *sin,
    void *cos,
    size_t seq_len,
    size_t nhead,
    size_t dhead,
    size_t batch,
    infiniDtype_t data_type,
    infiniDtype_t pos_type,
    ptrdiff_t y_stride_seqlen,
    ptrdiff_t y_stride_nhead,
    ptrdiff_t y_stride_batch,
    ptrdiff_t x_stride_seqlen,
    ptrdiff_t x_stride_nhead,
    ptrdiff_t x_stride_batch,
    void *stream);

extern "C" infiniStatus_t rope_kernel_neox_launch(
    void *y,
    void *x,
    void *pos,
    void *sin,
    void *cos,
    size_t seq_len,
    size_t nhead,
    size_t dhead,
    size_t batch,
    infiniDtype_t data_type,
    infiniDtype_t pos_type,
    ptrdiff_t y_stride_seqlen,
    ptrdiff_t y_stride_nhead,
    ptrdiff_t y_stride_batch,
    ptrdiff_t x_stride_seqlen,
    ptrdiff_t x_stride_nhead,
    ptrdiff_t x_stride_batch,
    void *stream);

DESCRIPTOR(ascend)

#endif // __ACLNN_ROPE_H__
