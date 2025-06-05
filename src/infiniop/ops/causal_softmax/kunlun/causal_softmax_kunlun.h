#ifndef __CAUSAL_SOFTMAX_KUNLUN_H__
#define __CAUSAL_SOFTMAX_KUNLUN_H__

#include "../causal_softmax.h"

DESCRIPTOR(kunlun)

void causalSoftmaxF32(void *y, const void *x, void *workspace, int batch, int seq_len, int total_seq_len,
                      int y_stride_b, int y_stride_i, int x_stride_b, int x_stride_i, void *stream);

#endif
