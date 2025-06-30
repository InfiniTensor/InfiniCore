#include "../../../devices/ascend/ascend_kernel_common.h"

using namespace AscendC;

template <typename T>
class TimeMixingKernel {
public:
    __aicore__ inline TimeMixingKernel() {}
    __aicore__ inline void init();
    __aicore__ inline void process();

private:
    __aicore__ inline void copyIn();
    __aicore__ inline void copyOut();
    __aicore__ inline void compute();

private:
    TPipe _pipe;
    GlobalTensor<T> _r_gm, _y_gm, _w_gm, _k_gm, _v_gm, _a_gm, _b_gm;
    TQue<QuePosition::VECIN, BUFFER_NUM> _in_queue_r, _in_queue_w, _in_queue_k, _in_queue_v, _in_queue_a, _in_queue_b;
    TQue<QuePosition::VECOUT, BUFFER_NUM> _out_queue_y;

    int _batch;
    int _seq_len;
    int _channel;
    int _hidden_size;
    int _head_num;
    int _copy_len;
}

template <typename T>
__aicore__ inline void TimeMixingKernel<T>::init(GM_ADDR y, GM_ADDR r, GM_ADDR w, GM_ADDR k, GM_ADDR v, GM_ADDR a, GM_ADDR b, int B, int T, int C, int H, int N) {
    _batch = B;
    _seq_len = T;
    _channel = C;
    _hidden_size = N;
    _head_num = H;
    _block_idx = GetBlockIdx();
    _copy_len = alignTileLen<T>(_hidden_size, BYTE_ALIGN);

    _y_gm.setGlobalBuffer((__gm__ T *)y);
    _r_gm.setGlobalBuffer((__gm__ T *)r);
    _w_gm.setGlobalBuffer((__gm__ T *)w);
    _k_gm.setGlobalBuffer((__gm__ T *)k);
    _v_gm.setGlobalBuffer((__gm__ T *)v);
    _a_gm.setGlobalBuffer((__gm__ T *)a);
    _b_gm.setGlobalBuffer((__gm__ T *)b);

    _pipe.InitBuffer(_in_queue_r, BUFFER_NUM, _copy_len * sizeof(T));
    _pipe.InitBuffer(_in_queue_w, BUFFER_NUM, _copy_len * sizeof(T));
    _pipe.InitBuffer(_in_queue_k, BUFFER_NUM, _copy_len * sizeof(T));
    _pipe.InitBuffer(_in_queue_v, BUFFER_NUM, _copy_len * sizeof(T));
    _pipe.InitBuffer(_in_queue_a, BUFFER_NUM, _copy_len * sizeof(T));
    _pipe.InitBuffer(_in_queue_b, BUFFER_NUM, _copy_len * sizeof(T));
    _pipe.InitBuffer(_out_queue_y, BUFFER_NUM, _copy_len * sizeof(T));
}

template <typename T>
__aicore__ inline void TimeMixingKernel<T>::process() {
    copyIn();
    compute();
    copyOut();
}

template <typename T>
__aicore__ inline void TimeMixingKernel<T>::copyIn() {
    LocalTensor<T> rLocal = _in_queue_r.AllocTensor<T>();
    LocalTensor<T> wLocal = _in_queue_w.AllocTensor<T>();
    LocalTensor<T> kLocal = _in_queue_k.AllocTensor<T>();
    LocalTensor<T> vLocal = _in_queue_v.AllocTensor<T>();
    LocalTensor<T> aLocal = _in_queue_a.AllocTensor<T>();
    LocalTensor<T> bLocal = _in_queue_b.AllocTensor<T>();
}

extern "C" __global__ __aicore__ void
time_mixing_kernel_fp16(
    GM_ADDR y,
    GM_ADDR r,
    GM_ADDR w,
    GM_ADDR k,
    GM_ADDR v,
    GM_ADDR a,
    GM_ADDR b,
    int B, int T, int C, int H, int N) {
    TimeMixingKernel<half> op;
    op.init();
    op.process();
}

extern "C" __global__ __aicore__ void
time_mixing_kernel_fp32(
    GM_ADDR y,
    GM_ADDR r,
    GM_ADDR w,
    GM_ADDR k,
    GM_ADDR v,
    GM_ADDR a,
    GM_ADDR b,
    int B, int T, int C, int H, int N) {
    TimeMixingKernel<float> op;
    op.init();
    op.process();
}

extern "C" infiniStatus_t
time_mixing_kernel_launch(void *y, void *r, void *w, void *k, void *v, void *a, void *b,
                          int B, int T, int C, int H, int N,
                          infiniDtype_t dt, void *stream) {
    switch (dt) {
    case INFINI_DTYPE_F16:
        time_mixing_kernel_fp16<<<B * H, nullptr, stream>>>();
        break;
    case INFINI_DTYPE_F32:
        time_mixing_kernel_fp32<<<B * H, nullptr, stream>>>();
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}