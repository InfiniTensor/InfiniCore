#include "../../../devices/ascend/ascend_kernel_common.h"

using namespace AscendC;

template <typename T>
class TimeMixingKernel {
public:
    __aicore__ inline TimeMixingKernel() {}
    __aicore__ inline void init(GM_ADDR y, GM_ADDR r, GM_ADDR w, GM_ADDR k, GM_ADDR v, GM_ADDR a, GM_ADDR b, int B, int S, int C, int H, int N);
    __aicore__ inline void process();

private:
    __aicore__ inline void copyIn(int t);
    __aicore__ inline void copyOut(int t);
    __aicore__ inline void compute(int t);

private:
    TPipe _pipe;
    GlobalTensor<T> _r_gm, _y_gm, _w_gm, _k_gm, _v_gm, _a_gm, _b_gm;
    TQue<QuePosition::VECIN, BUFFER_NUM> _in_queue_r, _in_queue_w, _in_queue_k, _in_queue_v, _in_queue_a, _in_queue_b;
    TQue<QuePosition::VECOUT, BUFFER_NUM> _out_queue_y;
    TBuf<TPosition::VECCALC> _stateBuf;
    TBuf<TPosition::VECCALC> _rBuf, _wBuf, _kBuf, _vBuf, _aBuf, _bBuf;
    TBuf<TPosition::VECCALC> _tmpBuf, _tmp1Buf;

    int _batch;
    int _seq_len;
    int _channel;
    int _hidden_size;
    int _head_num;
    int _copy_len;
    int _batch_idx;
    int _head_idx;
};

template <typename T>
__aicore__ inline void TimeMixingKernel<T>::init(GM_ADDR y, GM_ADDR r, GM_ADDR w, GM_ADDR k, GM_ADDR v, GM_ADDR a, GM_ADDR b, int B, int S, int C, int H, int N) {
    _batch = B;
    _seq_len = S;
    _channel = C;
    _hidden_size = N;
    _head_num = H;
    int _block_idx = GetBlockIdx();
    _copy_len = alignTileLen<T>(_hidden_size, BYTE_ALIGN);
    _batch_idx = _block_idx / _head_num;
    _head_idx = _block_idx % _head_num;

    _y_gm.SetGlobalBuffer((__gm__ T *)y);
    _r_gm.SetGlobalBuffer((__gm__ T *)r);
    _w_gm.SetGlobalBuffer((__gm__ T *)w);
    _k_gm.SetGlobalBuffer((__gm__ T *)k);
    _v_gm.SetGlobalBuffer((__gm__ T *)v);
    _a_gm.SetGlobalBuffer((__gm__ T *)a);
    _b_gm.SetGlobalBuffer((__gm__ T *)b);

    _pipe.InitBuffer(_in_queue_r, BUFFER_NUM, _copy_len * sizeof(T));
    _pipe.InitBuffer(_in_queue_w, BUFFER_NUM, _copy_len * sizeof(T));
    _pipe.InitBuffer(_in_queue_k, BUFFER_NUM, _copy_len * sizeof(T));
    _pipe.InitBuffer(_in_queue_v, BUFFER_NUM, _copy_len * sizeof(T));
    _pipe.InitBuffer(_in_queue_a, BUFFER_NUM, _copy_len * sizeof(T));
    _pipe.InitBuffer(_in_queue_b, BUFFER_NUM, _copy_len * sizeof(T));
    _pipe.InitBuffer(_out_queue_y, BUFFER_NUM, _copy_len * sizeof(T));

    _pipe.InitBuffer(_rBuf, _copy_len * sizeof(float));
    _pipe.InitBuffer(_wBuf, _copy_len * sizeof(float));
    _pipe.InitBuffer(_kBuf, _copy_len * sizeof(float));
    _pipe.InitBuffer(_vBuf, _copy_len * sizeof(float));
    _pipe.InitBuffer(_aBuf, _copy_len * sizeof(float));
    _pipe.InitBuffer(_bBuf, _copy_len * sizeof(float));
    _pipe.InitBuffer(_tmpBuf, _copy_len * sizeof(float));
    _pipe.InitBuffer(_tmp1Buf, _copy_len * sizeof(float));

    _pipe.InitBuffer(_stateBuf, _hidden_size * _copy_len * sizeof(float));
    LocalTensor<float> stateBuffer = _stateBuf.Get<float>();
    for (int i = 0; i < _hidden_size * _copy_len; ++i) {
        stateBuffer.SetValue(i, 0.0f);
    }
}

template <typename T>
__aicore__ inline void TimeMixingKernel<T>::process() {
    for (int i = 0; i < _seq_len; ++i) {
        copyIn(i);
        compute(i);
        copyOut(i);
    }
}

template <typename T>
__aicore__ inline void TimeMixingKernel<T>::copyIn(int t) {
    LocalTensor<T> rLocal = _in_queue_r.AllocTensor<T>();
    LocalTensor<T> wLocal = _in_queue_w.AllocTensor<T>();
    LocalTensor<T> kLocal = _in_queue_k.AllocTensor<T>();
    LocalTensor<T> vLocal = _in_queue_v.AllocTensor<T>();
    LocalTensor<T> aLocal = _in_queue_a.AllocTensor<T>();
    LocalTensor<T> bLocal = _in_queue_b.AllocTensor<T>();
    ptrdiff_t idx = _batch_idx * _seq_len * _channel + t * _channel + _head_idx * _hidden_size;
    DataCopy(rLocal, _r_gm[idx], _copy_len);
    DataCopy(wLocal, _w_gm[idx], _copy_len);
    DataCopy(kLocal, _k_gm[idx], _copy_len);
    DataCopy(vLocal, _v_gm[idx], _copy_len);
    DataCopy(aLocal, _a_gm[idx], _copy_len);
    DataCopy(bLocal, _b_gm[idx], _copy_len);

    _in_queue_r.EnQue(rLocal);
    _in_queue_w.EnQue(wLocal);
    _in_queue_k.EnQue(kLocal);
    _in_queue_v.EnQue(vLocal);
    _in_queue_a.EnQue(aLocal);
    _in_queue_b.EnQue(bLocal);
}

template <typename T>
__aicore__ inline void TimeMixingKernel<T>::compute(int t) {
    LocalTensor<T> rLocal = _in_queue_r.DeQue<T>();
    LocalTensor<T> wLocal = _in_queue_w.DeQue<T>();
    LocalTensor<T> kLocal = _in_queue_k.DeQue<T>();
    LocalTensor<T> vLocal = _in_queue_v.DeQue<T>();
    LocalTensor<T> aLocal = _in_queue_a.DeQue<T>();
    LocalTensor<T> bLocal = _in_queue_b.DeQue<T>();

    LocalTensor<float> rBuf = _rBuf.Get<float>();
    LocalTensor<float> wBuf = _wBuf.Get<float>();
    LocalTensor<float> kBuf = _kBuf.Get<float>();
    LocalTensor<float> vBuf = _vBuf.Get<float>();
    LocalTensor<float> aBuf = _aBuf.Get<float>();
    LocalTensor<float> bBuf = _bBuf.Get<float>();
    LocalTensor<uint8_t> sharedTmpBuffer = _tmp1Buf.Get<uint8_t>();
    LocalTensor<float> tmpBuffer = _tmpBuf.Get<float>();

    Cast(rBuf, rLocal, RoundMode::CAST_NONE, _copy_len);
    Cast(wBuf, wLocal, RoundMode::CAST_NONE, _copy_len);
    Exp<float, 15, true>(tmpBuffer, wBuf, sharedTmpBuffer, _copy_len);
    Muls(tmpBuffer, tmpBuffer, -1.0f, _copy_len);
    Exp<float, 15, true>(wBuf, tmpBuffer, sharedTmpBuffer, _copy_len);
    Cast(kBuf, kLocal, RoundMode::CAST_NONE, _copy_len);
    Cast(vBuf, vLocal, RoundMode::CAST_NONE, _copy_len);
    Cast(aBuf, aLocal, RoundMode::CAST_NONE, _copy_len);
    Cast(bBuf, bLocal, RoundMode::CAST_NONE, _copy_len);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> stateBuffer = _stateBuf.Get<float>();
    LocalTensor<T> yLocal = _out_queue_y.AllocTensor<T>();
    for (int j = 0; j < _hidden_size; ++j) {
        float sa = 0.0f;
        for (int i = 0; i < _hidden_size; ++i) {
            sa += aBuf(i) * stateBuffer(j * _copy_len + i);
        }
        float y = 0.0f;
        float vv = vBuf(j);
        for (int i = 0; i < _hidden_size; ++i) {
            float tmp = stateBuffer(j * _copy_len + i) * wBuf(i) + kBuf(i) * vv + bBuf(i) * sa;
            stateBuffer(j * _copy_len + i) = tmp;
            y += tmp * rBuf(i);
        }
        yLocal(j) = static_cast<T>(y);
    }
    _out_queue_y.EnQue(yLocal);
    _in_queue_r.FreeTensor(rLocal);
    _in_queue_w.FreeTensor(wLocal);
    _in_queue_k.FreeTensor(kLocal);
    _in_queue_v.FreeTensor(vLocal);
    _in_queue_a.FreeTensor(aLocal);
    _in_queue_b.FreeTensor(bLocal);
}

template <>
__aicore__ inline void TimeMixingKernel<float>::compute(int t) {
    LocalTensor<float> rLocal = _in_queue_r.DeQue<float>();
    LocalTensor<float> wLocal = _in_queue_w.DeQue<float>();
    LocalTensor<float> kLocal = _in_queue_k.DeQue<float>();
    LocalTensor<float> vLocal = _in_queue_v.DeQue<float>();
    LocalTensor<float> aLocal = _in_queue_a.DeQue<float>();
    LocalTensor<float> bLocal = _in_queue_b.DeQue<float>();

    LocalTensor<uint8_t> sharedTmpBuffer = _tmp1Buf.Get<uint8_t>();
    LocalTensor<float> tmpBuffer = _tmpBuf.Get<float>();

    Exp<float, 15, true>(tmpBuffer, wLocal, sharedTmpBuffer, _copy_len);
    Muls(tmpBuffer, tmpBuffer, -1.0f, _copy_len);
    Exp<float, 15, true>(wLocal, tmpBuffer, sharedTmpBuffer, _copy_len);

    LocalTensor<float> stateBuffer = _stateBuf.Get<float>();
    LocalTensor<float> yLocal = _out_queue_y.AllocTensor<float>();
    for (int j = 0; j < _hidden_size; ++j) {
        float sa = 0.0f;
        for (int i = 0; i < _hidden_size; ++i) {
            sa += aLocal(i) * stateBuffer(j * _copy_len + i);
        }
        float y = 0.0f;
        float vv = vLocal(j);
        for (int i = 0; i < _hidden_size; ++i) {
            float tmp = stateBuffer(j * _copy_len + i) * wLocal(i) + kLocal(i) * vv + bLocal(i) * sa;
            stateBuffer(j * _copy_len + i) = tmp;
            y += tmp * rLocal(i);
        }
        yLocal(j) = y;
    }
    _out_queue_y.EnQue(yLocal);
    _in_queue_r.FreeTensor(rLocal);
    _in_queue_w.FreeTensor(wLocal);
    _in_queue_k.FreeTensor(kLocal);
    _in_queue_v.FreeTensor(vLocal);
    _in_queue_a.FreeTensor(aLocal);
    _in_queue_b.FreeTensor(bLocal);
}

template <typename T>
__aicore__ inline void TimeMixingKernel<T>::copyOut(int t) {
    LocalTensor<T> yLocal = _out_queue_y.DeQue<T>();
    ptrdiff_t idx = _batch_idx * _seq_len * _channel + t * _channel + _head_idx * _hidden_size;
    DataCopy(_y_gm[idx], yLocal, _copy_len);
    _out_queue_y.FreeTensor(yLocal);
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
    op.init(y, r, w, k, v, a, b, B, T, C, H, N);
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
    op.init(y, r, w, k, v, a, b, B, T, C, H, N);
    op.process();
}

extern "C" infiniStatus_t
time_mixing_kernel_launch(void *y, void *r, void *w, void *k, void *v, void *a, void *b,
                          int B, int T, int C, int H, int N,
                          infiniDtype_t dt, void *stream) {
    switch (dt) {
    case INFINI_DTYPE_F16:
        time_mixing_kernel_fp16<<<B * H, nullptr, stream>>>(y, r, w, k, v, a, b, B, T, C, H, N);
        break;
    case INFINI_DTYPE_F32:
        time_mixing_kernel_fp32<<<B * H, nullptr, stream>>>(y, r, w, k, v, a, b, B, T, C, H, N);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}