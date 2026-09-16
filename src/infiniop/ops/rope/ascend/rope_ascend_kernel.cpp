#include "../../../devices/ascend/ascend_kernel_common.h"
#include <type_traits>

using namespace AscendC;

constexpr int32_t MASK_FP32 = 64;

template <typename T>
__aicore__ inline size_t AlignUpElems(size_t len) {
    size_t bytes = len * sizeof(T);
    size_t aligned = (bytes + BYTE_ALIGN - 1) / BYTE_ALIGN * BYTE_ALIGN;
    return aligned / sizeof(T);
}

__aicore__ inline void BuildPairSwapOffsets(LocalTensor<int32_t> &gatherOffset, int32_t count) {
    for (int32_t i = 0; i < 8 && i < count; ++i) {
        gatherOffset.SetValue(i, i ^ 1);
    }
    int32_t scalarValue = 8;
    while (scalarValue < count) {
        int32_t nextValue = scalarValue * 2;
        if (nextValue < count) {
            Adds(gatherOffset[scalarValue], gatherOffset, scalarValue, scalarValue);
        } else {
            Adds(gatherOffset[scalarValue], gatherOffset, scalarValue, count - scalarValue);
            break;
        }
        scalarValue = nextValue;
    }
}

__aicore__ inline void NegateEvenElements(LocalTensor<float> &src, int32_t count) {
    SetMaskNorm();
    const int32_t repeatTimes = count / MASK_FP32;
    const int32_t remainder = count % MASK_FP32;
    const uint64_t fullMask = 0x5555555555555555ULL;
    const uint64_t partialMask = 0x55ULL;
    SetVectorMask<float, MaskMode::NORMAL>(0, fullMask);
    Muls<float, false>(src, src, float(-1), MASK_PLACEHOLDER, repeatTimes, {1, 1, 8, 8});
    if (remainder) {
        SetVectorMask<float, MaskMode::NORMAL>(0, partialMask);
        Muls<float, false>(src[repeatTimes * MASK_FP32], src[repeatTimes * MASK_FP32], float(-1),
                           MASK_PLACEHOLDER, remainder / 8, {1, 1, 1, 1});
    }
    ResetMask();
}

// ==================== GPT_J Kernel (vectorized, soft-pipeline) ====================

template <typename T, typename U>
class RoPEKernel {
public:
    __aicore__ inline RoPEKernel() {}
    __aicore__ inline void init(GM_ADDR y, GM_ADDR x, GM_ADDR pos, GM_ADDR sin, GM_ADDR cos, size_t dh,
                                size_t nhead, size_t batch, ptrdiff_t st_ynt, ptrdiff_t st_ynh,
                                ptrdiff_t st_ynbatch, ptrdiff_t st_xnt, ptrdiff_t st_xnh,
                                ptrdiff_t st_xbatch);
    __aicore__ inline void process(size_t seq_len);

private:
    __aicore__ inline void copyIn(size_t i);
    __aicore__ inline void compute();
    __aicore__ inline void copyOut(size_t i);
    __aicore__ inline void computeGptJ(LocalTensor<T> &xUb, LocalTensor<T> &sinUb, LocalTensor<T> &cosUb,
                                       LocalTensor<T> &yUb);

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> _in_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> _sin_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> _cos_que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> _out_que;
    TBuf<TPosition::VECCALC> _offset_buf;
    TBuf<TPosition::VECCALC> _tmp_a;
    TBuf<TPosition::VECCALC> _tmp_b;
    TBuf<TPosition::VECCALC> _tmp_c;
    TBuf<TPosition::VECCALC> _tmp_d;
    TBuf<TPosition::VECCALC> _tmp_e;
    TBuf<TPosition::VECCALC> _tmp_f;

    GlobalTensor<T> _x_gm, _y_gm;
    GlobalTensor<U> _p_gm;
    GlobalTensor<T> _sin_gm;
    GlobalTensor<T> _cos_gm;

    size_t _block_idx;
    size_t _tile_len;
    size_t _half_len;
    size_t _copy_len;
    size_t _half_copy_len;
    size_t _batch;
    size_t _nhead;
    size_t _batch_idx;
    size_t _head_idx;

    ptrdiff_t _st_ynt;
    ptrdiff_t _st_ynh;
    ptrdiff_t _st_ynbatch;
    ptrdiff_t _st_xnt;
    ptrdiff_t _st_xnh;
    ptrdiff_t _st_xbatch;
};

template <typename T, typename U>
__aicore__ inline void RoPEKernel<T, U>::init(GM_ADDR y, GM_ADDR x, GM_ADDR pos, GM_ADDR sin, GM_ADDR cos,
                                              size_t dh, size_t nhead, size_t batch, ptrdiff_t st_ynt,
                                              ptrdiff_t st_ynh, ptrdiff_t st_ynbatch, ptrdiff_t st_xnt,
                                              ptrdiff_t st_xnh, ptrdiff_t st_xbatch) {
    this->_tile_len = dh;
    this->_half_len = dh / 2;
    this->_nhead = nhead;
    this->_batch = batch;
    this->_st_ynt = st_ynt;
    this->_st_ynh = st_ynh;
    this->_st_ynbatch = st_ynbatch;
    this->_st_xnt = st_xnt;
    this->_st_xnh = st_xnh;
    this->_st_xbatch = st_xbatch;
    _copy_len = AlignUpElems<T>(dh);
    _half_copy_len = AlignUpElems<T>(dh / 2);

    _block_idx = GetBlockIdx();
    _batch_idx = _block_idx / _nhead;
    _head_idx = _block_idx % _nhead;

    _x_gm.SetGlobalBuffer((__gm__ T *)x);
    _p_gm.SetGlobalBuffer((__gm__ U *)pos);
    _sin_gm.SetGlobalBuffer((__gm__ T *)sin);
    _cos_gm.SetGlobalBuffer((__gm__ T *)cos);
    _y_gm.SetGlobalBuffer((__gm__ T *)y);

    pipe.InitBuffer(_in_que, BUFFER_NUM, _copy_len * sizeof(T));
    pipe.InitBuffer(_out_que, BUFFER_NUM, _copy_len * sizeof(T));
    pipe.InitBuffer(_sin_que, BUFFER_NUM, _half_copy_len * sizeof(T));
    pipe.InitBuffer(_cos_que, BUFFER_NUM, _half_copy_len * sizeof(T));
    pipe.InitBuffer(_offset_buf, _copy_len * sizeof(int32_t) * 2);

    if constexpr (std::is_same<T, float>::value) {
        pipe.InitBuffer(_tmp_a, _copy_len * sizeof(T));
        pipe.InitBuffer(_tmp_b, _copy_len * sizeof(T));
        pipe.InitBuffer(_tmp_c, _copy_len * sizeof(T));
    } else {
        pipe.InitBuffer(_tmp_a, _copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_b, _copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_c, _copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_d, _copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_e, _half_copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_f, _half_copy_len * sizeof(float));
    }

    LocalTensor<int32_t> offsets = _offset_buf.Get<int32_t>();
    BuildPairSwapOffsets(offsets, static_cast<int32_t>(_tile_len));
    Muls(offsets, offsets, static_cast<int32_t>(sizeof(float)), static_cast<int32_t>(_tile_len));
    LocalTensor<int32_t> exp = offsets[_copy_len];
    for (uint32_t i = 0; i < _tile_len; ++i) {
        exp.SetValue(static_cast<int32_t>(i), static_cast<int32_t>((i / 2) * sizeof(float)));
    }
}

template <typename T, typename U>
__aicore__ inline void RoPEKernel<T, U>::copyIn(size_t i) {
    LocalTensor<T> input_ub = _in_que.AllocTensor<T>();
    LocalTensor<T> sin_ub = _sin_que.AllocTensor<T>();
    LocalTensor<T> cos_ub = _cos_que.AllocTensor<T>();

    auto idx = _batch_idx * _st_xbatch + i * _st_xnt + _head_idx * _st_xnh;
    DataCopyPad(input_ub, _x_gm[idx],
                {1, static_cast<uint32_t>(_tile_len * sizeof(T)), 0, 0, 0}, {true, 0, 0, 0});
    auto pos_idx = _p_gm(i);
    DataCopyPad(sin_ub, _sin_gm[pos_idx * _half_len],
                {1, static_cast<uint32_t>(_half_len * sizeof(T)), 0, 0, 0}, {true, 0, 0, 0});
    DataCopyPad(cos_ub, _cos_gm[pos_idx * _half_len],
                {1, static_cast<uint32_t>(_half_len * sizeof(T)), 0, 0, 0}, {true, 0, 0, 0});
    _in_que.EnQue(input_ub);
    _sin_que.EnQue(sin_ub);
    _cos_que.EnQue(cos_ub);
}

template <typename T, typename U>
__aicore__ inline void RoPEKernel<T, U>::computeGptJ(LocalTensor<T> &xUb, LocalTensor<T> &sinUb,
                                                     LocalTensor<T> &cosUb, LocalTensor<T> &yUb) {
    LocalTensor<int32_t> offsets = _offset_buf.Get<int32_t>();
    LocalTensor<uint32_t> swapOff = offsets.ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> expOff = offsets[_copy_len].ReinterpretCast<uint32_t>();

    if constexpr (std::is_same<T, float>::value) {
        LocalTensor<T> cosFull = _tmp_a.Get<T>();
        LocalTensor<T> sinFull = _tmp_b.Get<T>();
        LocalTensor<T> xSwap = _tmp_c.Get<T>();

        Gather(cosFull, cosUb, expOff, 0, _tile_len);
        Gather(sinFull, sinUb, expOff, 0, _tile_len);
        PipeBarrier<PIPE_V>();
        Mul(yUb, xUb, cosFull, _tile_len);
        Gather(xSwap, xUb, swapOff, 0, _tile_len);
        PipeBarrier<PIPE_V>();
        Mul(xSwap, xSwap, sinFull, _tile_len);
        PipeBarrier<PIPE_V>();
        NegateEvenElements(xSwap, static_cast<int32_t>(_tile_len));
        PipeBarrier<PIPE_V>();
        Add(yUb, yUb, xSwap, _tile_len);
    } else {
        LocalTensor<float> xF = _tmp_a.Get<float>();
        LocalTensor<float> yF = _tmp_b.Get<float>();
        LocalTensor<float> xSwap = _tmp_c.Get<float>();
        LocalTensor<float> cosFull = _tmp_d.Get<float>();
        LocalTensor<float> sinF = _tmp_e.Get<float>();
        LocalTensor<float> cosF = _tmp_f.Get<float>();

        Cast(xF, xUb, RoundMode::CAST_NONE, _tile_len);
        Cast(sinF, sinUb, RoundMode::CAST_NONE, _half_len);
        Cast(cosF, cosUb, RoundMode::CAST_NONE, _half_len);
        PipeBarrier<PIPE_V>();

        Gather(cosFull, cosF, expOff, 0, _tile_len);
        Gather(xSwap, sinF, expOff, 0, _tile_len);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> sinFull = xSwap;

        Mul(yF, xF, cosFull, _tile_len);
        Gather(cosFull, xF, swapOff, 0, _tile_len);
        PipeBarrier<PIPE_V>();
        Mul(cosFull, cosFull, sinFull, _tile_len);
        PipeBarrier<PIPE_V>();
        NegateEvenElements(cosFull, static_cast<int32_t>(_tile_len));
        PipeBarrier<PIPE_V>();
        Add(yF, yF, cosFull, _tile_len);
        PipeBarrier<PIPE_V>();
        Cast(yUb, yF, RoundMode::CAST_RINT, _tile_len);
    }
}

template <typename T, typename U>
__aicore__ inline void RoPEKernel<T, U>::compute() {
    LocalTensor<T> input_ub = _in_que.DeQue<T>();
    LocalTensor<T> sin_ub = _sin_que.DeQue<T>();
    LocalTensor<T> cos_ub = _cos_que.DeQue<T>();
    LocalTensor<T> output_ub = _out_que.AllocTensor<T>();
    computeGptJ(input_ub, sin_ub, cos_ub, output_ub);
    _out_que.EnQue<T>(output_ub);
    _in_que.FreeTensor(input_ub);
    _sin_que.FreeTensor(sin_ub);
    _cos_que.FreeTensor(cos_ub);
}

template <typename T, typename U>
__aicore__ inline void RoPEKernel<T, U>::copyOut(size_t i) {
    LocalTensor<T> output_ub = _out_que.DeQue<T>();
    auto idy = _batch_idx * _st_ynbatch + i * _st_ynt + _head_idx * _st_ynh;
    DataCopyPad(_y_gm[idy], output_ub, {1, static_cast<uint32_t>(_tile_len * sizeof(T)), 0, 0, 0});
    _out_que.FreeTensor(output_ub);
}

template <typename T, typename U>
__aicore__ inline void RoPEKernel<T, U>::process(size_t seq_len) {
    if (_batch_idx >= _batch || seq_len == 0) {
        return;
    }
    copyIn(0);
    for (size_t i = 0; i < seq_len; ++i) {
        if (i + 1 < seq_len) {
            copyIn(i + 1);
        }
        compute();
        copyOut(i);
    }
}

// ==================== GPT_NEOX Kernel ====================

template <typename T, typename U>
class RoPEKernelNeox {
public:
    __aicore__ inline RoPEKernelNeox() {}
    __aicore__ inline void init(GM_ADDR y, GM_ADDR x, GM_ADDR pos, GM_ADDR sin, GM_ADDR cos, size_t dh,
                                size_t nhead, size_t batch, ptrdiff_t st_ynt, ptrdiff_t st_ynh,
                                ptrdiff_t st_ynbatch, ptrdiff_t st_xnt, ptrdiff_t st_xnh,
                                ptrdiff_t st_xbatch);
    __aicore__ inline void process(size_t seq_len);

private:
    __aicore__ inline void copyIn(size_t i);
    __aicore__ inline void compute();
    __aicore__ inline void copyOut(size_t i);

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> _in_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> _sin_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> _cos_que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> _out_que;
    TBuf<TPosition::VECCALC> _tmp_a;
    TBuf<TPosition::VECCALC> _tmp_b;
    TBuf<TPosition::VECCALC> _tmp_c;
    TBuf<TPosition::VECCALC> _tmp_d;
    TBuf<TPosition::VECCALC> _tmp_e;
    TBuf<TPosition::VECCALC> _tmp_f;

    GlobalTensor<T> _x_gm, _y_gm;
    GlobalTensor<U> _p_gm;
    GlobalTensor<T> _sin_gm;
    GlobalTensor<T> _cos_gm;

    size_t _block_idx;
    size_t _tile_len;
    size_t _copy_len;
    size_t _half_len;
    size_t _half_copy_len;
    size_t _batch;
    size_t _nhead;
    size_t _batch_idx;
    size_t _head_idx;

    ptrdiff_t _st_ynt;
    ptrdiff_t _st_ynh;
    ptrdiff_t _st_ynbatch;
    ptrdiff_t _st_xnt;
    ptrdiff_t _st_xnh;
    ptrdiff_t _st_xbatch;
};

template <typename T, typename U>
__aicore__ inline void RoPEKernelNeox<T, U>::init(GM_ADDR y, GM_ADDR x, GM_ADDR pos, GM_ADDR sin, GM_ADDR cos,
                                                  size_t dh, size_t nhead, size_t batch, ptrdiff_t st_ynt,
                                                  ptrdiff_t st_ynh, ptrdiff_t st_ynbatch, ptrdiff_t st_xnt,
                                                  ptrdiff_t st_xnh, ptrdiff_t st_xbatch) {
    this->_tile_len = dh;
    this->_half_len = dh / 2;
    this->_nhead = nhead;
    this->_batch = batch;
    this->_st_ynt = st_ynt;
    this->_st_ynh = st_ynh;
    this->_st_ynbatch = st_ynbatch;
    this->_st_xnt = st_xnt;
    this->_st_xnh = st_xnh;
    this->_st_xbatch = st_xbatch;
    _copy_len = AlignUpElems<T>(dh);
    _half_copy_len = AlignUpElems<T>(dh / 2);

    _block_idx = GetBlockIdx();
    _batch_idx = _block_idx / _nhead;
    _head_idx = _block_idx % _nhead;

    _x_gm.SetGlobalBuffer((__gm__ T *)x);
    _p_gm.SetGlobalBuffer((__gm__ U *)pos);
    _sin_gm.SetGlobalBuffer((__gm__ T *)sin);
    _cos_gm.SetGlobalBuffer((__gm__ T *)cos);
    _y_gm.SetGlobalBuffer((__gm__ T *)y);

    pipe.InitBuffer(_in_que, BUFFER_NUM, _copy_len * sizeof(T));
    pipe.InitBuffer(_out_que, BUFFER_NUM, _copy_len * sizeof(T));
    pipe.InitBuffer(_sin_que, BUFFER_NUM, _half_copy_len * sizeof(T));
    pipe.InitBuffer(_cos_que, BUFFER_NUM, _half_copy_len * sizeof(T));

    if constexpr (std::is_same<T, float>::value) {
        pipe.InitBuffer(_tmp_a, _half_copy_len * sizeof(T));
        pipe.InitBuffer(_tmp_b, _half_copy_len * sizeof(T));
    } else {
        pipe.InitBuffer(_tmp_a, _copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_b, _copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_c, _half_copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_d, _half_copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_e, _half_copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_f, _half_copy_len * sizeof(float));
    }
}

template <typename T, typename U>
__aicore__ inline void RoPEKernelNeox<T, U>::copyIn(size_t i) {
    LocalTensor<T> input_ub = _in_que.AllocTensor<T>();
    LocalTensor<T> sin_ub = _sin_que.AllocTensor<T>();
    LocalTensor<T> cos_ub = _cos_que.AllocTensor<T>();

    auto idx = _batch_idx * _st_xbatch + i * _st_xnt + _head_idx * _st_xnh;
    DataCopyPad(input_ub, _x_gm[idx],
                {1, static_cast<uint32_t>(_tile_len * sizeof(T)), 0, 0, 0}, {true, 0, 0, 0});
    auto pos_idx = _p_gm(i);
    DataCopyPad(sin_ub, _sin_gm[pos_idx * _half_len],
                {1, static_cast<uint32_t>(_half_len * sizeof(T)), 0, 0, 0}, {true, 0, 0, 0});
    DataCopyPad(cos_ub, _cos_gm[pos_idx * _half_len],
                {1, static_cast<uint32_t>(_half_len * sizeof(T)), 0, 0, 0}, {true, 0, 0, 0});
    _in_que.EnQue(input_ub);
    _sin_que.EnQue(sin_ub);
    _cos_que.EnQue(cos_ub);
}

template <typename T, typename U>
__aicore__ inline void RoPEKernelNeox<T, U>::compute() {
    LocalTensor<T> input_ub = _in_que.DeQue<T>();
    LocalTensor<T> sin_ub = _sin_que.DeQue<T>();
    LocalTensor<T> cos_ub = _cos_que.DeQue<T>();
    LocalTensor<T> output_ub = _out_que.AllocTensor<T>();

    if constexpr (std::is_same<T, float>::value) {
        LocalTensor<T> t0 = _tmp_a.Get<T>();
        LocalTensor<T> t1 = _tmp_b.Get<T>();
        Mul(t0, input_ub, cos_ub, _half_len);
        Mul(t1, input_ub[_half_len], sin_ub, _half_len);
        PipeBarrier<PIPE_V>();
        Sub(output_ub, t0, t1, _half_len);
        Mul(t0, input_ub, sin_ub, _half_len);
        Mul(t1, input_ub[_half_len], cos_ub, _half_len);
        PipeBarrier<PIPE_V>();
        Add(output_ub[_half_len], t0, t1, _half_len);
    } else {
        LocalTensor<float> xF = _tmp_a.Get<float>();
        LocalTensor<float> yF = _tmp_b.Get<float>();
        LocalTensor<float> t0 = _tmp_c.Get<float>();
        LocalTensor<float> t1 = _tmp_d.Get<float>();
        LocalTensor<float> sinF = _tmp_e.Get<float>();
        LocalTensor<float> cosF = _tmp_f.Get<float>();

        Cast(xF, input_ub, RoundMode::CAST_NONE, _tile_len);
        Cast(sinF, sin_ub, RoundMode::CAST_NONE, _half_len);
        Cast(cosF, cos_ub, RoundMode::CAST_NONE, _half_len);
        PipeBarrier<PIPE_V>();

        Mul(t0, xF, cosF, _half_len);
        Mul(t1, xF[_half_len], sinF, _half_len);
        PipeBarrier<PIPE_V>();
        Sub(yF, t0, t1, _half_len);
        Mul(t0, xF, sinF, _half_len);
        Mul(t1, xF[_half_len], cosF, _half_len);
        PipeBarrier<PIPE_V>();
        Add(yF[_half_len], t0, t1, _half_len);
        PipeBarrier<PIPE_V>();
        Cast(output_ub, yF, RoundMode::CAST_RINT, _tile_len);
    }

    _out_que.EnQue<T>(output_ub);
    _in_que.FreeTensor(input_ub);
    _sin_que.FreeTensor(sin_ub);
    _cos_que.FreeTensor(cos_ub);
}

template <typename T, typename U>
__aicore__ inline void RoPEKernelNeox<T, U>::copyOut(size_t i) {
    LocalTensor<T> output_ub = _out_que.DeQue<T>();
    auto idy = _batch_idx * _st_ynbatch + i * _st_ynt + _head_idx * _st_ynh;
    DataCopyPad(_y_gm[idy], output_ub, {1, static_cast<uint32_t>(_tile_len * sizeof(T)), 0, 0, 0});
    _out_que.FreeTensor(output_ub);
}

template <typename T, typename U>
__aicore__ inline void RoPEKernelNeox<T, U>::process(size_t seq_len) {
    if (_batch_idx >= _batch || seq_len == 0) {
        return;
    }
    copyIn(0);
    for (size_t i = 0; i < seq_len; ++i) {
        if (i + 1 < seq_len) {
            copyIn(i + 1);
        }
        compute();
        copyOut(i);
    }
}

// ==================== Kernel Launch Macros ====================

#define ROPE_KERNEL_INIT_ARGS y, x, pos, sin, cos, dhead, nhead, batch,        \
                              y_stride_seqlen, y_stride_nhead, y_stride_batch, \
                              x_stride_seqlen, x_stride_nhead, x_stride_batch

#define CASE_POSTYPE(POS_TYPE_ENUM, TYPE, POS_T) \
    case POS_TYPE_ENUM: {                        \
        RoPEKernel<TYPE, POS_T> op;              \
        op.init(ROPE_KERNEL_INIT_ARGS);          \
        op.process(seq_len);                     \
        break;                                   \
    }

#define ROPE_KERNEL(TYPE, POSTYPE)                     \
    switch (POSTYPE) {                                 \
        CASE_POSTYPE(INFINI_DTYPE_I8, TYPE, int8_t)    \
        CASE_POSTYPE(INFINI_DTYPE_I16, TYPE, int16_t)  \
        CASE_POSTYPE(INFINI_DTYPE_I32, TYPE, int32_t)  \
        CASE_POSTYPE(INFINI_DTYPE_I64, TYPE, int64_t)  \
        CASE_POSTYPE(INFINI_DTYPE_U8, TYPE, uint8_t)   \
        CASE_POSTYPE(INFINI_DTYPE_U16, TYPE, uint16_t) \
        CASE_POSTYPE(INFINI_DTYPE_U32, TYPE, uint32_t) \
        CASE_POSTYPE(INFINI_DTYPE_U64, TYPE, uint64_t) \
    default:                                           \
        break;                                         \
    }

#define DEFINE_ROPE_KERNEL(KERNEL_NAME, TYPE)                         \
    __global__ __aicore__ void KERNEL_NAME(GM_ADDR y,                 \
                                           GM_ADDR x,                 \
                                           GM_ADDR pos,               \
                                           GM_ADDR sin,               \
                                           GM_ADDR cos,               \
                                           size_t seq_len,            \
                                           size_t dhead,              \
                                           size_t nhead,              \
                                           size_t batch,              \
                                           ptrdiff_t y_stride_seqlen, \
                                           ptrdiff_t y_stride_nhead,  \
                                           ptrdiff_t y_stride_batch,  \
                                           ptrdiff_t x_stride_seqlen, \
                                           ptrdiff_t x_stride_nhead,  \
                                           ptrdiff_t x_stride_batch,  \
                                           int32_t pos_type) {        \
        ROPE_KERNEL(TYPE, pos_type)                                   \
    }

DEFINE_ROPE_KERNEL(rope_kernel_float, float)
DEFINE_ROPE_KERNEL(rope_kernel_half, half)
DEFINE_ROPE_KERNEL(rope_kernel_bf16, bfloat16_t)

#undef DEFINE_ROPE_KERNEL
#undef ROPE_KERNEL
#undef CASE_POSTYPE
#undef ROPE_KERNEL_INIT_ARGS

#define ROPE_NEOX_KERNEL_INIT_ARGS y, x, pos, sin, cos, dhead, nhead, batch,        \
                                   y_stride_seqlen, y_stride_nhead, y_stride_batch, \
                                   x_stride_seqlen, x_stride_nhead, x_stride_batch

#define CASE_NEOX_POSTYPE(POS_TYPE_ENUM, TYPE, POS_T) \
    case POS_TYPE_ENUM: {                             \
        RoPEKernelNeox<TYPE, POS_T> op;               \
        op.init(ROPE_NEOX_KERNEL_INIT_ARGS);          \
        op.process(seq_len);                          \
        break;                                        \
    }

#define ROPE_NEOX_KERNEL(TYPE, POSTYPE)                     \
    switch (POSTYPE) {                                      \
        CASE_NEOX_POSTYPE(INFINI_DTYPE_I8, TYPE, int8_t)    \
        CASE_NEOX_POSTYPE(INFINI_DTYPE_I16, TYPE, int16_t)  \
        CASE_NEOX_POSTYPE(INFINI_DTYPE_I32, TYPE, int32_t)  \
        CASE_NEOX_POSTYPE(INFINI_DTYPE_I64, TYPE, int64_t)  \
        CASE_NEOX_POSTYPE(INFINI_DTYPE_U8, TYPE, uint8_t)   \
        CASE_NEOX_POSTYPE(INFINI_DTYPE_U16, TYPE, uint16_t) \
        CASE_NEOX_POSTYPE(INFINI_DTYPE_U32, TYPE, uint32_t) \
        CASE_NEOX_POSTYPE(INFINI_DTYPE_U64, TYPE, uint64_t) \
    default:                                                \
        break;                                              \
    }

#define DEFINE_ROPE_NEOX_KERNEL(KERNEL_NAME, TYPE)                    \
    __global__ __aicore__ void KERNEL_NAME(GM_ADDR y,                 \
                                           GM_ADDR x,                 \
                                           GM_ADDR pos,               \
                                           GM_ADDR sin,               \
                                           GM_ADDR cos,               \
                                           size_t seq_len,            \
                                           size_t dhead,              \
                                           size_t nhead,              \
                                           size_t batch,              \
                                           ptrdiff_t y_stride_seqlen, \
                                           ptrdiff_t y_stride_nhead,  \
                                           ptrdiff_t y_stride_batch,  \
                                           ptrdiff_t x_stride_seqlen, \
                                           ptrdiff_t x_stride_nhead,  \
                                           ptrdiff_t x_stride_batch,  \
                                           int32_t pos_type) {        \
        ROPE_NEOX_KERNEL(TYPE, pos_type)                              \
    }

DEFINE_ROPE_NEOX_KERNEL(rope_kernel_neox_float, float)
DEFINE_ROPE_NEOX_KERNEL(rope_kernel_neox_half, half)
DEFINE_ROPE_NEOX_KERNEL(rope_kernel_neox_bf16, bfloat16_t)

#undef DEFINE_ROPE_NEOX_KERNEL
#undef ROPE_NEOX_KERNEL
#undef CASE_NEOX_POSTYPE
#undef ROPE_NEOX_KERNEL_INIT_ARGS

extern "C" infiniStatus_t rope_kernel_launch(
    void *y, void *x, void *pos, void *sin, void *cos, size_t seq_len, size_t nhead, size_t dhead,
    size_t batch, infiniDtype_t dtype, infiniDtype_t pos_type, ptrdiff_t y_stride_seqlen,
    ptrdiff_t y_stride_nhead, ptrdiff_t y_stride_batch, ptrdiff_t x_stride_seqlen,
    ptrdiff_t x_stride_nhead, ptrdiff_t x_stride_batch, void *stream) {

#define LAUNCH_ROPE_KERNEL(DTYPE_ENUM, KERNEL_NAME)                          \
    case DTYPE_ENUM:                                                         \
        KERNEL_NAME<<<batch * nhead, nullptr, stream>>>(y, x, pos, sin, cos, \
                                                        seq_len,             \
                                                        dhead,               \
                                                        nhead,               \
                                                        batch,               \
                                                        y_stride_seqlen,     \
                                                        y_stride_nhead,      \
                                                        y_stride_batch,      \
                                                        x_stride_seqlen,     \
                                                        x_stride_nhead,      \
                                                        x_stride_batch,      \
                                                        pos_type);           \
        return INFINI_STATUS_SUCCESS;

    switch (dtype) {
        LAUNCH_ROPE_KERNEL(INFINI_DTYPE_F16, rope_kernel_half)
        LAUNCH_ROPE_KERNEL(INFINI_DTYPE_F32, rope_kernel_float)
        LAUNCH_ROPE_KERNEL(INFINI_DTYPE_BF16, rope_kernel_bf16)
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

extern "C" infiniStatus_t rope_kernel_neox_launch(
    void *y, void *x, void *pos, void *sin, void *cos, size_t seq_len, size_t nhead, size_t dhead,
    size_t batch, infiniDtype_t dtype, infiniDtype_t pos_type, ptrdiff_t y_stride_seqlen,
    ptrdiff_t y_stride_nhead, ptrdiff_t y_stride_batch, ptrdiff_t x_stride_seqlen,
    ptrdiff_t x_stride_nhead, ptrdiff_t x_stride_batch, void *stream) {

#define LAUNCH_ROPE_NEOX_KERNEL(DTYPE_ENUM, KERNEL_NAME)                     \
    case DTYPE_ENUM:                                                         \
        KERNEL_NAME<<<batch * nhead, nullptr, stream>>>(y, x, pos, sin, cos, \
                                                        seq_len,             \
                                                        dhead,               \
                                                        nhead,               \
                                                        batch,               \
                                                        y_stride_seqlen,     \
                                                        y_stride_nhead,      \
                                                        y_stride_batch,      \
                                                        x_stride_seqlen,     \
                                                        x_stride_nhead,      \
                                                        x_stride_batch,      \
                                                        pos_type);           \
        return INFINI_STATUS_SUCCESS;

    switch (dtype) {
        LAUNCH_ROPE_NEOX_KERNEL(INFINI_DTYPE_F16, rope_kernel_neox_half)
        LAUNCH_ROPE_NEOX_KERNEL(INFINI_DTYPE_F32, rope_kernel_neox_float)
        LAUNCH_ROPE_NEOX_KERNEL(INFINI_DTYPE_BF16, rope_kernel_neox_bf16)
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}
