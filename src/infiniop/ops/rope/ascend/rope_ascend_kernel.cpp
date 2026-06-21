#include "../../../devices/ascend/ascend_kernel_common.h"
#include <type_traits>

using namespace AscendC;

template <typename T, typename U>
class RoPEKernel {
public:
    __aicore__ inline RoPEKernel() {}
    __aicore__ inline void init(GM_ADDR y,
                                GM_ADDR x,
                                GM_ADDR pos,
                                GM_ADDR sin,
                                GM_ADDR cos,
                                size_t dh,
                                size_t nhead,
                                size_t batch,
                                ptrdiff_t st_ynt,
                                ptrdiff_t st_ynh,
                                ptrdiff_t st_ynbatch,
                                ptrdiff_t st_xnt,
                                ptrdiff_t st_xnh,
                                ptrdiff_t st_xbatch);
    __aicore__ inline void process(size_t seq_len);

private:
    __aicore__ inline void copyIn(size_t i);
    __aicore__ inline void compute(size_t i);
    __aicore__ inline void copyOut(size_t i);

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> _in_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> _sin_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> _cos_que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> _out_que;
    TBuf<TPosition::VECCALC> _tmp_odd_buf;
    TBuf<TPosition::VECCALC> _tmp_even_buf;
    TBuf<TPosition::VECCALC> _tmp_odd_buf1;
    TBuf<TPosition::VECCALC> _tmp_odd_buf2;
    TBuf<TPosition::VECCALC> _tmp_even_buf1;
    TBuf<TPosition::VECCALC> _tmp_even_buf2;
    TBuf<TPosition::VECCALC> _tmp_float_input;
    TBuf<TPosition::VECCALC> _tmp_float_sin;
    TBuf<TPosition::VECCALC> _tmp_float_cos;
    TBuf<TPosition::VECCALC> _tmp_float_output;

    GlobalTensor<T> _x_gm, _y_gm;
    GlobalTensor<U> _p_gm;
    GlobalTensor<T> _sin_gm;
    GlobalTensor<T> _cos_gm;

    size_t _block_idx;
    size_t _tile_len;
    size_t _copy_len;
    size_t _half_copy_len;
    size_t _batch;
    size_t _nhead;

    ptrdiff_t _st_ynt;
    ptrdiff_t _st_ynh;
    ptrdiff_t _st_ynbatch;
    ptrdiff_t _st_xnt;
    ptrdiff_t _st_xnh;
    ptrdiff_t _st_xbatch;
};

template <typename T, typename U>
__aicore__ inline void RoPEKernel<T, U>::init(GM_ADDR y,
                                              GM_ADDR x,
                                              GM_ADDR pos,
                                              GM_ADDR sin,
                                              GM_ADDR cos,
                                              size_t dh,
                                              size_t nhead,
                                              size_t batch,
                                              ptrdiff_t st_ynt,
                                              ptrdiff_t st_ynh,
                                              ptrdiff_t st_ynbatch,
                                              ptrdiff_t st_xnt,
                                              ptrdiff_t st_xnh,
                                              ptrdiff_t st_xbatch) {
    this->_tile_len = dh;
    this->_nhead = nhead;
    this->_batch = batch;
    this->_st_ynt = st_ynt;
    this->_st_ynh = st_ynh;
    this->_st_ynbatch = st_ynbatch;
    this->_st_xnt = st_xnt;
    this->_st_xnh = st_xnh;
    this->_st_xbatch = st_xbatch;
    _copy_len = alignTileLen<T>(dh, BYTE_ALIGN);
    _half_copy_len = alignTileLen<T>(dh, BYTE_ALIGN);

    _block_idx = GetBlockIdx();

    _x_gm.SetGlobalBuffer((__gm__ T *)x);
    _p_gm.SetGlobalBuffer((__gm__ U *)pos);
    _sin_gm.SetGlobalBuffer((__gm__ T *)sin);
    _cos_gm.SetGlobalBuffer((__gm__ T *)cos);
    _y_gm.SetGlobalBuffer((__gm__ T *)y);

    pipe.InitBuffer(_in_que, BUFFER_NUM, _copy_len * sizeof(T));
    pipe.InitBuffer(_out_que, BUFFER_NUM, _tile_len * sizeof(T));
    pipe.InitBuffer(_sin_que, BUFFER_NUM, _half_copy_len * sizeof(T));
    pipe.InitBuffer(_cos_que, BUFFER_NUM, _half_copy_len * sizeof(T));
    // pipe.InitBuffer(_tmp_odd_buf, _tile_len / 2 * sizeof(T));
    // pipe.InitBuffer(_tmp_even_buf, _tile_len / 2 * sizeof(T));
    // pipe.InitBuffer(_tmp_odd_buf1, _tile_len / 2 * sizeof(T));
    // pipe.InitBuffer(_tmp_odd_buf2, _tile_len / 2 * sizeof(T));
    // pipe.InitBuffer(_tmp_even_buf1, _tile_len / 2 * sizeof(T));
    // pipe.InitBuffer(_tmp_even_buf2, _tile_len / 2 * sizeof(T));

    if constexpr (std::is_same<T, bfloat16_t>::value) {
        pipe.InitBuffer(_tmp_float_input, _copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_float_sin, _half_copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_float_cos, _half_copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_float_output, _tile_len * sizeof(float));
        pipe.InitBuffer(_tmp_odd_buf, _tile_len / 2 * sizeof(float));
        pipe.InitBuffer(_tmp_even_buf, _tile_len / 2 * sizeof(float));
        pipe.InitBuffer(_tmp_odd_buf1, _tile_len / 2 * sizeof(float));
        pipe.InitBuffer(_tmp_odd_buf2, _tile_len / 2 * sizeof(float));
        pipe.InitBuffer(_tmp_even_buf1, _tile_len / 2 * sizeof(float));
        pipe.InitBuffer(_tmp_even_buf2, _tile_len / 2 * sizeof(float));
    } else {
        pipe.InitBuffer(_tmp_odd_buf, _tile_len / 2 * sizeof(T));
        pipe.InitBuffer(_tmp_even_buf, _tile_len / 2 * sizeof(T));
        pipe.InitBuffer(_tmp_odd_buf1, _tile_len / 2 * sizeof(T));
        pipe.InitBuffer(_tmp_odd_buf2, _tile_len / 2 * sizeof(T));
        pipe.InitBuffer(_tmp_even_buf1, _tile_len / 2 * sizeof(T));
        pipe.InitBuffer(_tmp_even_buf2, _tile_len / 2 * sizeof(T));
    }
}

template <typename T, typename U>
__aicore__ inline void RoPEKernel<T, U>::copyIn(size_t i) {
    LocalTensor<T> input_ub = _in_que.AllocTensor<T>();
    LocalTensor<T> sin_ub = _sin_que.AllocTensor<T>();
    LocalTensor<T> cos_ub = _cos_que.AllocTensor<T>();

    size_t batch_idx = _block_idx / _nhead;
    size_t head_idx = _block_idx % _nhead;

    auto idx = batch_idx * _st_xbatch + i * _st_xnt + head_idx * _st_xnh;
    DataCopy(input_ub, _x_gm[idx], _copy_len);
    auto pos_idx = _p_gm(i);
    DataCopy(sin_ub, _sin_gm[pos_idx * _tile_len / 2], _half_copy_len);
    DataCopy(cos_ub, _cos_gm[pos_idx * _tile_len / 2], _half_copy_len);
    _in_que.EnQue(input_ub);
    _sin_que.EnQue(sin_ub);
    _cos_que.EnQue(cos_ub);
}

template <typename T, typename U>
__aicore__ inline void RoPEKernel<T, U>::compute(size_t i) {
    LocalTensor<T> input_ub = _in_que.DeQue<T>();
    LocalTensor<T> sin_ub = _sin_que.DeQue<T>();
    LocalTensor<T> cos_ub = _cos_que.DeQue<T>();
    LocalTensor<T> output_ub = _out_que.AllocTensor<T>();

    if constexpr (std::is_same<T, bfloat16_t>::value) {
        LocalTensor<float> input_float = _tmp_float_input.Get<float>();
        LocalTensor<float> sin_float = _tmp_float_sin.Get<float>();
        LocalTensor<float> cos_float = _tmp_float_cos.Get<float>();
        LocalTensor<float> tmp_odd_f = _tmp_odd_buf.Get<float>();
        LocalTensor<float> tmp_even_f = _tmp_even_buf.Get<float>();
        LocalTensor<float> tmp_odd1_f = _tmp_odd_buf1.Get<float>();
        LocalTensor<float> tmp_odd2_f = _tmp_odd_buf2.Get<float>();
        LocalTensor<float> tmp_even1_f = _tmp_even_buf1.Get<float>();
        LocalTensor<float> tmp_even2_f = _tmp_even_buf2.Get<float>();

        Cast(input_float, input_ub, AscendC::RoundMode::CAST_NONE, _copy_len);
        Cast(sin_float, sin_ub, AscendC::RoundMode::CAST_NONE, _half_copy_len);
        Cast(cos_float, cos_ub, AscendC::RoundMode::CAST_NONE, _half_copy_len);

        uint64_t rsvdCnt = 0;
        GatherMaskParams gMaskParams = {
            1,
            static_cast<uint16_t>((_tile_len * sizeof(float) + 255) / 256),
            8,
            8,
        };

        // DumpTensor(input_float, 0, _copy_len);
        // DumpTensor(sin_float, 1, _half_copy_len);
        // DumpTensor(cos_float, 2, _half_copy_len);

        GatherMask<float>(tmp_odd_f, input_float, 1, false, 0, gMaskParams, rsvdCnt);
        GatherMask<float>(tmp_even_f, input_float, 2, false, 0, gMaskParams, rsvdCnt);
        PipeBarrier<PIPE_V>();

        // DumpTensor(tmp_odd_f, 3, _tile_len / 2);
        // DumpTensor(tmp_even_f, 4, _tile_len / 2);

        Mul<float>(tmp_odd1_f, tmp_odd_f, cos_float, _tile_len / 2);
        Mul<float>(tmp_odd2_f, tmp_even_f, sin_float, _tile_len / 2);
        PipeBarrier<PIPE_V>();

        // DumpTensor(tmp_odd1_f, 5, _tile_len / 2);
        // DumpTensor(tmp_odd2_f, 6, _tile_len / 2);

        Sub<float>(tmp_odd1_f, tmp_odd1_f, tmp_odd2_f, _tile_len / 2);

        // DumpTensor(tmp_odd1_f, 7, _tile_len / 2);

        Mul<float>(tmp_even1_f, tmp_odd_f, sin_float, _tile_len / 2);
        Mul<float>(tmp_even2_f, tmp_even_f, cos_float, _tile_len / 2);
        PipeBarrier<PIPE_V>();

        // DumpTensor(tmp_even1_f, 8, _tile_len / 2);
        // DumpTensor(tmp_even2_f, 9, _tile_len / 2);

        Add<float>(tmp_even1_f, tmp_even1_f, tmp_even2_f, _tile_len / 2);

        // DumpTensor(tmp_odd1_f, 10, _tile_len / 2);
        // DumpTensor(tmp_even1_f, 11, _tile_len / 2);

        LocalTensor<float> output_float = _tmp_float_output.Get<float>();
        for (uint32_t j = 0; j < _tile_len / 2; j += 1) {
            output_float(j * 2) = tmp_odd1_f(j);
            output_float(j * 2 + 1) = tmp_even1_f(j);
        }

        // DumpTensor(output_float, 12, _tile_len);

        Cast(output_ub, output_float, AscendC::RoundMode::CAST_RINT, _tile_len);
    } else {
        LocalTensor<T> tmp_odd = _tmp_odd_buf.Get<T>();
        LocalTensor<T> tmp_even = _tmp_even_buf.Get<T>();
        LocalTensor<T> tmp_odd1 = _tmp_odd_buf1.Get<T>();
        LocalTensor<T> tmp_odd2 = _tmp_odd_buf2.Get<T>();
        LocalTensor<T> tmp_even1 = _tmp_even_buf1.Get<T>();
        LocalTensor<T> tmp_even2 = _tmp_even_buf2.Get<T>();

        uint64_t rsvdCnt = 0;
        GatherMaskParams gMaskParams = {
            1,
            static_cast<uint16_t>((_tile_len * sizeof(T) + 255) / 256),
            8,
            8,
        };

        // DumpTensor(input_ub, 0, _copy_len);
        // DumpTensor(sin_ub, 1, _half_copy_len);
        // DumpTensor(cos_ub, 2, _half_copy_len);

        GatherMask<T>(tmp_odd, input_ub, 1, false, 0, gMaskParams, rsvdCnt);
        GatherMask<T>(tmp_even, input_ub, 2, false, 0, gMaskParams, rsvdCnt);
        PipeBarrier<PIPE_V>();

        // DumpTensor(tmp_odd, 3, _tile_len / 2);
        // DumpTensor(tmp_even, 4, _tile_len / 2);

        Mul<T>(tmp_odd1, tmp_odd, cos_ub, _tile_len / 2);
        Mul<T>(tmp_odd2, tmp_even, sin_ub, _tile_len / 2);
        PipeBarrier<PIPE_V>();

        // DumpTensor(tmp_odd1, 5, _tile_len / 2);
        // DumpTensor(tmp_odd2, 6, _tile_len / 2);

        Sub<T>(tmp_odd1, tmp_odd1, tmp_odd2, _tile_len / 2);

        // DumpTensor(tmp_odd1, 7, _tile_len / 2);

        Mul<T>(tmp_even1, tmp_odd, sin_ub, _tile_len / 2);
        Mul<T>(tmp_even2, tmp_even, cos_ub, _tile_len / 2);
        PipeBarrier<PIPE_V>();

        // DumpTensor(tmp_even1, 8, _tile_len / 2);
        // DumpTensor(tmp_even2, 9, _tile_len / 2);

        Add<T>(tmp_even1, tmp_even1, tmp_even2, _tile_len / 2);

        // DumpTensor(tmp_odd1, 10, _tile_len / 2);
        // DumpTensor(tmp_even1, 11, _tile_len / 2);

        for (uint32_t j = 0; j < _tile_len / 2; j += 1) {
            output_ub(j * 2) = tmp_odd1(j);
            output_ub(j * 2 + 1) = tmp_even1(j);
        }
        // DumpTensor(output_ub, 12, _tile_len);
    }

    _out_que.EnQue<T>(output_ub);
    _in_que.FreeTensor(input_ub);
    _sin_que.FreeTensor(sin_ub);
    _cos_que.FreeTensor(cos_ub);
}

template <typename T, typename U>
__aicore__ inline void RoPEKernel<T, U>::copyOut(size_t i) {
    LocalTensor<T> output_ub = _out_que.DeQue<T>();
    // size_t batch_idx = _block_idx / _nhead;
    // size_t head_idx = _block_idx % _nhead;
    // auto idy = batch_idx * _st_ynbatch + i * _st_ynt + head_idx * _st_ynh;
    // auto idy = i * _st_ynt + _block_idx * _st_ynh;
    size_t batch_idx = _block_idx / _nhead;
    size_t head_idx = _block_idx % _nhead;
    auto idy = batch_idx * _st_ynbatch + i * _st_ynt + head_idx * _st_ynh;
    DataCopyExtParams params = {1, static_cast<uint32_t>(_tile_len * sizeof(T)), 0, 0, 0};
    DataCopyPad(_y_gm[idy], output_ub, params);
    _out_que.FreeTensor(output_ub);
}

template <typename T, typename U>
__aicore__ inline void RoPEKernel<T, U>::process(size_t seq_len) {
    for (size_t i = 0; i < seq_len; ++i) {
        copyIn(i);
        compute(i);
        copyOut(i);
    }
}

// ==================== GPT_NEOX Kernel ====================

template <typename T, typename U>
class RoPEKernelNeox {
public:
    __aicore__ inline RoPEKernelNeox() {}
    __aicore__ inline void init(GM_ADDR y,
                                GM_ADDR x,
                                GM_ADDR pos,
                                GM_ADDR sin,
                                GM_ADDR cos,
                                size_t dh,
                                size_t nhead,
                                size_t batch,
                                ptrdiff_t st_ynt,
                                ptrdiff_t st_ynh,
                                ptrdiff_t st_ynbatch,
                                ptrdiff_t st_xnt,
                                ptrdiff_t st_xnh,
                                ptrdiff_t st_xbatch);
    __aicore__ inline void process(size_t seq_len);

private:
    __aicore__ inline void copyIn(size_t i);
    __aicore__ inline void compute(size_t i);
    __aicore__ inline void copyOut(size_t i);

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> _in_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> _sin_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> _cos_que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> _out_que;
    TBuf<TPosition::VECCALC> _tmp_first_half;
    TBuf<TPosition::VECCALC> _tmp_second_half;
    TBuf<TPosition::VECCALC> _tmp_result1;
    TBuf<TPosition::VECCALC> _tmp_result2;
    TBuf<TPosition::VECCALC> _tmp_result3;
    TBuf<TPosition::VECCALC> _tmp_result4;
    TBuf<TPosition::VECCALC> _tmp_float_input;
    TBuf<TPosition::VECCALC> _tmp_float_sin;
    TBuf<TPosition::VECCALC> _tmp_float_cos;
    TBuf<TPosition::VECCALC> _tmp_float_output;

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

    ptrdiff_t _st_ynt;
    ptrdiff_t _st_ynh;
    ptrdiff_t _st_ynbatch;
    ptrdiff_t _st_xnt;
    ptrdiff_t _st_xnh;
    ptrdiff_t _st_xbatch;
};

template <typename T, typename U>
__aicore__ inline void RoPEKernelNeox<T, U>::init(GM_ADDR y,
                                                  GM_ADDR x,
                                                  GM_ADDR pos,
                                                  GM_ADDR sin,
                                                  GM_ADDR cos,
                                                  size_t dh,
                                                  size_t nhead,
                                                  size_t batch,
                                                  ptrdiff_t st_ynt,
                                                  ptrdiff_t st_ynh,
                                                  ptrdiff_t st_ynbatch,
                                                  ptrdiff_t st_xnt,
                                                  ptrdiff_t st_xnh,
                                                  ptrdiff_t st_xbatch) {
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
    _copy_len = alignTileLen<T>(dh, BYTE_ALIGN);
    _half_copy_len = alignTileLen<T>(_half_len, BYTE_ALIGN);

    _block_idx = GetBlockIdx();

    _x_gm.SetGlobalBuffer((__gm__ T *)x);
    _p_gm.SetGlobalBuffer((__gm__ U *)pos);
    _sin_gm.SetGlobalBuffer((__gm__ T *)sin);
    _cos_gm.SetGlobalBuffer((__gm__ T *)cos);
    _y_gm.SetGlobalBuffer((__gm__ T *)y);

    pipe.InitBuffer(_in_que, BUFFER_NUM, _copy_len * sizeof(T));
    pipe.InitBuffer(_out_que, BUFFER_NUM, _copy_len * sizeof(T));
    pipe.InitBuffer(_sin_que, BUFFER_NUM, _half_copy_len * sizeof(T));
    pipe.InitBuffer(_cos_que, BUFFER_NUM, _half_copy_len * sizeof(T));

    if constexpr (std::is_same<T, bfloat16_t>::value) {
        size_t half_float_copy_len = alignTileLen<float>(_half_len, BYTE_ALIGN);
        pipe.InitBuffer(_tmp_float_input, _copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_float_sin, half_float_copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_float_cos, half_float_copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_float_output, _tile_len * sizeof(float));
        pipe.InitBuffer(_tmp_first_half, half_float_copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_second_half, half_float_copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_result1, half_float_copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_result2, half_float_copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_result3, half_float_copy_len * sizeof(float));
        pipe.InitBuffer(_tmp_result4, half_float_copy_len * sizeof(float));
    } else {
        pipe.InitBuffer(_tmp_first_half, _half_copy_len * sizeof(T));
        pipe.InitBuffer(_tmp_second_half, _half_copy_len * sizeof(T));
        pipe.InitBuffer(_tmp_result1, _half_copy_len * sizeof(T));
        pipe.InitBuffer(_tmp_result2, _half_copy_len * sizeof(T));
        pipe.InitBuffer(_tmp_result3, _half_copy_len * sizeof(T));
        pipe.InitBuffer(_tmp_result4, _half_copy_len * sizeof(T));
    }
}

template <typename T, typename U>
__aicore__ inline void RoPEKernelNeox<T, U>::copyIn(size_t i) {
    LocalTensor<T> input_ub = _in_que.AllocTensor<T>();
    LocalTensor<T> sin_ub = _sin_que.AllocTensor<T>();
    LocalTensor<T> cos_ub = _cos_que.AllocTensor<T>();

    size_t batch_idx = _block_idx / _nhead;
    size_t head_idx = _block_idx % _nhead;

    auto idx = batch_idx * _st_xbatch + i * _st_xnt + head_idx * _st_xnh;
    DataCopy(input_ub, _x_gm[idx], _copy_len);
    auto pos_idx = _p_gm(i);
    DataCopyExtParams halfCopyParams = {1, static_cast<uint32_t>(_half_len * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> halfPadParams{true, 0, 0, 0};
    DataCopyPad(sin_ub, _sin_gm[pos_idx * _half_len], halfCopyParams, halfPadParams);
    DataCopyPad(cos_ub, _cos_gm[pos_idx * _half_len], halfCopyParams, halfPadParams);
    _in_que.EnQue(input_ub);
    _sin_que.EnQue(sin_ub);
    _cos_que.EnQue(cos_ub);
}

template <typename T, typename U>
__aicore__ inline void RoPEKernelNeox<T, U>::compute(size_t i) {
    LocalTensor<T> input_ub = _in_que.DeQue<T>();
    LocalTensor<T> sin_ub = _sin_que.DeQue<T>();
    LocalTensor<T> cos_ub = _cos_que.DeQue<T>();
    LocalTensor<T> output_ub = _out_que.AllocTensor<T>();

    if constexpr (std::is_same<T, bfloat16_t>::value) {
        LocalTensor<float> input_f = _tmp_float_input.Get<float>();
        LocalTensor<float> sin_f = _tmp_float_sin.Get<float>();
        LocalTensor<float> cos_f = _tmp_float_cos.Get<float>();
        LocalTensor<float> first_half_f = _tmp_first_half.Get<float>();
        LocalTensor<float> second_half_f = _tmp_second_half.Get<float>();
        LocalTensor<float> result1_f = _tmp_result1.Get<float>();
        LocalTensor<float> result2_f = _tmp_result2.Get<float>();
        LocalTensor<float> result3_f = _tmp_result3.Get<float>();
        LocalTensor<float> result4_f = _tmp_result4.Get<float>();
        size_t half_float_copy_len = alignTileLen<float>(_half_len, BYTE_ALIGN);

        Cast(input_f, input_ub, AscendC::RoundMode::CAST_NONE, _copy_len);
        Cast(sin_f, sin_ub, AscendC::RoundMode::CAST_NONE, half_float_copy_len);
        Cast(cos_f, cos_ub, AscendC::RoundMode::CAST_NONE, half_float_copy_len);

        for (size_t j = 0; j < half_float_copy_len; j++) {
            if (j < _half_len) {
                first_half_f(j) = input_f(j);
                second_half_f(j) = input_f(_half_len + j);
            } else {
                first_half_f(j) = 0.0f;
                second_half_f(j) = 0.0f;
            }
        }
        PipeBarrier<PIPE_V>();

        Mul<float>(result1_f, first_half_f, cos_f, half_float_copy_len);
        Mul<float>(result2_f, second_half_f, sin_f, half_float_copy_len);
        PipeBarrier<PIPE_V>();
        Sub<float>(result3_f, result1_f, result2_f, half_float_copy_len);

        Mul<float>(result1_f, first_half_f, sin_f, half_float_copy_len);
        Mul<float>(result2_f, second_half_f, cos_f, half_float_copy_len);
        PipeBarrier<PIPE_V>();
        Add<float>(result4_f, result1_f, result2_f, half_float_copy_len);

        LocalTensor<float> output_f = _tmp_float_output.Get<float>();
        for (size_t j = 0; j < _half_len; j++) {
            output_f(j) = result3_f(j);
            output_f(_half_len + j) = result4_f(j);
        }

        Cast(output_ub, output_f, AscendC::RoundMode::CAST_RINT, _tile_len);

    } else {
        LocalTensor<T> first_half = _tmp_first_half.Get<T>();
        LocalTensor<T> second_half = _tmp_second_half.Get<T>();
        LocalTensor<T> result1 = _tmp_result1.Get<T>();
        LocalTensor<T> result2 = _tmp_result2.Get<T>();
        LocalTensor<T> result3 = _tmp_result3.Get<T>();
        LocalTensor<T> result4 = _tmp_result4.Get<T>();

        for (size_t j = 0; j < _half_copy_len; j++) {
            if (j < _half_len) {
                first_half(j) = input_ub(j);
                second_half(j) = input_ub(_half_len + j);
            } else {
                first_half(j) = static_cast<T>(0);
                second_half(j) = static_cast<T>(0);
            }
        }
        PipeBarrier<PIPE_V>();

        Mul<T>(result1, first_half, cos_ub, _half_copy_len);
        Mul<T>(result2, second_half, sin_ub, _half_copy_len);
        PipeBarrier<PIPE_V>();
        Sub<T>(result3, result1, result2, _half_copy_len);

        Mul<T>(result1, first_half, sin_ub, _half_copy_len);
        Mul<T>(result2, second_half, cos_ub, _half_copy_len);
        PipeBarrier<PIPE_V>();
        Add<T>(result4, result1, result2, _half_copy_len);

        for (size_t j = 0; j < _half_len; j++) {
            output_ub(j) = result3(j);
            output_ub(_half_len + j) = result4(j);
        }
    }
    // DumpTensor(output_ub, 5, _tile_len);
    _out_que.EnQue<T>(output_ub);
    _in_que.FreeTensor(input_ub);
    _sin_que.FreeTensor(sin_ub);
    _cos_que.FreeTensor(cos_ub);
}

template <typename T, typename U>
__aicore__ inline void RoPEKernelNeox<T, U>::copyOut(size_t i) {
    LocalTensor<T> output_ub = _out_que.DeQue<T>();
    size_t batch_idx = _block_idx / _nhead;
    size_t head_idx = _block_idx % _nhead;
    auto idy = batch_idx * _st_ynbatch + i * _st_ynt + head_idx * _st_ynh;
    DataCopyExtParams params = {1, static_cast<uint32_t>(_tile_len * sizeof(T)), 0, 0, 0};
    DataCopyPad(_y_gm[idy], output_ub, params);
    _out_que.FreeTensor(output_ub);
}

template <typename T, typename U>
__aicore__ inline void RoPEKernelNeox<T, U>::process(size_t seq_len) {
    for (size_t i = 0; i < seq_len; ++i) {
        copyIn(i);
        compute(i);
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

// ==================== GPT_NEOX Kernel Launch ====================

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

// ==================== External Launch Functions ====================

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
    infiniDtype_t dtype,
    infiniDtype_t pos_type,
    ptrdiff_t y_stride_seqlen,
    ptrdiff_t y_stride_nhead,
    ptrdiff_t y_stride_batch,
    ptrdiff_t x_stride_seqlen,
    ptrdiff_t x_stride_nhead,
    ptrdiff_t x_stride_batch,
    void *stream) {

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
    void *y,
    void *x,
    void *pos,
    void *sin,
    void *cos,
    size_t seq_len,
    size_t nhead,
    size_t dhead,
    size_t batch,
    infiniDtype_t dtype,
    infiniDtype_t pos_type,
    ptrdiff_t y_stride_seqlen,
    ptrdiff_t y_stride_nhead,
    ptrdiff_t y_stride_batch,
    ptrdiff_t x_stride_seqlen,
    ptrdiff_t x_stride_nhead,
    ptrdiff_t x_stride_batch,
    void *stream) {

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
