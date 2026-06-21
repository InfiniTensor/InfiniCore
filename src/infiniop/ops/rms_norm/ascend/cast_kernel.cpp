#include "../../../devices/ascend/ascend_kernel_common.h"

using namespace AscendC;

template <typename SrcT, typename DstT>
class CastKernel {
public:
    __aicore__ inline CastKernel() {}
    __aicore__ inline void init(GM_ADDR dst, GM_ADDR src, size_t count);
    __aicore__ inline void process();

private:
    __aicore__ inline void copyIn();
    __aicore__ inline void compute();
    __aicore__ inline void copyOut();

    GlobalTensor<SrcT> _src_gm;
    GlobalTensor<DstT> _dst_gm;
    TQue<QuePosition::VECIN, BUFFER_NUM> _in_queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> _out_queue;
    TPipe _pipe;
    size_t _tile_len, _copy_len;
};

template <typename SrcT, typename DstT>
__aicore__ inline void CastKernel<SrcT, DstT>::init(GM_ADDR dst, GM_ADDR src, size_t count) {
    _tile_len = count;
    _copy_len = alignTileLen<SrcT>(_tile_len, BYTE_ALIGN);

    _dst_gm.SetGlobalBuffer((__gm__ DstT *)dst);
    _src_gm.SetGlobalBuffer((__gm__ SrcT *)src);

    _pipe.InitBuffer(_in_queue, BUFFER_NUM, _copy_len * sizeof(SrcT));
    _pipe.InitBuffer(_out_queue, BUFFER_NUM, _copy_len * sizeof(DstT));
}

template <typename SrcT, typename DstT>
__aicore__ inline void CastKernel<SrcT, DstT>::copyIn() {
    LocalTensor<SrcT> srcLocal = _in_queue.AllocTensor<SrcT>();
    DataCopy(srcLocal, _src_gm, _copy_len);
    _in_queue.EnQue(srcLocal);
}

template <typename SrcT, typename DstT>
__aicore__ inline void CastKernel<SrcT, DstT>::compute() {
    LocalTensor<SrcT> srcLocal = _in_queue.DeQue<SrcT>();
    LocalTensor<DstT> dstLocal = _out_queue.AllocTensor<DstT>();
    Cast(dstLocal, srcLocal, RoundMode::CAST_NONE, _copy_len);
    _out_queue.EnQue<DstT>(dstLocal);
    _in_queue.FreeTensor(srcLocal);
}

template <typename SrcT, typename DstT>
__aicore__ inline void CastKernel<SrcT, DstT>::copyOut() {
    LocalTensor<DstT> dstLocal = _out_queue.DeQue<DstT>();
    if (_tile_len * sizeof(DstT) % BYTE_ALIGN != 0) {
        DataCopyExtParams dcep = {1, static_cast<uint32_t>(_tile_len * sizeof(DstT)), 0, 0, 0};
        DataCopyPad(_dst_gm, dstLocal, dcep);
    } else {
        DataCopy(_dst_gm, dstLocal, _tile_len);
    }
    _out_queue.FreeTensor(dstLocal);
}

template <typename SrcT, typename DstT>
__aicore__ inline void CastKernel<SrcT, DstT>::process() {
    copyIn();
    compute();
    copyOut();
}

#define DEFINE_CAST_KERNEL(KERNEL_NAME, SRC_T, DST_T)                \
    __global__ __aicore__ void KERNEL_NAME(GM_ADDR dst, GM_ADDR src, \
                                           size_t count) {           \
        CastKernel<SRC_T, DST_T> op;                                 \
        op.init(dst, src, count);                                    \
        op.process();                                                \
    }

DEFINE_CAST_KERNEL(cast_kernel_f16_to_f32, half, float)
DEFINE_CAST_KERNEL(cast_kernel_bf16_to_f32, bfloat16_t, float)

#undef DEFINE_CAST_KERNEL

extern "C" infiniStatus_t rms_norm_cast_w_launch(
    void *dst, const void *src,
    infiniDtype_t src_dtype, infiniDtype_t dst_dtype,
    size_t count, void *stream) {

    if (dst_dtype != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#define LAUNCH_CAST(DTYPE_ENUM, KERNEL_NAME) \
    case DTYPE_ENUM:                         \
        KERNEL_NAME<<<1, nullptr, stream>>>( \
            dst, (GM_ADDR)src, count);       \
        return INFINI_STATUS_SUCCESS;

    switch (src_dtype) {
        LAUNCH_CAST(INFINI_DTYPE_F16, cast_kernel_f16_to_f32)
        LAUNCH_CAST(INFINI_DTYPE_BF16, cast_kernel_bf16_to_f32)
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH_CAST
}
