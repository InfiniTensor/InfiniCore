#include "../../../devices/ascend/ascend_kernel_common.h"

using namespace AscendC;

template <typename T>
class RandomSampleKernel {
public:
    __aicore__ inline RandomSampleKernel() {}
    __aicore__ inline void init(GM_ADDR probs, GM_ADDR result, GM_ADDR topkValAddr, GM_ADDR topkIdxAddr, float randomVal, float topp, int topk, float temperature, int32_t n);
    __aicore__ inline void process();

private:
    __aicore__ inline void copyIn();
    __aicore__ inline void copyOut();
    __aicore__ inline void compute();
    __aicore__ inline void SoftMax(LocalTensor<T> &topkValIn,
                                   LocalTensor<T> &softMaxOut);
    __aicore__ inline void InclusiveSum(LocalTensor<T> &topkValIn,
                                        LocalTensor<T> &topkValOut);
    __aicore__ inline void RandomSample(LocalTensor<T> &valIn,
                                        LocalTensor<int64_t> &Index,
                                        LocalTensor<int64_t> &result);

    GlobalTensor<T> pGm,
        topkValGm;
    GlobalTensor<int64_t> topkIdxGm, resGm;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> topkValQue;
    TQue<QuePosition::VECIN, 1> topkIdxQue;
    TQue<QuePosition::VECOUT, 1> resQue;

    TBuf<TPosition::VECCALC> inBuf;
    TBuf<TPosition::VECCALC> tmpBuf1;
    TBuf<TPosition::VECCALC> tmpBuf2;
    TBuf<TPosition::VECCALC> tmpBuf3;
    TBuf<TPosition::VECCALC> softMaxOutBuf;
    TBuf<TPosition::VECCALC> inclusiveSumOutBuf;

private:
    int32_t topk_;
    int32_t voc_;
    float random_val_;
    float topp_;
    float invTemp_;
    float negMax = 0.f;
    float sum = 0.f;

    int32_t topkAligned;
    int32_t topkIdxAligned;
    int32_t vocAligned;
    int32_t bufferLen;
};

template <typename T>
__aicore__ inline void RandomSampleKernel<T>::init(GM_ADDR probs, GM_ADDR result, GM_ADDR topkValAddr, GM_ADDR topkIdxAddr, float randomVal, float topp, int topk, float temperature, int32_t n) {
    // AscendC::printf("=============0.0\n");
    topk_ = topk;
    voc_ = n;
    random_val_ = randomVal;
    topp_ = topp;
    invTemp_ = 1.0f / temperature;

    // CumSumInfo
    topkAligned = alignTileLen<T>(topk_, BYTE_ALIGN);
    vocAligned = alignTileLen<T>(voc_, BYTE_ALIGN);
    topkIdxAligned = (topk_ + 3) / 4 * 4;
    bufferLen = topkAligned > BLOCK_LEN ? topkAligned : BLOCK_LEN;
    // AscendC::printf("topkAligned = %d, vocAligned = %d, topkIdxAligned = %d, bufferLen = %d\n", topkAligned, vocAligned, topkIdxAligned, bufferLen);
    // AscendC::printf("=============0.1\n");

    // Set GlobalTensor
    pGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(probs), voc_);
    topkValGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(topkValAddr), topk_);
    topkIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(topkIdxAddr), topk_);
    resGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(result), 1);
    // AscendC::printf("=============0.2\n");

    // Global input and output
    pipe.InitBuffer(topkValQue, 1, topkAligned * sizeof(T));
    pipe.InitBuffer(topkIdxQue, 1, topkIdxAligned * sizeof(int64_t));
    pipe.InitBuffer(resQue, 1, BYTE_ALIGN); // 32 bytes for aligned
    pipe.InitBuffer(inBuf, BLOCK_LEN * sizeof(T));
    pipe.InitBuffer(tmpBuf1, bufferLen * sizeof(T));
    pipe.InitBuffer(tmpBuf2, bufferLen * sizeof(T));
    pipe.InitBuffer(tmpBuf3, bufferLen * sizeof(T));
    pipe.InitBuffer(softMaxOutBuf, topkAligned * sizeof(T));
    pipe.InitBuffer(inclusiveSumOutBuf, topkAligned * sizeof(T));
    // AscendC::printf("=============0.3\n");
}

template <typename T>
__aicore__ inline void RandomSampleKernel<T>::process() {
    // AscendC::printf("=============1.0\n");
    copyIn();
    // AscendC::printf("=============2.0\n");
    compute();
    // AscendC::printf("=============3.0\n");
    copyOut();
    // AscendC::printf("=============4.0\n");
}

template <typename T>
__aicore__ inline void RandomSampleKernel<T>::SoftMax(LocalTensor<T> &topkValIn,
                                                      LocalTensor<T> &softMaxOut) {
    float invSum = 1.0f / sum;
    LocalTensor<T> tmpBuffer = tmpBuf1.Get<T>();
    LocalTensor<T> tmpBuffer2 = tmpBuf2.Get<T>();
    LocalTensor<T> tmpBuffer3 = tmpBuf3.Get<T>();
    Adds(tmpBuffer, topkValIn, static_cast<T>(negMax), topk_);
    Muls(tmpBuffer2, tmpBuffer, static_cast<T>(invTemp_), topk_);
    Exp(tmpBuffer3, tmpBuffer2, topk_);
    Muls(softMaxOut, tmpBuffer3, static_cast<T>(invSum), topk_);
}

template <typename T>
__aicore__ inline void RandomSampleKernel<T>::InclusiveSum(LocalTensor<T> &topkValIn,
                                                           LocalTensor<T> &topkValOut) {
    static constexpr CumSumConfig cumSumConfig{true, false, false};
    LocalTensor<T> lastRowLocal;
    CumSum<T, cumSumConfig>(topkValOut, lastRowLocal, topkValIn,
                            {1, static_cast<uint32_t>(topkAligned)});
}

template <typename T>
__aicore__ inline void RandomSampleKernel<T>::RandomSample(LocalTensor<T> &valIn,
                                                           LocalTensor<int64_t> &Index,
                                                           LocalTensor<int64_t> &result) {
    int end = 0;
    for (end = 0; end < topk_; end++) {
        if (static_cast<float>(valIn(end)) >= topp_) {
            break;
        }
    }
    if (end < topk_ - 1) {
        end += 1;
    } else {
        end = topk_;
    }

    auto randomVal = random_val_ * static_cast<float>(valIn(end - 1));
    for (int i = 0; i < end; i++) {
        if (randomVal < static_cast<float>(valIn(i))) {
            result(0) = Index(i);
            return;
        }
    }
    result(0) = Index(end - 1);
}

template <typename T>
__aicore__ inline void RandomSampleKernel<T>::copyIn() {
    LocalTensor<T> topkValLocal = topkValQue.AllocTensor<T>();
    LocalTensor<int64_t> topkIdxLocal = topkIdxQue.AllocTensor<int64_t>();
    DataCopy(topkValLocal, topkValGm, topkAligned);
    // DumpTensor(topkValLocal, 1, topkAligned);
    DataCopy(topkIdxLocal, topkIdxGm, topkIdxAligned);
    // DumpTensor(topkIdxLocal, 2, topkIdxAligned);
    // Get Max val of input
    negMax = -static_cast<float>(topkValLocal(0));

    // Copy in p and compute sum
    int32_t repeatTimes = voc_ / BLOCK_LEN;
    int32_t remainder = voc_ % BLOCK_LEN;
    float sum_s = 0.f;
    LocalTensor<T> inBuffer = inBuf.Get<T>();
    LocalTensor<T> tmpBuffer = tmpBuf1.Get<T>();
    LocalTensor<T> tmpBuffer2 = tmpBuf2.Get<T>();
    LocalTensor<T> tmpBuffer3 = tmpBuf3.Get<T>();
    for (int32_t i = 0; i < repeatTimes; i++) {
        DataCopy(inBuffer, pGm[i * BLOCK_LEN], BLOCK_LEN);
        Adds(tmpBuffer, inBuffer, static_cast<T>(negMax), BLOCK_LEN);
        Muls(tmpBuffer2, tmpBuffer, static_cast<T>(invTemp_), BLOCK_LEN);
        Exp(tmpBuffer3, tmpBuffer2, BLOCK_LEN);
        sum_s = 0.f;
        for (int j = 0; j < BLOCK_LEN; ++j) {
            sum_s += static_cast<float>(tmpBuffer3(j));
        }
        sum += sum_s;
    }
    if (remainder != 0) {
        int32_t remainderAligned = alignTileLen<T>(remainder, BYTE_ALIGN);
        DataCopy(inBuffer, pGm[repeatTimes * BLOCK_LEN], remainderAligned);
        Adds(tmpBuffer, inBuffer, static_cast<T>(negMax), remainder);
        Muls(tmpBuffer2, tmpBuffer, static_cast<T>(invTemp_), remainder);
        Exp(tmpBuffer3, tmpBuffer2, remainder);
        sum_s = 0.f;
        for (int i = 0; i < remainder; ++i) {
            sum_s += static_cast<float>(tmpBuffer3(i));
        }
        sum += sum_s;
    }

    topkValQue.EnQue(topkValLocal);
    topkIdxQue.EnQue(topkIdxLocal);
}

template <typename T>
__aicore__ inline void RandomSampleKernel<T>::compute() {
    // Get input data
    LocalTensor<T> topkValLocal = topkValQue.DeQue<T>();

    // SoftMax
    LocalTensor<T> softMaxOutLocal = softMaxOutBuf.Get<T>();
    SoftMax(topkValLocal, softMaxOutLocal);
    // DumpTensor(softMaxOutLocal, 3, topkAligned);

    // InclusiveSum
    LocalTensor<T> inclusiveOutLocal = inclusiveSumOutBuf.Get<T>();
    InclusiveSum(softMaxOutLocal, inclusiveOutLocal);
    // DumpTensor(inclusiveOutLocal, 4, topkAligned);

    // randomSample
    LocalTensor<int64_t> topkIdxLocal = topkIdxQue.DeQue<int64_t>();
    LocalTensor<int64_t> resultLocal = resQue.AllocTensor<int64_t>();
    RandomSample(inclusiveOutLocal, topkIdxLocal, resultLocal);
    // DumpTensor(resultLocal, 5, topkIdxAligned);

    topkValQue.FreeTensor(topkValLocal);
    topkIdxQue.FreeTensor(topkIdxLocal);
    resQue.EnQue(resultLocal);
}

template <typename T>
__aicore__ inline void RandomSampleKernel<T>::copyOut() {
    LocalTensor<int64_t> resLocal = resQue.DeQue<int64_t>();
    DataCopy(resGm, resLocal, BYTE_ALIGN / sizeof(int64_t));
    resQue.FreeTensor(resLocal);
}

extern "C" __global__ __aicore__ void random_sample_kernel_fp16(
    GM_ADDR probs,
    GM_ADDR result,
    GM_ADDR topkValAddr,
    GM_ADDR topkIdxAddr,
    float randomVal,
    float topp,
    int topk,
    float temperature,
    int32_t n) {
    RandomSampleKernel<half> op;
    op.init(probs, result, topkValAddr, topkIdxAddr, randomVal, topp, topk, temperature, n);
    op.process();
}

extern "C" __global__ __aicore__ void random_sample_kernel_fp32(
    GM_ADDR probs,
    GM_ADDR result,
    GM_ADDR topkValAddr,
    GM_ADDR topkIdxAddr,
    float randomVal,
    float topp,
    int topk,
    float temperature,
    int32_t n) {
    RandomSampleKernel<float> op;
    op.init(probs, result, topkValAddr, topkIdxAddr, randomVal, topp, topk, temperature, n);
    op.process();
    // AscendC::printf("=============5.0\n");
}

extern "C" infiniStatus_t random_sample_kernel_launch(
    void *probs,
    void *result,
    void *topkValAddr,
    void *topkIdxAddr,
    float randomVal,
    float topp,
    int topk,
    float temperature,
    uint64_t n,
    infiniDtype_t dt_p,
    void *stream) {
    switch (dt_p) {
    case 12:
        random_sample_kernel_fp16<<<1, nullptr, stream>>>(probs, result, topkValAddr, topkIdxAddr, randomVal, topp, topk, temperature, n);
        break;
    case 13:
        random_sample_kernel_fp32<<<1, nullptr, stream>>>(probs, result, topkValAddr, topkIdxAddr, randomVal, topp, topk, temperature, n);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
