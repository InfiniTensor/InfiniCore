// #ifndef TOPK_RADIXSELECT_TOPK_RADIXSELECT_H_
// #define TOPK_RADIXSELECT_TOPK_RADIXSELECT_H_

// #include <algorithm>
// #include <cmath>
// #include <vector>

// #include <thrust/functional.h>
// #include <thrust/sort.h>

// #include "radixselect_l.cuh"
// #include "bitonic_sort.cuh"
// #include "utils.cuh"

// #include "error_check.hpp"

// static constexpr int MAX_GRID_SIZE = 1280;

// /* Computes workspace size in bytes.
//  * When taskNum > 1, n is length of the longest task.
//  */
// template <typename ValType>
// void getRadixSelectLWorkSpaceSize(const int &K,
//                                   const int &n,
//                                   const int &taskNum,
//                                   size_t *sizeInBytes) {
//     using CompT = typename radix_topk::ComputeT<ValType>::type;
//     *sizeInBytes = taskNum * (sizeof(int) * (1 << 12)   /* buffer for hist */
//                            + sizeof(CompT) * n * 2      /* buffer for val */
//                            + sizeof(int) * 5)           /* buffer for K taskLen(old & new) binId and globalCount */
//                  + sizeof(int) * (taskNum + 1);         /* buffer for taskOffset */
//     return;
// }

// /**
//  * @brief Entry function of radix top-k selection.
//  *
//  * @tparam IdxType index type
//  * @tparam LARGEST true: largest k; false: smallest k
//  * @tparam ASCEND true: sorted ascend; false: sorted descend
//  * @tparam WITHSCALE true: scaling; false: no scaling
//  * @tparam WITHIDXIN true: input indices; false: no input indices (default:
//  * false)
//  * @tparam ValType value type
//  *
//  * @param[in] valIn input value ptr
//  * @param[in] idxIn input indices ptr
//  * @param[out] valOut output value ptr
//  * @param[out] idxOut output indices ptr
//  * @param[in] workSpace workspace ptr
//  * @param[in] taskLen array of task lengths
//  * @param[in] taskNum batch size
//  * @param[in] K k for top-k
//  * @param[in] stream CUDA stream
//  *
//  * @throws @c std::runtime_error if @p workSpace is @c nullptr .
//  * @throws @c std::runtime_error if CUDA error occurs.
//  */
// template <typename IdxType, bool LARGEST, bool ASCEND, bool WITHSCALE, bool WITHIDXIN = 0,
//           typename ValType, bool WITHPACKING = 1>
// void topKRadixSelectL(const ValType *valIn,
//                       const IdxType *idxIn,
//                       ValType *valOut,
//                       IdxType *idxOut,
//                       void *workSpace,
//                       const int *taskLen,
//                       const int &taskNum,
//                       const int &K,
//                       cudaStream_t stream,
//                       bool needSorting = true) {
//     using CompT = typename radix_topk::ComputeT<ValType>::type;

//     static_assert(!WITHPACKING || (WITHPACKING && sizeof(ValType) <= 16),
//                   "radix topk: requires sizeof(ValType) <= 16");
//     constexpr int PACKSIZE = WITHPACKING ? 16 / sizeof(ValType) : 1;

//     if (workSpace == nullptr) {
//         throw std::runtime_error("radix topk: workspace should not be null");
//     }

//     const int stride = *std::max_element(taskLen, taskLen + taskNum);
//     // set valBuffer histPtr globalCountPtr
//     CompT *valBuffer[2]{static_cast<CompT *>(workSpace),
//                         static_cast<CompT *>(workSpace) + taskNum * stride};

//     int *histPtr = reinterpret_cast<int *>(valBuffer[1] + taskNum * stride);
//     int *globalCountPtr = histPtr + (1 << 12) * taskNum;

//     // set taskOffsetPtr
//     int *taskOffsetPtr = globalCountPtr + taskNum;
//     std::vector<int> taskOffset(taskNum + 1);
//     taskOffset[0] = 0;
//     for (int i = 0; i < taskNum; ++i) {
//         taskOffset[i + 1] = taskOffset[i] + taskLen[i];
//     }
//     CUDA_CHECK(cudaMemcpyAsync(taskOffsetPtr, taskOffset.data(),
//                                sizeof(int) * (taskNum + 1), cudaMemcpyDefault,
//                                stream));

//     // set taskLenPtr, zero-init
//     int *taskLenPtr[2]{taskOffsetPtr + taskNum + 1,
//                        taskOffsetPtr + 2 * taskNum + 1};
//     CUDA_CHECK(cudaMemsetAsync(taskLenPtr[0], 0, sizeof(int) * taskNum * 2, stream));

//     // set kPtr
//     int *kPtr = taskLenPtr[1] + taskNum;
//     std::vector<int> tmpK(taskNum);
//     for (int i = 0; i < taskNum; ++i) {
//         tmpK[i] = K;
//     }
//     CUDA_CHECK(cudaMemcpyAsync(kPtr, tmpK.data(), sizeof(int) * taskNum,
//                                cudaMemcpyDefault, stream));

//     // set binIdPtr
//     int *binIdPtr = kPtr + taskNum;

//     const int minTaskLen = *std::min_element(taskLen, taskLen + taskNum);
//     int gridSizeX = std::min(MAX_GRID_SIZE, std::max(1, minTaskLen / (1024 * PACKSIZE)));
//     int gridSizeY = std::min(MAX_GRID_SIZE / gridSizeX, taskNum);

//     std::vector<int> taskLenHost(taskNum);

//     // clear hist and globalCount
//     CUDA_CHECK(cudaMemsetAsync(histPtr, 0, sizeof(int) * ((1 << 12) + 1) * taskNum, stream));

//     // first iter
//     // get hist
//     radix_topk::countBinEx<1024, 0, 20, PACKSIZE, WITHSCALE, LARGEST>
//         <<<dim3(gridSizeX, gridSizeY), 1024, 0, stream>>>(
//             valIn, taskOffsetPtr, histPtr, taskNum);
//     KERNEL_CHECK();
//     // KERNEL_CHECK_SYNC(stream);

//     // select bin
//     radix_topk::selectBin<LARGEST, 1024, (1 << 12)>
//         <<<taskNum, 1024, 0, stream>>>(
//             histPtr, binIdPtr, kPtr, taskLenPtr[0]);
//     KERNEL_CHECK();
//     // KERNEL_CHECK_SYNC(stream);

//     // select candidate
//     gridSizeX = std::min(MAX_GRID_SIZE, std::max(1, minTaskLen / (256 * PACKSIZE)));
//     gridSizeY = std::min(MAX_GRID_SIZE / gridSizeX, taskNum);
//     radix_topk::selectCandidateEx<256, 0, 20, PACKSIZE, WITHSCALE, LARGEST>
//         <<<dim3(gridSizeX, gridSizeY), 256, 0, stream>>>(
//             valIn, valBuffer[0], globalCountPtr, binIdPtr, taskOffsetPtr, stride, taskNum);
//     KERNEL_CHECK();
//     // KERNEL_CHECK_SYNC(stream);

//     // update taskLen
//     CUDA_CHECK(cudaMemcpyAsync(taskLenHost.data(), taskLenPtr[0],
//                                sizeof(int) * taskNum, cudaMemcpyDefault,
//                                stream));

//     int flag = 0;
//     // second iter
//     CUDA_CHECK(cudaStreamSynchronize(stream));
//     int maxTaskLen = *std::max_element(taskLenHost.begin(), taskLenHost.end());
//     if (maxTaskLen != 1) {
//         // clear hist and globalCount
//         int gridSize = (maxTaskLen + 1023) / 1024;
//         CUDA_CHECK(cudaMemsetAsync(histPtr, 0, sizeof(int) * ((1 << 12) + 1) * taskNum, stream));

//         // get hist
//         radix_topk::countBin<1024, 12, 20>
//             <<<dim3(gridSize, taskNum), 1024, 0, stream>>>(
//                 valBuffer[flag], taskLenPtr[flag], histPtr, stride);
//         KERNEL_CHECK();
//         // KERNEL_CHECK_SYNC(stream);

//         // select bin
//         radix_topk::selectBin<LARGEST, 1024, (1 << 12)>
//             <<<taskNum, 1024, 0, stream>>>(
//                 histPtr, binIdPtr, kPtr, taskLenPtr[flag ^ 1]);
//         KERNEL_CHECK();
//         // KERNEL_CHECK_SYNC(stream);

//         // sift candidate element
//         gridSize = (maxTaskLen + 255) / 256;
//         radix_topk::selectCandidate<256, 12, 20>
//             <<<dim3(gridSize, taskNum), 256, 0, stream>>>(
//                 valBuffer[flag], valBuffer[flag ^ 1], globalCountPtr, binIdPtr, taskLenPtr[flag], stride);
//         KERNEL_CHECK();
//         // KERNEL_CHECK_SYNC(stream);

//         // update taskLen
//         CUDA_CHECK(cudaMemcpyAsync(taskLenHost.data(), taskLenPtr[flag ^ 1],
//                                    sizeof(int) * taskNum, cudaMemcpyDefault,
//                                    stream));

//         flag ^= 1;
//     }

//     // third iter
//     CUDA_CHECK(cudaStreamSynchronize(stream));
//     maxTaskLen = *std::max_element(taskLenHost.begin(), taskLenHost.end());
//     if (maxTaskLen != 1) {
//         int gridSize = (maxTaskLen + 255) / 256;
//         // clear hist and globalCount
//         CUDA_CHECK(cudaMemsetAsync(histPtr + 3840 * taskNum, 0,
//                                    sizeof(int) * ((1 << 8) + 1) * taskNum,
//                                    stream));

//         // get hist
//         radix_topk::countBin<256, 24, 24>
//             <<<dim3(gridSize, taskNum), 256, 0, stream>>>(
//                 valBuffer[flag], taskLenPtr[flag], histPtr + 3840 * taskNum, stride);
//         KERNEL_CHECK();
//         // KERNEL_CHECK_SYNC(stream);

//         // select bin
//         radix_topk::selectBin<LARGEST, 256, (1 << 8)>
//             <<<taskNum, 256, 0, stream>>>(
//                 histPtr + 3840 * taskNum, binIdPtr, kPtr, taskLenPtr[flag ^ 1]);
//         KERNEL_CHECK();
//         // KERNEL_CHECK_SYNC(stream);

//         // sift candidate element
//         gridSize = (maxTaskLen + 255) / 256;
//         radix_topk::selectCandidate<256, 24, 24>
//             <<<dim3(gridSize, taskNum), 256, 0, stream>>>(
//                 valBuffer[flag], valBuffer[flag ^ 1], globalCountPtr, binIdPtr, taskLenPtr[flag], stride);
//         KERNEL_CHECK();
//         // KERNEL_CHECK_SYNC(stream);

//         flag ^= 1;
//     }

//     // clear globalCount
//     CUDA_CHECK(cudaMemsetAsync(globalCountPtr, 0, sizeof(int) * taskNum, stream));

//     // select result
//     gridSizeX = std::min(MAX_GRID_SIZE, std::max(1, minTaskLen / (256 * PACKSIZE)));
//     gridSizeY = std::min(MAX_GRID_SIZE / gridSizeX, taskNum);

// #define RADIX_TOPK_CALL_FILTER(CACHE_SIZE)                                          \
// do {                                                                                \
//     radix_topk::filter<LARGEST, 256, PACKSIZE, (CACHE_SIZE), WITHSCALE, WITHIDXIN>  \
//         <<<dim3(gridSizeX, gridSizeY), 256, 0, stream>>>(                           \
//             valIn, idxIn, valBuffer[flag], valOut, idxOut, globalCountPtr,          \
//             kPtr, taskOffsetPtr, stride, K, taskNum);                               \
// } while (0)

//     if (K <= 128) {
//         RADIX_TOPK_CALL_FILTER(128);
//     } else if (K <= 256) {
//         RADIX_TOPK_CALL_FILTER(256);
//     } else if (K <= 512) {
//         RADIX_TOPK_CALL_FILTER(512);
//     } else if (K <= 1024) {
//         RADIX_TOPK_CALL_FILTER(1024);
//     } else {
//         // for K > 1024, use a general filter
//         radix_topk::filter_general<LARGEST, 256, PACKSIZE, WITHSCALE, WITHIDXIN>
//             <<<dim3(gridSizeX, gridSizeY), 256, 0, stream>>>(
//                 valIn, idxIn, valBuffer[flag], valOut, idxOut, globalCountPtr,
//                 kPtr, taskOffsetPtr, stride, K, taskNum);
//     }
// #undef RADIX_TOPK_CALL_FILTER
//     KERNEL_CHECK();
//     // KERNEL_CHECK_SYNC(stream);

//     if (!needSorting) {
//         return;
//     }

// #define RADIX_TOPK_CALL_BITONIC_SORT(LENGTH, BLOCK)                             \
// do {                                                                            \
//     radix_topk::bitonic::bitonicSort_##LENGTH<ASCEND>                           \
//         <<<taskNum, (BLOCK), 0, stream>>>(valOut, idxOut, K, taskOffsetPtr);    \
// } while (0)

//     if (K <= 128) {
//         RADIX_TOPK_CALL_BITONIC_SORT(128, 128);
//     } else if (K <= 256) {
//         RADIX_TOPK_CALL_BITONIC_SORT(256, 256);
//     } else if (K <= 512) {
//         RADIX_TOPK_CALL_BITONIC_SORT(512, 512);
//     } else if (K <= 1024) {
//         RADIX_TOPK_CALL_BITONIC_SORT(1024, 1024);
//     } else if (K <= 2048) {
//         RADIX_TOPK_CALL_BITONIC_SORT(2048, 1024);
//     } else if (K <= 4096) {
//         RADIX_TOPK_CALL_BITONIC_SORT(4096, 1024);
//     } else {
//         // sort outputs with thrust::sort
// #ifdef CONFIG_RADIX_TOPK_ENABLE_NAN_FILTER
// #ifdef CONFIG_RADIX_TOPK_ENFORCE_NAN_ORDER_GT_4096
//         // preprocessing to ensure order
//         uint32_t nPass = (K + 1024 * PACKSIZE - 1) / (1024 * PACKSIZE);
//         radix_topk::convertNanInPlace<1024, PACKSIZE, ASCEND>
//             <<<dim3(taskNum, nPass), 1024, 0, stream>>>(valOut, K);
//         KERNEL_CHECK();
// #endif  // CONFIG_RADIX_TOPK_ENFORCE_NAN_ORDER_GT_4096
// #endif  // CONFIG_RADIX_TOPK_ENABLE_NAN_FILTER
//         for (int i = 0; i < taskNum; ++i) {
//             if (ASCEND) {
//                 thrust::sort_by_key(thrust::cuda::par.on(stream),
//                                     valOut + i * K, valOut + (i + 1) * K,
//                                     idxOut + i * K, thrust::less<ValType>());
//             } else {
//                 thrust::sort_by_key(thrust::cuda::par.on(stream),
//                                     valOut + i * K, valOut + (i + 1) * K,
//                                     idxOut + i * K, thrust::greater<ValType>());
//             }
//         }
//     }
// #undef RADIX_TOPK_CALL_BITONIC_SORT
//     KERNEL_CHECK();
//     // KERNEL_CHECK_SYNC(stream);

//     // set idle positions to default values (index: -1, value: 0)
//     for (int i = 0; i < taskNum; ++i) {
//         if (taskLen[i] < K) {
//             CUDA_CHECK(cudaMemsetAsync(idxOut + i * K + taskLen[i], -1,
//                                        (K - taskLen[i]) * sizeof(IdxType), stream));
//             CUDA_CHECK(cudaMemsetAsync(valOut + i * K + taskLen[i], 0,
//                                        (K - taskLen[i]) * sizeof(ValType), stream));
//         }
//     }

//     return;
// }

// #endif  // TOPK_RADIXSELECT_TOPK_RADIXSELECT_H_