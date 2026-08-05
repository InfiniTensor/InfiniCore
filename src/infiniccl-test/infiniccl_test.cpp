#include "infiniccl_test.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <numeric>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <vector>

#define TEST_INFINI(API__) CHECK_API_OR(API__, INFINI_STATUS_SUCCESS, return 1)
#define TEST_INFINI_THREAD(API__) CHECK_API_OR(API__, INFINI_STATUS_SUCCESS, return nullptr)

const size_t MAX_COUNT = 8ULL * 1024 * 1024;
const size_t TEST_COUNTS[] = {
    128,
    1024,
    4 * 1024,
    MAX_COUNT,
};
const infiniDtype_t TEST_DTYPES[] = {INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16};
const size_t WARM_UPS = 10;
const size_t ITERATIONS = 100;

struct ThreadArgs {
    int rank;
    int device_id;
    infinicclComm_t comm;
    infiniDevice_t device_type;
    infiniDtype_t dtype;
    size_t count;
    const void *data;
    const void *ans;
    int *result;
    double *time;
};

struct SendThreadArgs {
    int rank;
    int device_id;
    int peer;
    bool is_sender;
    infinicclComm_t comm;
    infiniDevice_t device_type;
    infiniDtype_t dtype;
    size_t count;
    const void *data;
    const void *ans;
    int *result;
    double *time;
};

struct BroadcastThreadArgs {
    int rank;
    int device_id;
    infinicclComm_t comm;
    infiniDevice_t device_type;
    infiniDtype_t dtype;
    size_t count;
    const void *data;
    const void *ans;
    int *result;
    double *time;
};

void setData(infiniDtype_t dtype, void *data, size_t count, float val) {
    switch (dtype) {
    case INFINI_DTYPE_F32:
        for (size_t i = 0; i < count; i++) {
            ((float *)data)[i] = val;
        }
        break;
    case INFINI_DTYPE_F16:
        for (size_t i = 0; i < count; i++) {
            ((fp16_t *)data)[i] = utils::cast<fp16_t>(val);
        }
        break;
    case INFINI_DTYPE_BF16:
        for (size_t i = 0; i < count; i++) {
            ((bf16_t *)data)[i] = utils::cast<bf16_t>(val);
        }
        break;
    default:
        std::abort();
        break;
    }
}

template <typename T>
int checkData(const T *actual_, const T *expected_, size_t count) {
    int failed = 0;
    for (size_t i = 0; i < count; i++) {
        if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
            float actual = utils::cast<float>(actual_[i]);
            float expected = utils::cast<float>(expected_[i]);
            if (std::abs(actual - expected) > 1e-4f) {
                failed += 1;
            }
        } else {
            if (std::abs(actual_[i] - expected_[i]) > 1e-4f) {
                failed += 1;
            }
        }
    }
    return failed;
}

int checkData(const void *actual, const void *expected, infiniDtype_t dtype, size_t count) {
    switch (dtype) {
    case INFINI_DTYPE_F32:
        return checkData((const float *)actual, (const float *)expected, count);
    case INFINI_DTYPE_F16:
        return checkData((const fp16_t *)actual, (const fp16_t *)expected, count);
    case INFINI_DTYPE_BF16:
        return checkData((const bf16_t *)actual, (const bf16_t *)expected, count);
    default:
        std::abort();
        return 1;
    }
}

double bandwidthGBps(size_t bytes, double ms) {
    if (ms <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(bytes) / ms / 1.0e6;
}

void *testAllReduceThread(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;
    *(args->result) = 1;
    TEST_INFINI_THREAD(infinirtSetDevice(args->device_type, args->device_id));
    infinirtStream_t stream;
    TEST_INFINI_THREAD(infinirtStreamCreate(&stream));
    void *output = std::malloc(args->count * infiniSizeOf(args->dtype));
    std::memset(output, 0, args->count * infiniSizeOf(args->dtype));
    void *buf;
    TEST_INFINI_THREAD(infinirtMalloc(&buf, args->count * infiniSizeOf(args->dtype)));
    TEST_INFINI_THREAD(infinirtMemcpy(buf, args->data, args->count * infiniSizeOf(args->dtype), INFINIRT_MEMCPY_H2D));
    TEST_INFINI_THREAD(infinicclAllReduce(buf, buf, args->count, args->dtype, INFINICCL_SUM, args->comm, stream));
    TEST_INFINI_THREAD(infinirtStreamSynchronize(stream));
    TEST_INFINI_THREAD(infinirtMemcpy(output, buf, args->count * infiniSizeOf(args->dtype), INFINIRT_MEMCPY_D2H));

    if (checkData(output, args->ans, args->dtype, args->count) != 0) {
        std::free(output);
        infinirtFree(buf);
        infinirtStreamDestroy(stream);
        return nullptr;
    }
    for (size_t i = 0; i < WARM_UPS; i++) {
        TEST_INFINI_THREAD(infinicclAllReduce(buf, buf, args->count, args->dtype, INFINICCL_SUM, args->comm, stream));
    }
    TEST_INFINI_THREAD(infinirtStreamSynchronize(stream));

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ITERATIONS; i++) {
        TEST_INFINI_THREAD(infinicclAllReduce(buf, buf, args->count, args->dtype, INFINICCL_SUM, args->comm, stream));
    }
    TEST_INFINI_THREAD(infinirtStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    *args->time = elapsed_ms / ITERATIONS;
    *args->result = 0;

    std::free(output);
    infinirtFree(buf);
    infinirtStreamDestroy(stream);
    return nullptr;
}

int testAllReduce(infiniDevice_t device_type, int ndevice) {
    std::vector<ThreadArgs> thread_args(ndevice);
    std::vector<infinicclComm_t> comms(ndevice);
    std::vector<pthread_t> threads(ndevice);
    std::vector<int> device_ids(ndevice);
    std::vector<int> results(ndevice);
    std::vector<double> times(ndevice);
    void *data = std::malloc(MAX_COUNT * sizeof(float));
    void *ans = std::malloc(MAX_COUNT * sizeof(float));

    for (int i = 0; i < ndevice; i++) {
        device_ids[i] = i;
    }

    for (infiniDtype_t dtype : TEST_DTYPES) {
        setData(dtype, data, MAX_COUNT, 1.0f);
        setData(dtype, ans, MAX_COUNT, 1.0f * ndevice);
        for (size_t count : TEST_COUNTS) {
            TEST_INFINI(infinicclCommInitAll(device_type, comms.data(), ndevice, device_ids.data()));
            std::cout << "Testing AllReduce with " << count << " elements of " << infiniDtypeToString(dtype) << std::endl;
            for (int rank = 0; rank < ndevice; rank++) {
                thread_args[rank] = {rank, device_ids[rank], comms[rank], device_type, dtype, count, data, ans, &results[rank], &times[rank]};
                pthread_create(&threads[rank], NULL, testAllReduceThread, &thread_args[rank]);
            }
            for (int rank = 0; rank < ndevice; rank++) {
                pthread_join(threads[rank], NULL);
            }
            int failed = std::accumulate(results.begin(), results.end(), 0);
            for (int rank = 0; rank < ndevice; rank++) {
                if (results[rank] != 0) {
                    std::cout << "Rank " << rank << ": incorrect results." << std::endl;
                } else {
                    auto bytes = count * infiniSizeOf(dtype);
                    std::cout << "Rank " << rank << ": " << times[rank] << " ms, "
                              << bandwidthGBps(bytes, times[rank]) << " GB/s payload." << std::endl;
                }
                infinicclCommDestroy(comms[rank]);
            }

            if (failed > 0) {
                std::cout << "Failed with " << failed << " errors." << std::endl
                          << std::endl;
                std::free(data);
                std::free(ans);
                return 1;
            }
            std::cout << std::endl;
        }
    }

    std::free(data);
    std::free(ans);
    return 0;
}

void *testBroadcastThread(void *arg) {
    BroadcastThreadArgs *args = (BroadcastThreadArgs *)arg;
    *(args->result) = 1;
    TEST_INFINI_THREAD(infinirtSetDevice(args->device_type, args->device_id));

    infinirtStream_t stream;
    TEST_INFINI_THREAD(infinirtStreamCreate(&stream));

    const size_t bytes = args->count * infiniSizeOf(args->dtype);
    void *buf = nullptr;
    void *output = std::malloc(bytes);
    TEST_INFINI_THREAD(infinirtMalloc(&buf, bytes));
    if (args->rank == 0) {
        TEST_INFINI_THREAD(infinirtMemcpy(buf, args->data, bytes, INFINIRT_MEMCPY_H2D));
    }

    TEST_INFINI_THREAD(infinicclBroadcast(buf, buf, args->count, args->dtype, 0, args->comm, stream));
    TEST_INFINI_THREAD(infinirtStreamSynchronize(stream));
    TEST_INFINI_THREAD(infinirtMemcpy(output, buf, bytes, INFINIRT_MEMCPY_D2H));

    int mismatches = checkData(output, args->ans, args->dtype, args->count);
    if (mismatches != 0) {
        std::cout << "Rank " << args->rank << ": broadcast correctness failed with "
                  << mismatches << " mismatches." << std::endl;
        std::free(output);
        infinirtFree(buf);
        infinirtStreamDestroy(stream);
        return nullptr;
    }

    for (size_t i = 0; i < WARM_UPS; i++) {
        TEST_INFINI_THREAD(infinicclBroadcast(buf, buf, args->count, args->dtype, 0, args->comm, stream));
    }
    TEST_INFINI_THREAD(infinirtStreamSynchronize(stream));

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ITERATIONS; i++) {
        TEST_INFINI_THREAD(infinicclBroadcast(buf, buf, args->count, args->dtype, 0, args->comm, stream));
    }
    TEST_INFINI_THREAD(infinirtStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    *args->time = elapsed_ms / ITERATIONS;
    *args->result = 0;

    std::free(output);
    infinirtFree(buf);
    infinirtStreamDestroy(stream);
    return nullptr;
}

int testBroadcast(infiniDevice_t device_type, int ndevice) {
    std::vector<BroadcastThreadArgs> thread_args(ndevice);
    std::vector<infinicclComm_t> comms(ndevice);
    std::vector<pthread_t> threads(ndevice);
    std::vector<int> device_ids(ndevice);
    std::vector<int> results(ndevice);
    std::vector<double> times(ndevice);
    void *data = std::malloc(MAX_COUNT * sizeof(float));
    void *ans = std::malloc(MAX_COUNT * sizeof(float));

    for (int i = 0; i < ndevice; i++) {
        device_ids[i] = i;
    }

    for (infiniDtype_t dtype : TEST_DTYPES) {
        setData(dtype, data, MAX_COUNT, 9.0f);
        setData(dtype, ans, MAX_COUNT, 9.0f);
        for (size_t count : TEST_COUNTS) {
            TEST_INFINI(infinicclCommInitAll(device_type, comms.data(), ndevice, device_ids.data()));
            std::cout << "Testing Broadcast root rank 0 with " << count << " elements of "
                      << infiniDtypeToString(dtype) << std::endl;

            for (int rank = 0; rank < ndevice; rank++) {
                thread_args[rank] = {rank, device_ids[rank], comms[rank], device_type, dtype, count,
                                     data, ans, &results[rank], &times[rank]};
                pthread_create(&threads[rank], NULL, testBroadcastThread, &thread_args[rank]);
            }
            for (int rank = 0; rank < ndevice; rank++) {
                pthread_join(threads[rank], NULL);
            }

            int failed = std::accumulate(results.begin(), results.end(), 0);
            for (int rank = 0; rank < ndevice; rank++) {
                if (results[rank] != 0) {
                    std::cout << "Rank " << rank << ": failed." << std::endl;
                } else {
                    auto bytes = count * infiniSizeOf(dtype);
                    std::cout << "Rank " << rank << ": " << times[rank] << " ms, "
                              << bandwidthGBps(bytes, times[rank]) << " GB/s payload." << std::endl;
                }
                infinicclCommDestroy(comms[rank]);
            }

            if (failed > 0) {
                std::free(data);
                std::free(ans);
                return 1;
            }
            std::cout << std::endl;
        }
    }

    std::free(data);
    std::free(ans);
    return 0;
}

void *testSendThread(void *arg) {
    SendThreadArgs *args = (SendThreadArgs *)arg;
    *(args->result) = 1;
    TEST_INFINI_THREAD(infinirtSetDevice(args->device_type, args->device_id));

    infinirtStream_t stream;
    TEST_INFINI_THREAD(infinirtStreamCreate(&stream));

    const size_t bytes = args->count * infiniSizeOf(args->dtype);
    void *buf = nullptr;
    void *output = std::malloc(bytes);
    TEST_INFINI_THREAD(infinirtMalloc(&buf, bytes));
    if (args->is_sender) {
        TEST_INFINI_THREAD(infinirtMemcpy(buf, args->data, bytes, INFINIRT_MEMCPY_H2D));
    }

    if (args->is_sender) {
        TEST_INFINI_THREAD(infinicclSend(buf, args->count, args->dtype, args->peer, args->comm, stream));
    } else {
        TEST_INFINI_THREAD(infinicclRecv(buf, args->count, args->dtype, args->peer, args->comm, stream));
    }
    TEST_INFINI_THREAD(infinirtStreamSynchronize(stream));

    int mismatches = 0;
    if (!args->is_sender) {
        TEST_INFINI_THREAD(infinirtMemcpy(output, buf, bytes, INFINIRT_MEMCPY_D2H));
        mismatches = checkData(output, args->ans, args->dtype, args->count);
        if (mismatches != 0) {
            std::cout << "Rank " << args->rank << ": send/recv correctness failed with "
                      << mismatches << " mismatches." << std::endl;
        }
    }

    for (size_t i = 0; i < WARM_UPS; i++) {
        if (args->is_sender) {
            TEST_INFINI_THREAD(infinicclSend(buf, args->count, args->dtype, args->peer, args->comm, stream));
        } else {
            TEST_INFINI_THREAD(infinicclRecv(buf, args->count, args->dtype, args->peer, args->comm, stream));
        }
    }
    TEST_INFINI_THREAD(infinirtStreamSynchronize(stream));

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ITERATIONS; i++) {
        if (args->is_sender) {
            TEST_INFINI_THREAD(infinicclSend(buf, args->count, args->dtype, args->peer, args->comm, stream));
        } else {
            TEST_INFINI_THREAD(infinicclRecv(buf, args->count, args->dtype, args->peer, args->comm, stream));
        }
    }
    TEST_INFINI_THREAD(infinirtStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    *args->time = elapsed_ms / ITERATIONS;
    *args->result = mismatches == 0 ? 0 : 1;

    std::free(output);
    infinirtFree(buf);
    infinirtStreamDestroy(stream);
    return nullptr;
}

int testSend(infiniDevice_t device_type, int ndevice, bool send_from_zero) {
    if (ndevice < 2) {
        std::cout << "Send/recv test requires at least 2 visible devices." << std::endl;
        return 1;
    }

    std::vector<int> device_ids = {0, 1};
    std::vector<infinicclComm_t> comms(2);
    std::vector<pthread_t> threads(2);
    std::vector<int> results(2);
    std::vector<double> times(2);
    std::vector<SendThreadArgs> thread_args(2);
    const int src_rank = send_from_zero ? 0 : 1;
    const int dst_rank = send_from_zero ? 1 : 0;

    for (infiniDtype_t dtype : TEST_DTYPES) {
        for (size_t count : TEST_COUNTS) {
            const size_t bytes = count * infiniSizeOf(dtype);
            void *data = std::malloc(bytes);
            void *ans = std::malloc(bytes);
            setData(dtype, data, count, 7.0f);
            setData(dtype, ans, count, 7.0f);

            TEST_INFINI(infinicclCommInitAll(device_type, comms.data(), 2, device_ids.data()));
            std::cout << "Testing Send rank " << src_rank << " -> rank " << dst_rank
                      << " with " << count << " elements of " << infiniDtypeToString(dtype) << std::endl;

            thread_args[0] = {0, device_ids[0], 1, send_from_zero, comms[0], device_type, dtype, count, data, ans, &results[0], &times[0]};
            thread_args[1] = {1, device_ids[1], 0, !send_from_zero, comms[1], device_type, dtype, count, data, ans, &results[1], &times[1]};
            pthread_create(&threads[0], NULL, testSendThread, &thread_args[0]);
            pthread_create(&threads[1], NULL, testSendThread, &thread_args[1]);
            pthread_join(threads[0], NULL);
            pthread_join(threads[1], NULL);

            int failed = results[0] + results[1];
            for (int rank = 0; rank < 2; rank++) {
                if (results[rank] != 0) {
                    std::cout << "Rank " << rank << ": failed." << std::endl;
                } else {
                    std::cout << "Rank " << rank << ": " << times[rank] << " ms, "
                              << bandwidthGBps(bytes, times[rank]) << " GB/s payload." << std::endl;
                }
                infinicclCommDestroy(comms[rank]);
            }

            std::free(data);
            std::free(ans);
            if (failed > 0) {
                return 1;
            }
            std::cout << std::endl;
        }
    }
    return 0;
}

namespace {

void throwErrno(const char *what) {
    throw std::runtime_error(std::string(what) + ": " + std::strerror(errno));
}

void sendAll(int fd, const void *data, size_t size) {
    const char *ptr = static_cast<const char *>(data);
    while (size > 0) {
        ssize_t n = send(fd, ptr, size, 0);
        if (n <= 0) {
            throwErrno("send");
        }
        ptr += n;
        size -= static_cast<size_t>(n);
    }
}

void recvAll(int fd, void *data, size_t size) {
    char *ptr = static_cast<char *>(data);
    while (size > 0) {
        ssize_t n = recv(fd, ptr, size, 0);
        if (n <= 0) {
            throwErrno("recv");
        }
        ptr += n;
        size -= static_cast<size_t>(n);
    }
}

void exchangeUniqueId(
    int rank,
    int world_size,
    const char *master_addr,
    int master_port,
    infinicclUniqueId_t *unique_id) {
    if (rank == 0) {
        int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd < 0) {
            throwErrno("socket");
        }
        int yes = 1;
        setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(static_cast<uint16_t>(master_port));
        if (bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
            close(listen_fd);
            throwErrno("bind");
        }
        if (listen(listen_fd, world_size - 1) != 0) {
            close(listen_fd);
            throwErrno("listen");
        }

        for (int i = 1; i < world_size; ++i) {
            int client_fd = accept(listen_fd, nullptr, nullptr);
            if (client_fd < 0) {
                close(listen_fd);
                throwErrno("accept");
            }
            sendAll(client_fd, unique_id, sizeof(*unique_id));
            close(client_fd);
        }
        close(listen_fd);
    } else {
        int fd = -1;
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(master_port));
        if (inet_pton(AF_INET, master_addr, &addr.sin_addr) != 1) {
            throw std::runtime_error("master-addr must be an IPv4 address");
        }

        for (int attempt = 0; attempt < 300; ++attempt) {
            fd = socket(AF_INET, SOCK_STREAM, 0);
            if (fd < 0) {
                throwErrno("socket");
            }
            if (connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
                break;
            }
            close(fd);
            fd = -1;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (fd < 0) {
            throw std::runtime_error("failed to connect to rank 0 unique-id server");
        }
        recvAll(fd, unique_id, sizeof(*unique_id));
        close(fd);
    }
}

} // namespace

int testMultiNodeSend(
    infiniDevice_t device_type,
    bool is_sender,
    const char *master_addr,
    int master_port) {
    try {
        const int world_size = 2;
        const bool is_master = master_addr == nullptr;
        const int rank = is_master ? 0 : 1;
        const int peer = 1 - rank;
        const int device_id = 0;
        const char *connect_addr = is_master ? "0.0.0.0" : master_addr;

        TEST_INFINI(infinirtSetDevice(device_type, device_id));

        infinicclUniqueId_t unique_id;
        if (rank == 0) {
            TEST_INFINI(infinicclGetUniqueId(&unique_id));
        }

        std::cout << "Global rank " << rank << "/" << world_size
                  << " exchanging unique id via " << connect_addr << ":" << master_port << std::endl;
        exchangeUniqueId(rank, world_size, connect_addr, master_port, &unique_id);

        infinicclComm_t comm = nullptr;
        TEST_INFINI(infinicclCommInitRank(&comm, world_size, unique_id, rank));

        infinirtStream_t stream;
        TEST_INFINI(infinirtStreamCreate(&stream));

        int failed = 0;
        for (infiniDtype_t dtype : TEST_DTYPES) {
            for (size_t count : TEST_COUNTS) {
                const size_t bytes = count * infiniSizeOf(dtype);
                void *data = std::malloc(bytes);
                void *ans = std::malloc(bytes);
                void *output = std::malloc(bytes);
                void *buf = nullptr;
                setData(dtype, data, count, 7.0f);
                setData(dtype, ans, count, 7.0f);
                std::memset(output, 0, bytes);

                TEST_INFINI(infinirtMalloc(&buf, bytes));
                if (is_sender) {
                    TEST_INFINI(infinirtMemcpy(buf, data, bytes, INFINIRT_MEMCPY_H2D));
                }

                if (is_sender) {
                    TEST_INFINI(infinicclSend(buf, count, dtype, peer, comm, stream));
                } else {
                    TEST_INFINI(infinicclRecv(buf, count, dtype, peer, comm, stream));
                }
                TEST_INFINI(infinirtStreamSynchronize(stream));

                int mismatches = 0;
                if (!is_sender) {
                    TEST_INFINI(infinirtMemcpy(output, buf, bytes, INFINIRT_MEMCPY_D2H));
                    mismatches = checkData(output, ans, dtype, count);
                    if (mismatches != 0) {
                        std::cout << "Global rank " << rank << " " << infiniDtypeToString(dtype)
                                  << " count=" << count << " failed with " << mismatches
                                  << " mismatches." << std::endl;
                    }
                }

                for (size_t i = 0; i < WARM_UPS; i++) {
                    if (is_sender) {
                        TEST_INFINI(infinicclSend(buf, count, dtype, peer, comm, stream));
                    } else {
                        TEST_INFINI(infinicclRecv(buf, count, dtype, peer, comm, stream));
                    }
                }
                TEST_INFINI(infinirtStreamSynchronize(stream));

                auto start = std::chrono::high_resolution_clock::now();
                for (size_t i = 0; i < ITERATIONS; i++) {
                    if (is_sender) {
                        TEST_INFINI(infinicclSend(buf, count, dtype, peer, comm, stream));
                    } else {
                        TEST_INFINI(infinicclRecv(buf, count, dtype, peer, comm, stream));
                    }
                }
                TEST_INFINI(infinirtStreamSynchronize(stream));
                auto end = std::chrono::high_resolution_clock::now();
                double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
                double avg_ms = elapsed_ms / ITERATIONS;

                infinirtFree(buf);
                std::free(data);
                std::free(ans);
                std::free(output);

                if (mismatches != 0) {
                    failed = 1;
                }
                std::cout << "Global rank " << rank << " " << (is_sender ? "send" : "recv")
                          << " " << infiniDtypeToString(dtype) << " count=" << count
                          << ", avg=" << avg_ms << " ms, " << bandwidthGBps(bytes, avg_ms)
                          << " GB/s payload." << std::endl;
            }
        }

        infinirtStreamDestroy(stream);
        infinicclCommDestroy(comm);
        return failed;
    } catch (const std::exception &e) {
        std::cout << "Global rank " << (is_sender ? 0 : 1) << " failed: " << e.what() << std::endl;
        return 1;
    }
}
