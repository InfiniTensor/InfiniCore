// Public graph/slot-binding regression. Compile as a shared test fixture and
// call check_paged_cache_graph(device) after initializing the Ascend runtime.
#include <algorithm>
#include <cstring>
#include <infinicore/context/context.hpp>
#include <infinicore/ops/paged_caching.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace infinicore;
extern "C" int check_paged_cache_shape(int device, size_t T, bool strided) {
    try {
        const Device dev(Device::Type::ASCEND, device);
        context::setDevice(dev);
        constexpr size_t H = 16, D = 128, BS = 256, NB = 4;
        auto k = Tensor::empty({T, H, D}, DataType::BF16, dev);
        auto v = Tensor::empty(k->shape(), DataType::BF16, dev);
        if (strided) {
            k = Tensor::empty({T, H * 3, D}, DataType::BF16, dev)->narrow({{1, H, H}});
            v = Tensor::empty({T, H * 3, D}, DataType::BF16, dev)->narrow({{1, H, H}});
        }
        auto kc = Tensor::empty({NB, BS, H, D}, DataType::BF16, dev);
        auto vc = Tensor::empty(kc->shape(), DataType::BF16, dev);
        auto slots = Tensor::empty({T}, DataType::I64, dev);
        std::vector<uint16_t> kh(k->numel()), vh(k->numel());
        for (size_t i = 0; i < kh.size(); ++i) {
            kh[i] = 0x3f00 + i % 100;
            vh[i] = 0x4000 + i % 100;
        }
        std::vector<uint16_t> expected_k(kc->numel(), 0x42f0), expected_v = expected_k, got = expected_k;
        auto contiguous_k = Tensor::empty(k->shape(), DataType::BF16, dev);
        auto contiguous_v = Tensor::empty(v->shape(), DataType::BF16, dev);
        context::memcpyH2D(contiguous_k->data(), kh.data(), kh.size() * 2, false);
        context::memcpyH2D(contiguous_v->data(), vh.data(), vh.size() * 2, false);
        k->copy_from(contiguous_k);
        v->copy_from(contiguous_v);
        context::memcpyH2D(kc->data(), expected_k.data(), expected_k.size() * 2, false);
        context::memcpyH2D(vc->data(), expected_v.data(), expected_v.size() * 2, false);
        std::vector<int64_t> host_slots(T);
        auto bind = [&] {
            context::memcpyH2D(slots->data(), host_slots.data(), T * 8, false);
            graph::bind_eager_host_int_array(slots, host_slots.data(), T);
        };
        auto verify = [&] {
            context::syncStream();
            for (int which = 0; which < 2; ++which) {
                context::memcpyD2H(got.data(), (which ? vc : kc)->data(), got.size() * 2);
                if (got != (which ? expected_v : expected_k)) {
                    throw std::runtime_error("full cache differs from CPU golden");
                }
            }
        };
        auto update_expected = [&] {
            for (size_t i = 0; i < T; ++i) {
                std::copy_n(kh.data() + i * H * D, H * D, expected_k.data() + host_slots[i] * H * D);
                std::copy_n(vh.data() + i * H * D, H * D, expected_v.data() + host_slots[i] * H * D);
            }
        };
        for (size_t i = 0; i < T; ++i) { host_slots[i] = i; }
        bind();
        context::startGraphRecording();
        op::paged_caching_(kc->permute({0, 2, 1, 3}), vc->permute({0, 2, 1, 3}), k, v, slots);
        auto graph = context::stopGraphRecording();
        graph->run();
        update_expected();
        verify();
        for (int invalid = 0; invalid < 5; ++invalid) {
            for (size_t i = 0; i < T; ++i) { host_slots[i] = 100 + i; }
            if (invalid == 0) {
                host_slots[1] = host_slots[0];
            }
            if (invalid == 1) {
                host_slots[0] = -1;
            }
            if (invalid == 2) {
                host_slots[0] = NB * BS;
            }
            bind();
            if (invalid == 3) {
                graph::clear_eager_host_int_arrays();
            }
            if (invalid == 4) {
                graph::bind_eager_host_int_array(slots, host_slots.data(), T - 1);
            }
            bool rejected = false;
            try {
                graph->run();
            } catch (const std::runtime_error &error) {
                rejected = std::string(error.what()).find("PagedCaching graph requires") != std::string::npos;
            }
            if (!rejected) {
                throw std::runtime_error("invalid replay was not rejected before launch");
            }
            verify();
        }
        for (int repetition = 0; repetition < 3; ++repetition) {
            for (size_t i = 0; i < T; ++i) { host_slots[i] = 256 * (repetition + 1) + T - 1 - i; }
            bind();
            graph->run();
            update_expected();
            verify();
        }
        graph.reset();
        graph::clear_eager_host_int_arrays();
        std::cout << "PASS PAGED_CACHE_GRAPH device=" << device
                  << " rejected_replays=5 valid_changed_slots=4 full_cache_cpu_exact=1" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL PAGED_CACHE_GRAPH " << error.what() << std::endl;
        return 1;
    }
}

extern "C" int check_paged_cache_graph(int device) { return check_paged_cache_shape(device, 41, false); }
