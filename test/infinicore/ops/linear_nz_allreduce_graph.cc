#include <acl/acl.h>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <infiniccl.h>
#include <infinicore/context/context.hpp>
#include <infinicore/ops/ascend_format_cast.hpp>
#include <infinicore/ops/distributed/allreduce.hpp>
#include <infinicore/ops/gemm.hpp>
#include <infiniop.h>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
using namespace infinicore;
struct Barrier {
    std::mutex m;
    std::condition_variable cv;
    int n = 0, g = 0;
    void wait() {
        std::unique_lock<std::mutex> l(m);
        int old = g;
        if (++n == 4) {
            n = 0;
            ++g;
            cv.notify_all();
        } else {
            cv.wait(l, [&] { return g != old; });
        }
    }
};
static void ck(int x, const char *s) {
    if (x) {
        throw std::runtime_error(std::string(s) + ":" + std::to_string(x));
    }
}
static uint16_t bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return u >> 16;
}
static float fp(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
#include <infinicore/ops/linear_allreduce.hpp>

extern "C" int check_pipelined_linear(int M, int K, int N) {
    context::setDevice(Device(Device::Type::ASCEND, 0));
    infinicclComm_t comms[4]{};
    int ids[] = {0, 1, 2, 3};
    ck(infinicclCommInitAll(INFINI_DEVICE_ASCEND, comms, 4, ids), "init");
    Barrier barrier;
    std::array<int, 4> errors{};
    std::vector<std::thread> threads;
    for (int rank = 0; rank < 4; ++rank) {
        threads.emplace_back([&, rank] {try{Device dev(Device::Type::ASCEND,rank);context::setDevice(dev);auto a=Tensor::empty({size_t(M),size_t(K)},DataType::BF16,dev);std::vector<uint16_t>host(a->numel()),wh(size_t(K)*N);std::vector<Tensor>weights,biases;
 for(int l=0;l<4;++l){for(int k=0;k<K;++k){for(int n=0;n<N;++n){wh[size_t(k)*N+n]=bf(float((k%2)?-1:1)*(k%4+1)/16*float(n%13-6)/128*(rank+1));
}}auto nd=Tensor::empty({size_t(K),size_t(N)},DataType::BF16,dev);context::memcpyH2D(nd->data(),wh.data(),wh.size()*2,false);weights.push_back(op::ascend_format_cast_nz(nd));auto bias=Tensor::empty({size_t(N)},DataType::BF16,dev);std::vector<uint16_t>bh(N);for(int n=0;n<N;++n){bh[n]=bf(float(n%17-8)/64);
}context::memcpyH2D(bias->data(),bh.data(),bh.size()*2,false);biases.push_back(bias->as_strided({size_t(M), size_t(N)}, {0, 1}));}
 auto fill=[&](int phase){for(int m=0;m<M;++m){for(int k=0;k<K;++k){host[size_t(m)*K+k]=bf(float(m%7+phase+1)/64*float((k%2)?-1:1)*(k%4+1)/16);
}}context::memcpyH2D(a->data(),host.data(),host.size()*2,false);};fill(0);barrier.wait();std::vector<Tensor>out;
 context::startGraphRecording();for(int l=0;l<4;++l){out.push_back(op::linear_allreduce_packed_nz(a,weights[l],l%2?std::optional<Tensor>(biases[l]):std::nullopt,comms[rank]));
}auto graph=context::stopGraphRecording();barrier.wait();
 float worst=0;for(int phase=0;phase<5;++phase){fill(phase);barrier.wait();graph->run();context::syncStream();std::vector<std::vector<uint16_t>>actual;for(auto o:out){actual.emplace_back(o->numel());context::memcpyD2H(actual.back().data(),o->data(),o->nbytes());}barrier.wait();for(int l=0;l<4;++l){auto expected=op::linear_allreduce_packed_nz(a,weights[l],l%2?std::optional<Tensor>(biases[l]):std::nullopt,comms[rank]);context::syncStream();std::vector<uint16_t>ref(expected->numel());context::memcpyD2H(ref.data(),expected->data(),expected->nbytes());for(int m=0;m<M;++m){for(int n=0;n<N;++n){size_t j=size_t(m)*N+n;float x=fp(actual[l][j]),y=fp(ref[j]),e=std::abs(x-y);worst=std::max(worst,e);if(!std::isfinite(x)||e>.002+.01*std::abs(y)){throw std::runtime_error("pipeline/eager mismatch");
}double sum=0;for(int k=0;k<4;++k){double v=double(k+1)/16;sum+=v*v;}sum*=K/4;float gold=0;for(int rr=0;rr<4;++rr){gold+=fp(bf(float(sum)*float(m%7+phase+1)/64*float(n%13-6)/128*(rr+1)));
}gold=fp(bf(gold));if(l%2){gold=fp(bf(gold+float(n%17-8)/64));
}if(std::abs(x-gold)>.004+.015*std::abs(gold)){throw std::runtime_error("pipeline CPU formula mismatch");
}}
}}barrier.wait();}
 for(int i=0;i<5;++i){graph->run();
}context::syncStream();barrier.wait();auto start=std::chrono::steady_clock::now();for(int i=0;i<20;++i){graph->run();
}context::syncStream();double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/80;barrier.wait();graph.reset();std::cout<<"PIPELINE_CORE_PASS M="<<M<<" K="<<K<<" N="<<N<<" rank="<<rank<<" ms="<<ms<<" max_abs="<<worst<<" CPU_checked=1 changed_inputs=5 bias_cases=2"<<std::endl;infinicclCommDestroy(comms[rank]);}catch(std::exception const&e){errors[rank]=1;std::cerr<<"PIPELINE_CORE_FAIL rank="<<rank<<" "<<e.what()<<std::endl;} });
    }
    for (auto &t : threads) {
        t.join();
    }
    return errors[0] + errors[1] + errors[2] + errors[3];
}
