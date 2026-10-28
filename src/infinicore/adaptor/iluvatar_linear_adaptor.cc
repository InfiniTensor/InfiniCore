#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
#include "infinicore/adaptor/iluvatar_linear_adaptor.hpp"
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
namespace infinicore::adaptor::iluvatar_linear {
namespace {
using Linear = at::Tensor (*)(at::Tensor &, at::Tensor &, int64_t,
                              const std::optional<at::Tensor> &, const std::optional<at::Tensor> &, std::optional<bool>);
using Gemv = at::Tensor (*)(at::Tensor &, at::Tensor &,
                            const std::optional<at::Tensor> &, const std::optional<at::Tensor> &);
struct Symbols {
    void *handle = nullptr;
    Linear linear = nullptr;
    Gemv gemv = nullptr;
};
Symbols &symbols() {
    static Symbols s;
    static std::once_flag once;
    std::call_once(once, [] {
        const char *override_path = std::getenv("INFINICORE_IXFORMER_LINEAR_SO");
        const char *paths[] = {override_path, "libixformer.so",
                               "/usr/local/lib/python3.10/site-packages/ixformer/libixformer.so",
                               "/usr/local/lib/python3.12/site-packages/ixformer/libixformer.so"};
        for (const auto *path : paths) {
            if (!path || !*path) {
                continue;
            }
            auto *handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
            if (!handle) {
                continue;
            }
            auto linear = reinterpret_cast<Linear>(dlsym(handle,
                                                         "_ZN8ixformer5infer15ixformer_linearERN2at6TensorES3_lRKSt8optionalIS2_ES7_S4_IbE"));
            auto gemv = reinterpret_cast<Gemv>(dlsym(handle,
                                                     "_ZN8ixformer5infer18ixformer_linear_exERN2at6TensorES3_RKSt8optionalIS2_ES7_"));
            if (linear && gemv) {
                s = {handle, linear, gemv};
                return;
            }
            dlclose(handle);
        }
    });
    return s;
}
} // namespace
bool available() {
    const auto *disabled = std::getenv("INFINICORE_DISABLE_IXFORMER_LINEAR");
    return !(disabled && std::string(disabled) == "1") && symbols().handle;
}
void run(at::Tensor input, at::Tensor weight, at::Tensor output, bool gemv) {
    // Called by the existing deferred Gemm operator, including native graph capture.
    // Never allow the vendor's ATen stream to diverge from the InfiniCore stream.
    set_aten_stream_to_infinicore();
    auto &s = symbols();
    const std::optional<at::Tensor> out{output};
    if (gemv) {
        s.gemv(input, weight, std::nullopt, out);
    } else {
        s.linear(input, weight, -1, std::nullopt, out, false);
    }
}
} // namespace infinicore::adaptor::iluvatar_linear
#else
#include "infinicore/adaptor/iluvatar_linear_adaptor.hpp"
namespace infinicore::adaptor::iluvatar_linear {
bool available() { return false; }
} // namespace infinicore::adaptor::iluvatar_linear
#endif
