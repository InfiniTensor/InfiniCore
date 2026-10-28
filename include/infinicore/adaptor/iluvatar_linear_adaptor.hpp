#pragma once
namespace infinicore::adaptor::iluvatar_linear {
bool available();
}
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
#include <ATen/ATen.h>
namespace infinicore::adaptor::iluvatar_linear {
void run(at::Tensor input, at::Tensor weight, at::Tensor output, bool gemv);
} // namespace infinicore::adaptor::iluvatar_linear
#endif
