#ifndef __RANDOM_SAMPLE_H__
#define __RANDOM_SAMPLE_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                             \
                                                          \
    namespace op::random_sample::NAMESPACE {              \
    class Descriptor final : public InfiniopDescriptor {  \
        struct Opaque;                                    \
        Opaque *_opaque;                                  \
                                                          \
        RandomSampleInfo _info;                           \
        size_t _min_workspace_size;                       \
                                                          \
        Descriptor(                                       \
            RandomSampleInfo info,                        \
            size_t min_workspace_size,                    \
            Opaque *opaque,                               \
            infiniDevice_t device_type,                   \
            int device_id)                                \
            : InfiniopDescriptor{device_type, device_id}, \
              _opaque(opaque),                            \
              _info(info),                                \
              _min_workspace_size(min_workspace_size) {}  \
                                                          \
    public:                                               \
        ~Descriptor();                                    \
                                                          \
        static infiniStatus_t create(                     \
            infiniopHandle_t handle,                      \
            Descriptor **desc_ptr,                        \
            infiniopTensorDescriptor_t result_desc,       \
            infiniopTensorDescriptor_t probs_desc);       \
                                                          \
        size_t minWorkspaceSize() const;                  \
                                                          \
        infiniStatus_t calculate(                         \
            void *workspace,                              \
            size_t workspace_size,                        \
            void *result,                                 \
            const void *probs,                            \
            float random_val,                             \
            float topp,                                   \
            int topk,                                     \
            float temperature,                            \
            float repetition_penalty,                     \
            const uint32_t *previous_tokens,              \
            size_t previous_tokens_len,                   \
            void *stream) const;                          \
    };                                                    \
    }

namespace op::random_sample {

struct CalculateArgs {
    void *workspace;
    size_t workspace_size;
    void *result;
    const void *probs;
    float random_val, topp, temperature, repetition_penalty;
    int topk;
    const uint32_t *previous_tokens;
    size_t previous_tokens_len;
    void *stream;
};

class Calculate {

    template <class Tidx, class Tval, class Algo>
    static void switch_f(Algo algo, size_t n, CalculateArgs args) {
        // Handle disabled topk (0 or -1 means consider all tokens, like vLLM)
        int effective_topk = args.topk;
        if (effective_topk <= 0) {
            effective_topk = static_cast<int>(n);  // Consider all tokens
        }

        if (args.random_val == 0 || args.topp == 0 || effective_topk == 1 || args.temperature == 0) {
            algo.template argmax<Tidx, Tval>(
                args.workspace, args.workspace_size,
                args.result, args.probs, n,
                args.stream);
        } else {
            algo.template random<Tidx, Tval>(
                args.workspace, args.workspace_size,
                args.result, args.probs, n,
                args.random_val, args.topp, effective_topk, args.temperature, args.repetition_penalty,
                args.previous_tokens, args.previous_tokens_len,
                args.stream);
        }
    }

    template <class Tidx, class Algo>
    static void switch_val(
        Algo algo,
        infiniDtype_t dt_p, size_t n, CalculateArgs args) {
        switch (dt_p) {
        case INFINI_DTYPE_F16:
            switch_f<Tidx, fp16_t>(algo, n, args);
            break;
        case INFINI_DTYPE_BF16:
            switch_f<Tidx, bf16_t>(algo, n, args);
            break;
        case INFINI_DTYPE_F32:
            switch_f<Tidx, float>(algo, n, args);
            break;
        case INFINI_DTYPE_F64:
            switch_f<Tidx, double>(algo, n, args);
            break;
        default:
            // unreachable
            std::abort();
        }
    }

public:
    template <class Algo>
    static infiniStatus_t calculate(
        Algo algo,
        RandomSampleInfo info,
        void *workspace, size_t workspace_size,
        void *result, const void *probs,
        float random_val, float topp, int topk, float temperature, float repetition_penalty,
        const uint32_t *previous_tokens, size_t previous_tokens_len,
        void *stream) {

#define CASE(DT_VAL, DT_TYP)                      \
    case DT_VAL:                                  \
        switch_val<DT_TYP>(                       \
            algo, info.dt_p, info.n,              \
            {workspace, workspace_size,           \
             result, probs,                       \
             random_val, topp, temperature, repetition_penalty, topk, \
             previous_tokens, previous_tokens_len, \
             stream});                            \
        break

        switch (info.dt_i) {
            CASE(INFINI_DTYPE_I8, int8_t);
            CASE(INFINI_DTYPE_I16, int16_t);
            CASE(INFINI_DTYPE_I32, int32_t);
            CASE(INFINI_DTYPE_I64, int64_t);
            CASE(INFINI_DTYPE_U8, uint8_t);
            CASE(INFINI_DTYPE_U16, uint16_t);
            CASE(INFINI_DTYPE_U32, uint32_t);
            CASE(INFINI_DTYPE_U64, uint64_t);
        default:
            // unreachable
            std::abort();
        }

#undef CASE

        return INFINI_STATUS_SUCCESS;
    }
};

} // namespace op::random_sample

#endif // __RANDOM_SAMPLE_H__
