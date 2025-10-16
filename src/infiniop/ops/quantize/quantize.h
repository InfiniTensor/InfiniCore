#ifndef __QUANTIZE_H__
#define __QUANTIZE_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                      \
                                                                                   \
    namespace op::quantize::NAMESPACE {                                            \
    class Descriptor final : public InfiniopDescriptor {                           \
        struct Opaque;                                                             \
        Opaque *_opaque;                                                           \
        infiniDtype_t _dtype;                                                      \
        QuantizeInfo _info;                                                        \
        size_t _workspace_size;                                                    \
                                                                                   \
        Descriptor(infiniDtype_t dtype, QuantizeInfo info, size_t workspace_size_, \
                   Opaque *opaque, infiniDevice_t device_type, int device_id)      \
            : InfiniopDescriptor{device_type, device_id}, _opaque(opaque),         \
              _dtype(dtype), _info(info), _workspace_size(workspace_size_) {}      \
                                                                                   \
    public:                                                                        \
        ~Descriptor();                                                             \
                                                                                   \
        size_t workspaceSize() const { return _workspace_size; }                   \
                                                                                   \
        static infiniStatus_t create(infiniopHandle_t handle,                      \
                                     Descriptor **desc_ptr,                        \
                                     infiniopTensorDescriptor_t input_desc,        \
                                     infiniopTensorDescriptor_t output_q_desc,     \
                                     infiniopTensorDescriptor_t output_s_desc);    \
                                                                                   \
        infiniStatus_t calculate(void *workspace, size_t workspace_size,           \
                                 void *input, void *output_q, void *output_s,      \
                                 int group_size, double eps, double min_8bit,      \
                                 double max_8bit, bool scale_ue8m0,                \
                                 void *stream) const;                              \
    };                                                                             \
    }

#endif // __QUANTIZE_H__
