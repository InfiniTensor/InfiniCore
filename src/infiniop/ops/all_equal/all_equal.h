#ifndef __ALL_EQUAL_H
#define __ALL_EQUAL_H

#include "../../handle.h"
#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                     \
                                                                  \
    namespace op::all_equal::NAMESPACE {                          \
    class Descriptor final : public InfiniopDescriptor {          \
        struct Opaque;                                            \
        Opaque *_opaque;                                          \
        op::all_equal::AllEqualInfo _info;                        \
        size_t _workspace_size;                                   \
                                                                  \
        Descriptor(                                               \
            Opaque *opaque,                                       \
            AllEqualInfo info,                                    \
            size_t workspace_size,                                \
            infiniDevice_t device_type,                           \
            int device_id)                                        \
            : InfiniopDescriptor{device_type, device_id},         \
              _opaque(opaque),                                    \
              _info(info),                                        \
              _workspace_size(workspace_size) {}                  \
                                                                  \
    public:                                                       \
        ~Descriptor();                                            \
                                                                  \
        size_t workspaceSize() const { return _workspace_size; }  \
                                                                  \
        static infiniStatus_t create(                             \
            infiniopHandle_t handle,                              \
            Descriptor **desc_ptr,                                \
            infiniopTensorDescriptor_t output_desc,               \
            std::vector<infiniopTensorDescriptor_t> input_descs); \
                                                                  \
        infiniStatus_t calculate(                                 \
            void *workspace, size_t workspace_size,               \
            void *output,                                         \
            std::vector<const void *> inputs,                     \
            void *stream) const;                                  \
    };                                                            \
    }

#endif // __ALL_EQUAL_H
