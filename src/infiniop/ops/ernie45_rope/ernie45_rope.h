#ifndef __ERNIE45_ROPE_H__
#define __ERNIE45_ROPE_H__

#include "../../operator.h"
#include "info.h"

#define MROPE_DESCRIPTOR(NAMESPACE)                                                                                          \
    namespace op::ernie45_rope::NAMESPACE {                                                                                  \
    class MropeDescriptor final : public InfiniopDescriptor {                                                                \
        struct Opaque;                                                                                                       \
        Opaque *_opaque;                                                                                                     \
        QKInfo _info;                                                                                                        \
        size_t _workspace_size;                                                                                              \
        MropeDescriptor(Opaque *opaque, QKInfo info, size_t workspace_size, infiniDevice_t device_type, int device_id)       \
            : InfiniopDescriptor{device_type, device_id}, _opaque(opaque), _info(info), _workspace_size(workspace_size) {}   \
                                                                                                                             \
    public:                                                                                                                  \
        ~MropeDescriptor();                                                                                                  \
        size_t workspaceSize() const { return _workspace_size; }                                                             \
        static infiniStatus_t create(infiniopHandle_t handle, MropeDescriptor **desc_ptr, infiniopTensorDescriptor_t q_desc, \
                                     infiniopTensorDescriptor_t k_desc, infiniopTensorDescriptor_t pos_desc, double theta,   \
                                     size_t section_h, size_t section_w, size_t section_t);                                  \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *q, void *k, const void *positions,            \
                                 void *stream) const;                                                                        \
    };                                                                                                                       \
    }

#define VISION_ROPE_DESCRIPTOR(NAMESPACE)                                                                                         \
    namespace op::ernie45_rope::NAMESPACE {                                                                                       \
    class VisionRopeDescriptor final : public InfiniopDescriptor {                                                                \
        struct Opaque;                                                                                                            \
        Opaque *_opaque;                                                                                                          \
        VisionInfo _info;                                                                                                         \
        size_t _workspace_size;                                                                                                   \
        VisionRopeDescriptor(Opaque *opaque, VisionInfo info, size_t workspace_size, infiniDevice_t device_type, int device_id)   \
            : InfiniopDescriptor{device_type, device_id}, _opaque(opaque), _info(info), _workspace_size(workspace_size) {}        \
                                                                                                                                  \
    public:                                                                                                                       \
        ~VisionRopeDescriptor();                                                                                                  \
        size_t workspaceSize() const { return _workspace_size; }                                                                  \
        static infiniStatus_t create(infiniopHandle_t handle, VisionRopeDescriptor **desc_ptr, infiniopTensorDescriptor_t q_desc, \
                                     infiniopTensorDescriptor_t k_desc, infiniopTensorDescriptor_t pos_desc, double theta);       \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *q, void *k, const void *positions,                 \
                                 void *stream) const;                                                                             \
    };                                                                                                                            \
    }

#endif
