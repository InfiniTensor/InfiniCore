#ifndef DSV4_RMSNORM_SELF_H
#define DSV4_RMSNORM_SELF_H
#include "../../operator.h"
#include "info.h"
#define DESCRIPTOR(NAMESPACE)                                                                                                                                                                                                   \
    namespace op::dsv4_rmsnorm_self::NAMESPACE {                                                                                                                                                                                \
    class Descriptor final : public InfiniopDescriptor {                                                                                                                                                                        \
        Info _info;                                                                                                                                                                                                             \
        float _epsilon;                                                                                                                                                                                                         \
        size_t _workspace_size;                                                                                                                                                                                                 \
        Descriptor(Info info, float epsilon, size_t workspace_size, infiniDevice_t device_type, int device_id) : InfiniopDescriptor{device_type, device_id}, _info(info), _epsilon(epsilon), _workspace_size(workspace_size) {} \
                                                                                                                                                                                                                                \
    public:                                                                                                                                                                                                                     \
        size_t workspaceSize() const { return _workspace_size; }                                                                                                                                                                \
        const Info &info() const { return _info; }                                                                                                                                                                              \
        float epsilon() const { return _epsilon; }                                                                                                                                                                              \
        static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t y_desc, infiniopTensorDescriptor_t x_desc, float epsilon);                                                      \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *y, const void *x, void *stream) const;                                                                                                           \
    };                                                                                                                                                                                                                          \
    }
#endif
