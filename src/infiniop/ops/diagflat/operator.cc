 #include "../../operator.h"
 #include "../../handle.h"
 #include "infiniop/ops/diagflat.h"

 #ifdef ENABLE_CPU_API
 #include "cpu/diagflat_cpu.h"
 #endif

 __C infiniStatus_t infiniopCreateDiagflatDescriptor(
     infiniopHandle_t handle,
     infiniopDiagflatDescriptor_t *desc_ptr,
     infiniopTensorDescriptor_t y_desc,
     infiniopTensorDescriptor_t x_desc,
     int64_t offset) {

 #define CREATE(CASE, NAMESPACE)                                                       \
     case CASE:                                                                        \
         return op::diagflat::NAMESPACE::Descriptor::create(                          \
             handle,                                                                   \
             reinterpret_cast<op::diagflat::NAMESPACE::Descriptor **>(desc_ptr),      \
             y_desc,                                                                   \
             std::vector<infiniopTensorDescriptor_t>{x_desc},                         \
             offset)

     switch (handle->device) {
 #ifdef ENABLE_CPU_API
     CREATE(INFINI_DEVICE_CPU, cpu);
 #endif
     default:
         return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
     }

 #undef CREATE
 }

 __C infiniStatus_t
 infiniopGetDiagflatWorkspaceSize(
     infiniopDiagflatDescriptor_t desc,
     size_t *size) {

 #define GET(CASE, NAMESPACE)                                                             \
     case CASE:                                                                           \
         *size = reinterpret_cast<op::diagflat::NAMESPACE::Descriptor *>(desc)           \
                     ->workspaceSize();                                                   \
         return INFINI_STATUS_SUCCESS;

     switch (desc->device_type) {
 #ifdef ENABLE_CPU_API
     GET(INFINI_DEVICE_CPU, cpu);
 #endif
     default:
         return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
     }

 #undef GET
 }

 __C infiniStatus_t infiniopDiagflat(
     infiniopDiagflatDescriptor_t desc,
     void *workspace,
     size_t workspace_size,
     void *output,
     const void *input,
     void *stream) {

 #define CALCULATE(CASE, NAMESPACE)                                                      \
     case CASE:                                                                          \
         return reinterpret_cast<const op::diagflat::NAMESPACE::Descriptor *>(desc)      \
             ->calculate(workspace, workspace_size, output, {input}, stream)

     switch (desc->device_type) {
 #ifdef ENABLE_CPU_API
     CALCULATE(INFINI_DEVICE_CPU, cpu);
 #endif
     default:
         return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
     }

 #undef CALCULATE
 }

 __C infiniStatus_t
 infiniopDestroyDiagflatDescriptor(infiniopDiagflatDescriptor_t desc) {

 #define DELETE(CASE, NAMESPACE)                                                         \
     case CASE:                                                                          \
         delete reinterpret_cast<const op::diagflat::NAMESPACE::Descriptor *>(desc);     \
         return INFINI_STATUS_SUCCESS;

     switch (desc->device_type) {
 #ifdef ENABLE_CPU_API
     DELETE(INFINI_DEVICE_CPU, cpu);
 #endif
     default:
         return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
     }

 #undef DELETE
 }
