#ifndef __INFINIOP_CONCAT_MLA_Q_API_H__
#define __INFINIOP_CONCAT_MLA_Q_API_H__

#include "../operator_descriptor.h"

__INFINI_C __export infiniStatus_t infiniopConcatMlaQ(
    infiniopHandle_t handle,
    infiniopTensorDescriptor_t ql_nope_desc,
    infiniopTensorDescriptor_t q_pe_desc,
    infiniopTensorDescriptor_t q_out_desc,
    const void *ql_nope,
    const void *q_pe,
    void *q_out,
    void *stream);

#endif
