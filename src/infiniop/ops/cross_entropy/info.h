#ifndef CROSS_ENTROPY_INFO_H
#define CROSS_ENTROPY_INFO_H
#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

// #include "../../operator_descriptor.h"
#include <cstddef>

struct CrossEntropyInfo {
    int dtype;         // logits dtype
    int target_dtype;  // label dtype
    size_t outer_size; // batch * seq
    size_t vocab_size; // logits 最后一维
    ptrdiff_t x_stride;
};

#endif