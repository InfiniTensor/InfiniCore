#ifndef __TAKE_MOORE_API_H__
#define __TAKE_MOORE_API_H__

// 引入上层定义的 Descriptor 宏和基础类
// 这里假设 take.h 位于当前目录的上一级 (standard pattern)
#include "../take.h"

// 使用 take.h 中定义的 DESCRIPTOR 宏
// 这将自动生成 op::take::moore::Descriptor 类定义
DESCRIPTOR(moore)

#endif // __TAKE_MOORE_API_H__