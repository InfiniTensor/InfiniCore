#ifndef __INFINIRT_TEST_H__
#define __INFINIRT_TEST_H__
#include "../utils.h"

bool test_setDevice(infiniDevice_t device, int deviceId);
bool test_memcpy(infiniDevice_t device, int deviceId, size_t dataSize);

#endif
