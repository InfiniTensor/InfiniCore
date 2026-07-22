#ifndef INFINICCL_TEST_HPP
#define INFINICCL_TEST_HPP

#include <infiniccl.h>

#include "../utils.h"

int testAllReduce(infiniDevice_t device_type, int ndevice);
int testBroadcast(infiniDevice_t device_type, int ndevice);
int testSend(infiniDevice_t device_type, int ndevice, bool send_from_zero);
int testMultiNodeSend(
    infiniDevice_t device_type,
    bool is_sender,
    const char *master_addr,
    int master_port);

#endif // INFINICCL_TEST_HPP
