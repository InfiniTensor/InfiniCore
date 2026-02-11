class InfiniDeviceEnum:
    CPU = 0
    NVIDIA = 1
    CAMBRICON = 2
    ASCEND = 3
    METAX = 4
    MOORE = 5
    ILUVATAR = 6
    KUNLUN = 7
    HYGON = 8
    QY = 9
    ALI = 10


InfiniDeviceNames = {
    InfiniDeviceEnum.CPU: "CPU",
    InfiniDeviceEnum.NVIDIA: "NVIDIA",
    InfiniDeviceEnum.CAMBRICON: "Cambricon",
    InfiniDeviceEnum.ASCEND: "Ascend",
    InfiniDeviceEnum.METAX: "Metax",
    InfiniDeviceEnum.MOORE: "Moore",
    InfiniDeviceEnum.ILUVATAR: "Iluvatar",
    InfiniDeviceEnum.QY: "Qy",
    InfiniDeviceEnum.KUNLUN: "Kunlun",
    InfiniDeviceEnum.HYGON: "Hygon",
    InfiniDeviceEnum.ALI: "Ali",
}

torch_device_map = {
    InfiniDeviceEnum.CPU: "cpu",
    InfiniDeviceEnum.NVIDIA: "cuda",
    InfiniDeviceEnum.CAMBRICON: "mlu",
    InfiniDeviceEnum.ASCEND: "npu",
    InfiniDeviceEnum.METAX: "cuda",
    InfiniDeviceEnum.MOORE: "musa",
    InfiniDeviceEnum.ILUVATAR: "cuda",
    InfiniDeviceEnum.KUNLUN: "cuda",
    InfiniDeviceEnum.HYGON: "cuda",
    InfiniDeviceEnum.QY: "cuda",
    InfiniDeviceEnum.ALI: "cuda",
}
