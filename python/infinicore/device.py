import _infinicore


class device:
    def __init__(self, type=None, index=None):
        if type is None:
            type = "cpu"

        if isinstance(type, device):
            self.type = type.type
            self.index = type.index

            return

        if ":" in type:
            if index is not None:
                raise ValueError(
                    '`index` should not be provided when `type` contains `":"`.'
                )

            type, index = type.split(":")
            index = int(index)

        self.type = type

        self.index = index

        _type, _index = device._to_infinicore_device(type, index if index else 0)

        self._underlying = _infinicore.Device(_type, _index)

    def __repr__(self):
        return f"device(type='{self.type}'{f', index={self.index}' if self.index is not None else ''})"

    def __str__(self):
        return f"{self.type}{f':{self.index}' if self.index is not None else ''}"

    @staticmethod
    def _to_infinicore_device(type, index):
        all_device_types = tuple(_infinicore.Device.Type.__members__.values())[:-1]
        all_device_count = tuple(
            _infinicore.get_device_count(device) for device in all_device_types
        )

        torch_devices = {
            torch_type: {
                infinicore_type: 0
                for infinicore_type in all_device_types
                if _TORCH_DEVICE_MAP[infinicore_type] == torch_type
            }
            for torch_type in _TORCH_DEVICE_MAP.values()
        }

        for i, count in enumerate(all_device_count):
            infinicore_device_type = _infinicore.Device.Type(i)
            torch_devices[_TORCH_DEVICE_MAP[infinicore_device_type]][
                infinicore_device_type
            ] += count

        for infinicore_device_type, infinicore_device_count in torch_devices[
            type
        ].items():
            for i in range(infinicore_device_count):
                if index == 0:
                    return infinicore_device_type, i

                index -= 1


_TORCH_DEVICE_MAP = {
    _infinicore.Device.Type.CPU: "cpu",
    _infinicore.Device.Type.NVIDIA: "cuda",
    _infinicore.Device.Type.CAMBRICON: "mlu",
    _infinicore.Device.Type.ASCEND: "npu",
    _infinicore.Device.Type.METAX: "cuda",
    _infinicore.Device.Type.MOORE: "musa",
    _infinicore.Device.Type.ILUVATAR: "cuda",
    _infinicore.Device.Type.KUNLUN: "cuda",
    _infinicore.Device.Type.SUGON: "cuda",
}
