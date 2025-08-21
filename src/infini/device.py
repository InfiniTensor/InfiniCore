from libinfiniop import torch_device_map as _TORCH_DEVICE_MAP

from infini import core


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

        if index is None:
            index = 0

        self.type = type

        self.index = index

        type_, index_ = device._to_infini_device(type, index)

        self.type_ = type_

        self.index_ = index_

    @staticmethod
    def _to_infini_device(type, index):
        all_device_count = core.get_all_device_count()

        torch_devices = {
            torch_type: {
                infini_type.value: 0
                for _, infini_type in tuple(core._infini.Device.__members__.items())[
                    :-1
                ]
                if _TORCH_DEVICE_MAP[infini_type] == torch_type
            }
            for torch_type in _TORCH_DEVICE_MAP.values()
        }

        for i, count in enumerate(all_device_count):
            torch_devices[_TORCH_DEVICE_MAP[i]][i] += count

        for infini_device_type, infini_device_count in torch_devices[type].items():
            for i in range(infini_device_count):
                if index == 0:
                    return infini_device_type, i

                index -= 1


default_device = device("cpu")


def get_default_device():
    return default_device


def set_default_device(device):
    global default_device

    default_device = device(device)
