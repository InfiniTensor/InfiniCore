import sys

import infinicore


def use_ntops():
    import ntops

    return _TemporaryAttributes(
        (("ntops.torch.torch", infinicore),)
        + tuple(
            (f"infinicore.{op_name}", getattr(ntops.torch, op_name))
            for op_name in ntops.torch.__all__
        )
    )


class _TemporaryAttributes:
    def __init__(self, attribute_mappings):
        self._attribute_mappings = attribute_mappings

        self._original_values = {}

    def __enter__(self):
        for attr_path, new_value in self._attribute_mappings:
            parent, attr_name = self._resolve_path(attr_path)

            try:
                self._original_values[attr_path] = getattr(parent, attr_name)
            except AttributeError:
                pass

            setattr(parent, attr_name, new_value)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for attr_path, _ in self._attribute_mappings:
            parent, attr_name = self._resolve_path(attr_path)

            if attr_path in self._original_values:
                setattr(parent, attr_name, self._original_values[attr_path])
            else:
                delattr(parent, attr_name)

    @staticmethod
    def _resolve_path(path):
        *parent_parts, attr_name = path.split(".")

        curr = sys.modules[parent_parts[0]]

        for part in parent_parts[1:]:
            curr = getattr(curr, part)

        return curr, attr_name
