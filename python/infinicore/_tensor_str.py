import contextlib
import dataclasses
from typing import Any, Optional

from infinicore.lib import _infinicore


@dataclasses.dataclass
class __PrinterOptions:
    precision: int = 4
    threshold: float = 1000
    edgeitems: int = 3
    linewidth: int = 80
    sci_mode: Optional[bool] = None


PRINT_OPTS = __PrinterOptions()


def set_printoptions(
    precision=None,
    threshold=None,
    edgeitems=None,
    linewidth=None,
    profile=None,
    sci_mode=None,
):
    r"""Set options for printing.
    Args:
        precision: Number of digits of precision for floating point output (default = 4).
        threshold: Total number of array elements which trigger summarization rather than full `repr` (default = 1000).
        edgeitems: Number of array items in summary at beginning and end of each dimension (default = 3).
        linewidth: The number of characters per line (default = 80).
        profile: Sane defaults for pretty printing. Can override with any of  the above options. (any one of `default`, `short`, `full`)
        sci_mode: Enable (True) or disable (False) scientific notation.
                  If None (default) is specified, the value is automatically chosen by the framework.
    Example::
        >>> # Limit the precision of elements
        >>> torch.set_printoptions(precision=2)
        >>> torch.tensor([1.12345])
        tensor([1.12])
    """
    if profile is not None:
        if profile == "default":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80
        elif profile == "short":
            PRINT_OPTS.precision = 2
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 2
            PRINT_OPTS.linewidth = 80
        elif profile == "full":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = 2147483647  # CPP_INT32_MAX
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80
        else:
            raise ValueError(
                f"Invalid profile: {profile}. the profile must be one of 'default', 'short', 'full'"
            )

    if precision is not None:
        PRINT_OPTS.precision = precision
    if threshold is not None:
        PRINT_OPTS.threshold = threshold
    if edgeitems is not None:
        PRINT_OPTS.edgeitems = edgeitems
    if linewidth is not None:
        PRINT_OPTS.linewidth = linewidth
    PRINT_OPTS.sci_mode = sci_mode

    _infinicore.set_printoptions(
        PRINT_OPTS.precision,
        PRINT_OPTS.threshold,
        PRINT_OPTS.edgeitems,
        PRINT_OPTS.linewidth,
        PRINT_OPTS.sci_mode,
    )


def get_printoptions() -> dict[str, Any]:
    r"""Gets the current options for printing, as a dictionary that
    can be passed as ``**kwargs`` to set_printoptions().
    """
    return dataclasses.asdict(PRINT_OPTS)


@contextlib.contextmanager
def printoptions(
    precision=None, threshold=None, edgeitems=None, linewidth=None, sci_mode=None
):
    r"""Context manager that temporarily changes the print options."""
    old_kwargs = get_printoptions()
    set_printoptions(
        precision=precision,
        threshold=threshold,
        edgeitems=edgeitems,
        linewidth=linewidth,
        sci_mode=sci_mode,
    )
    try:
        yield
    finally:
        set_printoptions(**old_kwargs)


def _str(self):
    cpp_tensor_str = self._underlying.__str__()
    py_dtype_str = ", dtype=" + self.dtype.__repr__()

    py_tensor_str = cpp_tensor_str.split(", d")[0]
    if self.device.type != "cpu":
        py_device_str = ", device='" + self.device.__str__() + "'"
        py_tensor_str += py_device_str
    py_tensor_str += py_dtype_str + ")"

    return py_tensor_str


set_printoptions()
