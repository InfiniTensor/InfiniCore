import functools
import inspect
import pathlib

import ninetoothed
from ninetoothed import Tensor
from ninetoothed.aot import _HEADER_PATH
from ntops.kernels import element_wise, relu

CURRENT_FILE_PATH = pathlib.Path(__file__)

BUILD_DIRECTORY_PATH = CURRENT_FILE_PATH.parent.parent / "build" / "ninetoothed"

MAX_NDIM = 5


def _build_relu():
    all_tensors = tuple(
        (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))
        for ndim in range(1, MAX_NDIM + 1)
        for dtype in (ninetoothed.float16, ninetoothed.float32, ninetoothed.float64)
    )

    _build_element_wise(
        relu.application,
        all_tensors,
        caller="cuda",
        op_name="relu",
        output_dir=BUILD_DIRECTORY_PATH,
    )


def _build_element_wise(application, all_tensors, caller, op_name, output_dir):
    param_names = ("stream",) + tuple(inspect.signature(application).parameters.keys())
    param_types = ("NineToothedStream",) + tuple(
        "NineToothedTensor" for _ in range(len(param_names) - 1)
    )

    func_sig = f'extern "C" inline NineToothedResult launch_{op_name}({", ".join(f"{type} {param}" for param, type in zip(param_names, param_types))}, int ndim, int dtype)'

    headers = []
    launches = []

    for tensors in all_tensors:
        ndim = tensors[0].ndim
        dtype = tensors[0].dtype

        kernel_name = f"{op_name}_{ndim}_{dtype}"

        ninetoothed.make(
            functools.partial(element_wise.arrangement, block_size=1024),
            application,
            tensors,
            caller=caller,
            kernel_name=kernel_name,
            output_dir=output_dir,
        )

        header = output_dir / f"{kernel_name}.h"
        launch = f"""    if (ndim == {ndim} && dtype == INFINI_DTYPE_{dtype.replace("fp", "F")})
        return launch_{kernel_name}({", ".join(param_names)});"""

        headers.append(header)
        launches.append(launch)

    includes = f"{'\n'.join(f'#include "{header}"' for header in headers)}"

    op_def = f"""{func_sig} {{
{"\n".join(launches)}
    return INFINI_STATUS_NOT_IMPLEMENTED;
}}"""

    content = f"""#include "{_HEADER_PATH}"
#include "infinicore.h"

{includes}\n\n{op_def}\n"""

    (BUILD_DIRECTORY_PATH / f"{op_name}.h").write_text(content)


if __name__ == "__main__":
    BUILD_DIRECTORY_PATH.mkdir(exist_ok=True)

    _build_relu()
