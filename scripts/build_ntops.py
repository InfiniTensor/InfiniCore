import functools
import inspect
import itertools
import pathlib

import ninetoothed
from ninetoothed.aot import _HEADER_PATH

CURRENT_FILE_PATH = pathlib.Path(__file__)

BUILD_DIRECTORY_PATH = CURRENT_FILE_PATH.parent.parent / "build" / "ninetoothed"

MAX_NDIM = 5


def _build_op(premake, constexpr_param_grid, caller, op_name, output_dir):
    headers = []
    all_param_names = []
    launches = []

    for combination in _generate_param_value_combinations(constexpr_param_grid):
        arrangement, application, tensors = premake(**combination)

        for param_name, param_value in combination.items():
            if isinstance(param_value, str):
                combination[param_name] = (
                    f"INFINI_DTYPE_{combination['dtype'].replace('fp', 'F')}"
                )

        kernel_name = f"{op_name}_{_generate_suffix(combination.values())}"

        ninetoothed.make(
            arrangement,
            application,
            tensors,
            caller=caller,
            kernel_name=kernel_name,
            output_dir=output_dir,
        )

        header = output_dir / f"{kernel_name}.h"
        param_names = ("stream",) + tuple(
            inspect.signature(application).parameters.keys()
        )
        launch = f"""    if ({_generate_condition(combination)})
        return launch_{kernel_name}({", ".join(param_names)});"""

        headers.append(header)
        all_param_names.append(param_names)
        launches.append(launch)

    includes = f"{'\n'.join(f'#include "{header}"' for header in headers)}"

    param_names = list(
        functools.reduce(
            lambda x, y: dict.fromkeys(x) | dict.fromkeys(y),
            sorted(all_param_names, key=len, reverse=True),
            {},
        )
    )
    param_types = [
        "NineToothedStream",
    ] + ["NineToothedTensor" for _ in range(len(param_names) - 1)]

    for param_name in combination:
        param_names.append(param_name)
        param_types.append("int")

    param_decls = ", ".join(
        f"{type} {param}" for param, type in zip(param_names, param_types)
    )

    func_sig = f'extern "C" inline NineToothedResult launch_{op_name}({param_decls})'

    op_def = f"""{func_sig} {{
{"\n".join(launches)}
    return INFINI_STATUS_NOT_IMPLEMENTED;
}}"""

    content = f"""#include "{_HEADER_PATH}"
#include "infinicore.h"

{includes}\n\n{op_def}\n"""

    (BUILD_DIRECTORY_PATH / f"{op_name}.h").write_text(content)


def _generate_condition(combination):
    return " && ".join(f"{param} == {value}" for param, value in combination.items())


def _generate_suffix(values):
    return "_".join(f"{value}" for value in values)


def _generate_param_value_combinations(param_grid):
    keys = list(param_grid.keys())
    value_combinations = itertools.product(*param_grid.values())

    return tuple(dict(zip(keys, combination)) for combination in value_combinations)


if __name__ == "__main__":
    BUILD_DIRECTORY_PATH.mkdir(exist_ok=True)
