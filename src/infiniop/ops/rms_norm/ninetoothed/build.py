import ninetoothed
from ntops.kernels import rms_norm

import infiniop.ninetoothed.build


def build():
    MAX_NDIM = 5

    ndim_values = range(1, MAX_NDIM + 1)
    dtype_values = (ninetoothed.float16, ninetoothed.bfloat16, ninetoothed.float32)

    constexpr_param_grid = {
        "ndim": ndim_values,
        "num_normalized_dims": (1,),
        "input_dtype": dtype_values,
        "weight_dtype": dtype_values,
        "output_dtype": dtype_values,
        "block_size": (1024,),
    }

    infiniop.ninetoothed.build.build(
        rms_norm.premake,
        constexpr_param_grid,
        caller="cuda",
        op_name="rms_norm",
        output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
    )
