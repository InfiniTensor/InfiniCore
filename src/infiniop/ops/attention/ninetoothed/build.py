import ninetoothed
from ntops.kernels import scaled_dot_product_attention
from ntops.kernels.scaled_dot_product_attention import CausalVariant

import infiniop.ninetoothed.build


def build():
    with_kv_cache_values = (1,)
    emb_dim_values = (16, 32, 64, 128)
    is_causal_values = (1,)
    with_attn_mask_values = (0,)
    causal_variant_values = (CausalVariant.LOWER_RIGHT,)
    dtype_values = (ninetoothed.float16, ninetoothed.float32)
    block_size_m_values = (32,)
    block_size_n_values = (32,)

    constexpr_param_grid = {
        "with_kv_cache": with_kv_cache_values,
        "emb_dim": emb_dim_values,
        "is_causal": is_causal_values,
        "with_attn_mask": with_attn_mask_values,
        "causal_variant": causal_variant_values,
        "dtype": dtype_values,
        "block_size_m": block_size_m_values,
        "block_size_n": block_size_n_values,
    }

    infiniop.ninetoothed.build.build(
        scaled_dot_product_attention.premake,
        constexpr_param_grid,
        caller="cuda",
        op_name="scaled_dot_product_attention",
        output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
    )
