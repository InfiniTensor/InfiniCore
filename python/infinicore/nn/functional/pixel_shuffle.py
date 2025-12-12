from infinicore.tensor import Tensor


def pixel_shuffle(input, upscale_factor):
    """
    Rearranges elements in a tensor of shape (*, C*r^2, H, W) to a tensor of shape (*, C, H*r, W*r).
    
    This is useful for implementing efficient sub-pixel convolution with a stride of 1/r.
    
    Args:
        input: Input tensor of shape (*, C*r^2, H, W) where r is upscale_factor
        upscale_factor: Factor to increase spatial resolution by
    
    Returns:
        Tensor: Output tensor of shape (*, C, H*r, W*r)
    
    Examples:
        >>> input = infinicore.randn(1, 9, 4, 4)
        >>> output = infinicore.nn.functional.pixel_shuffle(input, 3)
        >>> print(output.shape)  # (1, 1, 12, 12)
    """
    shape = input.shape
    if len(shape) < 4:
        raise ValueError("pixel_shuffle: input must have at least 4 dimensions")
    
    # Calculate dimensions
    c_dim = len(shape) - 3  # Channel dimension index
    h_dim = len(shape) - 2  # Height dimension index
    w_dim = len(shape) - 1  # Width dimension index
    
    C_r2 = shape[c_dim]
    if C_r2 % (upscale_factor * upscale_factor) != 0:
        raise ValueError(
            f"pixel_shuffle: number of input channels ({C_r2}) must be divisible by "
            f"upscale_factor^2 ({upscale_factor * upscale_factor})"
        )
    
    C = C_r2 // (upscale_factor * upscale_factor)
    H = shape[h_dim]
    W = shape[w_dim]
    
    # Calculate batch dimensions (all dimensions before channel)
    batch_shape = shape[:c_dim]
    
    # Step 1: Reshape input from (..., C*r^2, H, W) to (..., C, r, r, H, W)
    new_shape = list(batch_shape) + [C, upscale_factor, upscale_factor, H, W]
    x = input.view(new_shape)
    
    # Step 2: Permute dimensions from (..., C, r, r, H, W) to (..., C, H, r, W, r)
    # After reshape: [batch..., C, r1, r2, H, W]
    # Permute to:    [batch..., C, H, r1, W, r2]
    # Following PyTorch's implementation: permute(0, 1, 4, 2, 5, 3) for 4D case
    # For general case: [batch_dims..., C, H, r1, W, r2]
    # After reshape, dimensions are: [batch_dims (0...n-1), C (n), r1 (n+1), r2 (n+2), H (n+3), W (n+4)]
    # where n = len(batch_shape)
    # Target order: [batch_dims (0...n-1), C (n), H (n+3), r1 (n+1), W (n+4), r2 (n+2)]
    permute_order = list(range(len(batch_shape)))  # Keep batch dimensions
    n = len(batch_shape)
    permute_order.extend([
        n,      # C (position n)
        n + 3,  # H (was at position n+3)
        n + 1,  # r1 (was at position n+1)
        n + 4,  # W (was at position n+4)
        n + 2,  # r2 (was at position n+2)
    ])
    x = x.permute(permute_order)
    
    # Step 3: Make contiguous before final view (permute creates non-contiguous tensor)
    # This is critical for both correctness and performance
    x = x.contiguous()
    
    # Step 4: Reshape from (..., C, H, r, W, r) to (..., C, H*r, W*r)
    output_shape = list(batch_shape) + [C, H * upscale_factor, W * upscale_factor]
    return x.view(output_shape)

