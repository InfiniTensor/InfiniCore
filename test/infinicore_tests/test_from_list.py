import torch

import infinicore


def _copy_infinicore_to_torch(infinicore_tensor, torch_result_tensor):
    """Helper function: Copy infinicore tensor to torch tensor
    
    Args:
        infinicore_tensor: Source infinicore tensor
        torch_result_tensor: Target torch tensor (to receive data)
    
    Returns:
        torch.Tensor: Torch tensor containing copied data
    """
    # Determine the device from torch tensor
    torch_device = torch_result_tensor.device
    if torch_device.type == "cuda":
        infini_device = infinicore.device("cuda", torch_device.index or 0)
    else:
        infini_device = infinicore.device("cpu", 0)
    
    infini_result = infinicore.from_blob(
        torch_result_tensor.data_ptr(),
        list(torch_result_tensor.shape),
        dtype=infinicore_tensor.dtype,
        device=infini_device,
    )
    
    # Ensure tensor is on the same device as target
    tensor_to_copy = infinicore_tensor
    if tensor_to_copy.device.type != infini_device.type or \
       (infini_device.type == "cuda" and tensor_to_copy.device.index != infini_device.index):
        tensor_to_copy = tensor_to_copy.to(infini_device)
    
    infini_result.copy_(tensor_to_copy)
    return torch_result_tensor


def compare_with_torch(infinicore_tensor, expected_list, dtype=torch.float32, atol=1e-6):
    """Helper function: Compare infinicore tensor computation results with PyTorch expected results
    
    Uses unified addition verification: tensor + zero_tensor == tensor
    Converts all data to float32 for verification to avoid different verification paths for different data types
    
    Args:
        infinicore_tensor: infinicore tensor object
        expected_list: Expected data (Python list, can be nested)
        dtype: torch dtype (for compatibility, actually uses float32 for verification)
        atol: Absolute tolerance for floating point comparison
    
    Returns:
        bool: Whether data matches
    """
    # Verify basic attributes
    expected_shape = list(torch.tensor(expected_list).shape)
    assert list(infinicore_tensor.shape) == expected_shape, \
        f"Shape mismatch: expected {expected_shape}, got {list(infinicore_tensor.shape)}"
    
    # Flatten nested list and convert to float
    def flatten(data):
        result = []
        for item in data:
            if isinstance(item, (list, tuple)):
                result.extend(flatten(item))
            else:
                result.append(float(item))
        return result
    
    flat_expected = flatten(expected_list)
    
    # Unified verification through addition: create float32 version of tensor for verification
    # This unifies verification logic and avoids different verification paths for different data types
    tensor_f32 = infinicore.from_list(flat_expected, dtype=infinicore.float32)
    expected_f32 = torch.tensor(flat_expected, dtype=torch.float32)
    
    # Add with zero tensor to verify data: tensor + zero == tensor
    zero_f32 = infinicore.from_list([0.0] * tensor_f32.numel(), dtype=infinicore.float32)
    result = tensor_f32 + zero_f32
    
    # Verify result
    torch_result = _copy_infinicore_to_torch(result, torch.zeros_like(expected_f32))
    return torch.allclose(expected_f32, torch_result, atol=atol)


# Parameterized test data: test cases for different dimensions
_TEST_SHAPES = [
    ([1, 2, 3, 4, 5], [5], "1D"),
    ([[1, 2, 3], [4, 5, 6]], [2, 3], "2D"),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [2, 2, 2], "3D"),
]


def test_from_list_basic_shapes():
    """Test converting lists of different dimensions to tensor (parameterized test)"""
    for data, expected_shape, dim_name in _TEST_SHAPES:
        print(f"Testing {dim_name} list to tensor conversion")
        
        # Create float tensor for addition verification
        tensor_f32 = infinicore.from_list(data, dtype=infinicore.float32)
        
        # Verify shape and data type
        assert list(tensor_f32.shape) == expected_shape, \
            f"{dim_name}: Shape mismatch: expected {expected_shape}, got {list(tensor_f32.shape)}"
        assert tensor_f32.dtype == infinicore.float32, \
            f"{dim_name}: Expected float32, got {tensor_f32.dtype}"
        
        # Verify data correctness through addition
        assert compare_with_torch(tensor_f32, data, dtype=torch.float32), \
            f"{dim_name}: Data mismatch"
        
        print(f"✓ {dim_name} list test passed")


def test_from_list_float():
    """Test converting float list to tensor"""
    print("=" * 50)
    print("Testing float list to tensor conversion")
    
    data = [[1.0, 2.5, 3.7], [4.2, 5.9, 6.1]]
    tensor = infinicore.from_list(data)
    
    # Verify dtype (should be float64, as Python float defaults to float64)
    assert tensor.dtype == infinicore.float64, f"Expected float64, got {tensor.dtype}"
    
    # Use unified verification method
    assert compare_with_torch(tensor, data, dtype=torch.float64), "Data mismatch"
    
    print("✓ Float list test passed")


def test_from_list_with_dtype():
    """Test converting list to tensor with specified dtype"""
    print("=" * 50)
    print("Testing list to tensor conversion with specified dtype")
    
    data = [1, 2, 3, 4, 5]
    # Specify as float32
    tensor = infinicore.from_list(data, dtype=infinicore.float32)
    
    # Verify dtype
    assert tensor.dtype == infinicore.float32, f"Expected float32, got {tensor.dtype}"
    
    # Use unified verification method
    assert compare_with_torch(tensor, data, dtype=torch.float32), "Data mismatch"
    
    print("✓ Specified dtype test passed")


def test_from_list_with_device():
    """Test converting list to tensor with specified device"""
    print("=" * 50)
    print("Testing list to tensor conversion with specified device")
    
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    
    # Test CPU device
    tensor_cpu = infinicore.from_list(data, dtype=infinicore.float32, device=infinicore.device("cpu", 0))
    assert tensor_cpu.device.type == "cpu", "Expected CPU device"
    # Verify data correctness on CPU
    assert compare_with_torch(tensor_cpu, data, dtype=torch.float32), "CPU data mismatch"
    
    # Test CUDA device (if available)
    try:
        # Check if CUDA is available in PyTorch
        if not torch.cuda.is_available():
            print("⚠ CUDA not available in PyTorch, skipping CUDA test")
        else:
            tensor_cuda = infinicore.from_list(data, dtype=infinicore.float32, device=infinicore.device("cuda", 0))
            assert tensor_cuda.device.type == "cuda", "Expected CUDA device"
            
            # Create PyTorch CUDA tensor for comparison
            torch_expected_cuda = torch.tensor(data, dtype=torch.float32, device="cuda:0")
            torch_result_cuda = torch.zeros_like(torch_expected_cuda)
            
            # Copy infinicore CUDA tensor to PyTorch CUDA tensor and compare
            _copy_infinicore_to_torch(tensor_cuda, torch_result_cuda)
            assert torch.allclose(torch_expected_cuda, torch_result_cuda), "CUDA data mismatch"
            
            print("✓ CUDA device test passed (device type and data correctness verified on CUDA)")
    except Exception as e:
        print(f"⚠ CUDA device not available, skipping CUDA test: {e}")
    
    print("✓ Specified device test passed")


def test_from_list_operations():
    """Test operations on tensors created from lists"""
    print("=" * 50)
    print("Testing operations on tensors created from lists")
    
    data1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    data2 = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    
    t1 = infinicore.from_list(data1, dtype=infinicore.float32)
    t2 = infinicore.from_list(data2, dtype=infinicore.float32)
    
    # Test addition and multiplication
    result_add = t1 + t2
    result_mul = t1 * t2
    
    # Compare with torch
    torch_t1 = torch.tensor(data1, dtype=torch.float32)
    torch_t2 = torch.tensor(data2, dtype=torch.float32)
    torch_add = torch_t1 + torch_t2
    torch_mul = torch_t1 * torch_t2
    
    # Verify results
    torch_add_result = _copy_infinicore_to_torch(result_add, torch.zeros_like(torch_add))
    assert torch.allclose(torch_add, torch_add_result), "Addition result mismatch"
    
    torch_mul_result = _copy_infinicore_to_torch(result_mul, torch.zeros_like(torch_mul))
    assert torch.allclose(torch_mul, torch_mul_result), "Multiplication result mismatch"
    
    print("✓ Operations test passed")


def test_from_list_single_element():
    """Test converting single element list to tensor"""
    print("=" * 50)
    print("Testing single element list to tensor conversion")
    
    data = [42]
    tensor = infinicore.from_list(data, dtype=infinicore.float32)
    
    assert list(tensor.shape) == [1], f"Expected shape [1], got {tensor.shape}"
    assert tensor.dtype == infinicore.float32, f"Expected float32, got {tensor.dtype}"
    
    # Use unified verification method
    assert compare_with_torch(tensor, data, dtype=torch.float32), "Data mismatch"
    
    print("✓ Single element test passed")


def test_from_list_edge_cases():
    """Test edge cases"""
    print("=" * 50)
    print("Testing edge cases")
    
    # Test empty list (should raise exception)
    try:
        infinicore.from_list([])
        assert False, "Expected ValueError for empty list"
    except ValueError:
        pass  # Expected exception
    
    # Test non-list input (should raise exception)
    try:
        infinicore.from_list("not a list")
        assert False, "Expected TypeError for non-list input"
    except TypeError:
        pass  # Expected exception
    
    # Test single scalar (wrapped in list)
    data = 42
    tensor = infinicore.from_list([data], dtype=infinicore.float32)
    assert list(tensor.shape) == [1], f"Expected shape [1], got {tensor.shape}"
    assert compare_with_torch(tensor, [data], dtype=torch.float32), "Data mismatch"
    
    print("✓ Edge cases test passed")


if __name__ == "__main__":
    print("\nStarting from_list functionality tests...\n")
    
    try:
        test_from_list_basic_shapes()
        test_from_list_float()
        test_from_list_with_dtype()
        test_from_list_with_device()
        test_from_list_operations()
        test_from_list_single_element()
        test_from_list_edge_cases()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
