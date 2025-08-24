import torch
import numpy as np

# Simple test case
input_tensor = torch.tensor([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]], dtype=torch.float32, device='cuda')
src_tensor = torch.tensor([[10.0, 20.0],
                          [30.0, 40.0]], dtype=torch.float32, device='cuda')
index_tensor = torch.tensor([[0, 1],
                            [1, 0]], dtype=torch.int64, device='cuda')

print("Input tensor:")
print(input_tensor)
print("\nSrc tensor:")
print(src_tensor)
print("\nIndex tensor:")
print(index_tensor)

# PyTorch scatter
output_torch = input_tensor.clone()
output_torch.scatter_(0, index_tensor, src_tensor)
print("\nPyTorch scatter result:")
print(output_torch)

# Expected behavior:
# For dim=0, index_tensor[i,j] tells us which row in output to place src_tensor[i,j]
# index_tensor[0,0] = 0 -> output[0,0] = src[0,0] = 10.0
# index_tensor[0,1] = 1 -> output[1,1] = src[0,1] = 20.0
# index_tensor[1,0] = 1 -> output[1,0] = src[1,0] = 30.0
# index_tensor[1,1] = 0 -> output[0,1] = src[1,1] = 40.0

print("\nExpected result:")
print("output[0,0] = src[0,0] = 10.0")
print("output[0,1] = src[1,1] = 40.0")
print("output[1,0] = src[1,0] = 30.0")
print("output[1,1] = src[0,1] = 20.0")