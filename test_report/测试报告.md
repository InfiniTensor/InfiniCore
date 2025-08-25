完成了算子赛题 T1-1-1 ：
- Exp
- Sin
- Cos
- LeakyRelu
- Tanh
- Sigmoid Backward
- HardSwish
- Cast
- Where

在内的所有9个算子的实现，以及 T1-1-2的：

……

并通过了CPU、Metax平台中Pytorch单元测试代码的验证。


# PyTorch单元测试
## 1. Exp算子测试

- 测试能够覆盖多种输入输出的形状以及排布且已通过CPU、Metax平台上的PyTorch测试
- 支持数据类型涵盖了f32, f16, bf16


### CPU平台测试结果
![alt text](image.png)
……
![alt text](image-1.png)

### Metax平台测试结果
![alt text](image-2.png)
……
![alt text](image-3.png)

## 2. Sin算子测试
- 测试能够覆盖多种输入输出的形状以及排布且已通过CPU、Metax平台上的PyTorch测试
- 支持数据类型涵盖了f32, f16, bf16

### CPU平台测试结果

![](sin-cpu-1.png)

……

![](sin-cpu-2.png)

### Metax平台测试结果

![](sin-metax-1.png)

……

![](sin-metax-2.png)

## 3. Cos算子测试

- 测试能够覆盖多种输入输出的形状以及排布且已通过CPU、Metax平台上的PyTorch测试
- 支持数据类型涵盖了f32, f16, bf16
### CPU平台测试结果
![alt text](image-6.png)
……
![alt text](image-7.png)

### Metax平台测试结果
![alt text](image-4.png)
……
![alt text](image-5.png)

## 4. LeakyRelu算子测试
- 测试能够覆盖多种输入输出的形状以及排布且已通过CPU、Metax平台上的PyTorch测试
- 支持数据类型涵盖了f32, f16, bf16
### CPU平台测试结果
![alt text](image-8.png)
……
![alt text](image-9.png)
### Metax平台测试结果
![alt text](image-10.png)
……
![alt text](image-11.png)

## 5. Tanh算子测试
- 测试能够覆盖多种输入输出的形状以及排布且已通过CPU、Metax平台上的PyTorch测试
- 支持数据类型涵盖了f32, f16, bf16

### CPU平台测试结果

![](tanh-cpu-1.png)

……

![](tanh-cpu-2.png)

### Metax平台测试结果

![](tanh-metax-1.png)

……

![](tanh-metax-2.png)

## 6. Sigmoid Backward算子测试
- 测试能够覆盖多种输入输出的形状以及排布且已通过CPU、Metax平台上的PyTorch测试
- 支持数据类型涵盖了f32, f16, bf16
### CPU平台测试结果
![alt text](image-12.png)
……
![alt text](image-13.png)

### Metax平台测试结果
![alt text](image-14.png)
……
![alt text](image-15.png)

## 7. HardSwish算子测试
- 测试能够覆盖多种输入输出的形状以及排布且已通过CPU、Metax平台上的PyTorch测试
- 支持数据类型涵盖了f32, f16, bf16
### CPU平台测试结果
![alt text](image-16.png)
……
![alt text](image-17.png)
### Metax平台测试结果
![alt text](image-18.png)
……
![alt text](image-19.png)

## 8. Cast算子测试
- 测试能够覆盖多种输入输出的形状以及排布且已通过CPU、Metax平台上的PyTorch测试
- 支持整数类型 (int32, int64, uint32, uint64) 之间互转
- 支持浮点类型 (f32, f16, f64) 之间互转
- 支持整数类型 (int32, int64, uint32, uint64) 到浮点类型 (f32, f16, f64) 的互转
### CPU平台测试结果
![alt text](image-20.png)
……
![alt text](image-21.png)
### Metax平台测试结果
![alt text](image-22.png)
……
![alt text](image-23.png)

## 9. Where算子测试

- 测试能够覆盖多种输入输出的形状以及排布且已通过CPU、Metax平台上的PyTorch测试
- 支持数据类型涵盖了f32, f16,f64, bf16,bool,I8,I16,I32,I64

### CPU平台测试结果

![](where-cpu-1.png)

……

![](where-cpu-2.png)

### Metax平台测试结果

![](where-metax-1.png)

……

![](where-metax-2.png)

