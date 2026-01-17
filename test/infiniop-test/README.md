# InfiniOP 测例生成

## 介绍

使用 python 脚本生成包含测例的 `.gguf` 文件，并使用 `infiniop-test` 程序进行测试。可以手动执行各个步骤，也可以使用自动化脚本一次性完成整个流程。

## 运行方式

- 编译 `infiniop-test` 程序

```bash
xmake build infiniop-test
```

- 生成测例

在`/test/infiniop-test/`目录执行矩阵乘测例生成脚本，执行结束以后会在`/test/infiniop-test/`目录生成`gemm.gguf`测例文件。

```bash
cd /test/infiniop-test/
python -m test_generate.testcases.gemm
```

- 测试测例

打印测试程序用法

```bash
infiniop-test --help
```

示例：在CPU上测试`gemm.gguf`测例文件，预热20次，测试1000次。

```bash
infiniop-test gemm.gguf --cpu --warmup 20 --run 1000
```

## 自动化运行方式

使用 python 脚本自动化完成上述所有步骤，包括编译、生成测例和在 CPU 上进行测试。

```bash
cd /test/infiniop-test/
python -m test_generate.auto_gemm_test --overwrite
```

### 参数说明

`--config` 模型列表配置文件，指定 `HuggingFace patterns`，默认值为 `config-hub.json`，示例如下。

```bash
{
  "facebook/opt-125m": [ "decoder.layers.*.self_attn.q_proj.weight" ],
  "t5-small": [ "encoder.block.*.layer.0.SelfAttention.q.weight" ]
}
```

`--model-path` 模型存储根目录，若指定，则扫描该目录下的模型文件夹而不使用模型列表配置文件，默认值为 `None`。

`--output` 输出目录，默认值为 `gguf_output`。

`--warmup` 预热次数，类型为整数，默认值为 20。

`--run` 测试次数，类型为整数，默认值为 1000。

`--overwrite` 覆盖已有输出，使用该参数时会覆盖已有的输出文件。

## 自定义测例

### GGUF文件格式

```text
GGUF File Contents:
Version: 3
Number of Meta KVs: 8
Number of Tensors: 4

Meta KVs:
Key: general.architecture, Type: GGUF_TYPE_STRING, Value: infiniop-test
Key: test_count, Type: GGUF_TYPE_UINT64, Value: 1
Key: test.0.op_name, Type: GGUF_TYPE_STRING, Value: matmul
Key: test.0.a.strides, Type: GGUF_TYPE_INT32, Value: [1, 5]
Key: test.0.b.strides, Type: GGUF_TYPE_INT32, Value: [1, 6]
Key: test.0.c.strides, Type: GGUF_TYPE_INT32, Value: [1, 6]
Key: test.0.alpha, Type: GGUF_TYPE_FLOAT32, Value: 1.000000
Key: test.0.beta, Type: GGUF_TYPE_FLOAT32, Value: 0.000000

Tensor INFOs:
Name: test.0.a, NDims: 2, Shape: [5, 4], DataType: F32, DataOffset: 0
Name: test.0.b, NDims: 2, Shape: [6, 5], DataType: F32, DataOffset: 96
Name: test.0.c, NDims: 2, Shape: [6, 4], DataType: F32, DataOffset: 224
Name: test.0.ans, NDims: 2, Shape: [6, 4], DataType: F64, DataOffset: 320
```

- `Meta` 中必须包含 `test_count` ，表示测例数量。
- 每个测例的 `Meta` 和 `Tensor` 名字以 `test.[id].` 开头，后接具体信息名称。数字 `[id]` 表示测例编号。编号必须为 0 到 test_count-1.
- `Tensor` 名字接 `.strides` 表示步长，若没有则默认为连续。
- 注意：gguf 中的 shape 和 stride 的存储方向是反向的，第一个数代表最后一维。

### GGUF测例构建要求

不参与计算的 `Tensor` 不应存储数据，避免 `GGUF` 文件中出现冗余内容。
此类 `Tensor` 应使用 `np.empty(tuple(0 for _ in shape), dtype=dtype)` 构造其数据字段,  且 `GGUF` 需存储此张量的形状数据 `.shape`、步长数据 `.strides`，否则无法成功构建，可使用 `contiguous_gguf_strides(shape)` 计算步长数据。

对于 `Elementwise` 算子，需包含零步长（zero-stride）测试。对于步长为0的张量，`GGUF` 不应存储冗余广播数据，可使用 `process_zero_stride_tensor`进行冗余数据移除，同时必须在 `GGUF` 中提供此张量的实际形状数据 `.shape`，否则无法成功构建。
