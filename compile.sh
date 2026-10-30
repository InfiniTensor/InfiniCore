xmake f --ascend-npu=y --aten=y --flash-attn=y --ccl=y --graph=y
xmake build && xmake install
xmake build _infinicore
xmake install _infinicore
pip install -e .
