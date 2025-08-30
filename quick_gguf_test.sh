# #!/bin/bash
# set -e

# ROOT=~/T1-GGUF/T1-2-3
# TEST_BIN=~/.infini/bin/infiniop-test
# NODE=gpua19

# PASSED=()
# PARTIAL=()

# for gguf in $(find "$ROOT" -type f -name "*.gguf"); do
#     if [[ $gguf == *bfloat16* ]]; then
#         atol=1e-2; rtol=1e-2
#     elif [[ $gguf == *float32* ]]; then
#         atol=1e-4; rtol=1e-4
#     elif [[ $gguf == *float16* ]]; then
#         atol=1e-3; rtol=1e-3
#     elif [[ $gguf == *int8* ]]; then
#         atol=1; rtol=1
#     else
#         echo "⚠️ 未知精度, 跳过: $gguf"
#         continue
#     fi

#     echo ">>> Running test: $gguf (atol=$atol, rtol=$rtol)"
#     OUTPUT=$(srun --nodelist=$NODE \
#         $TEST_BIN "$gguf" --nvidia --warmup 1 --run 1 \
#         --atol $atol --rtol $rtol 2>&1)

#     echo "$OUTPUT"

#     if echo "$OUTPUT" | grep -q "All tests passed"; then
#         PASSED+=("$gguf")
#     elif echo "$OUTPUT" | grep -q "failed"; then
#         PARTIAL+=("$gguf")
#     else
#         echo "⚠️ 无法判断测试结果: $gguf"
#     fi
# done

# echo ""
# echo "================= 测试总结 ================="
# echo "✅ 全部通过的测例:"
# for f in "${PASSED[@]}"; do
#     echo "  $f"
# done

# echo ""
# echo "⚠️ 部分通过的测例:"
# for f in "${PARTIAL[@]}"; do
#     echo "  $f"
# done
# echo "============================================"


#!/bin/bash
set -u  # 不允许未定义变量，但不会因为命令失败退出

ROOT=~/T1-GGUF/T1-2-3
TEST_BIN=~/.infini/bin/infiniop-test
NODE=gpua19

PASSED=()
PARTIAL=()
UNKNOWN=()

for gguf in $(find "$ROOT" -type f -name "*.gguf"); do
    if [[ $gguf == *bfloat16* ]]; then
        atol=1e-2; rtol=1e-2
    elif [[ $gguf == *float32* ]]; then
        atol=1e-4; rtol=1e-4
    elif [[ $gguf == *float16* ]]; then
        atol=1e-3; rtol=1e-3
    elif [[ $gguf == *int8* ]]; then
        atol=1; rtol=1
    else
        echo "⚠️ 未知精度, 跳过: $gguf"
        continue
    fi

    echo ">>> Running test: $gguf (atol=$atol, rtol=$rtol)"

    # 执行并捕获输出和返回值
    OUTPUT=$(srun --nodelist=$NODE \
        $TEST_BIN "$gguf" --nvidia --warmup 1 --run 1 \
        --atol $atol --rtol $rtol 2>&1)
    RET=$?

    echo "$OUTPUT"

    if echo "$OUTPUT" | grep -q "All tests passed"; then
        PASSED+=("$gguf")
    elif echo "$OUTPUT" | grep -q "failed"; then
        PARTIAL+=("$gguf")
    else
        echo "⚠️ 无法判断测试结果: $gguf (exit code=$RET)"
        UNKNOWN+=("$gguf")
    fi
done

echo ""
echo "================= 测试总结 ================="
echo "✅ 全部通过的测例:"
for f in "${PASSED[@]}"; do
    echo "  $f"
done

echo ""
echo "⚠️ 部分通过的测例:"
for f in "${PARTIAL[@]}"; do
    echo "  $f"
done

echo ""
echo "❓ 未能识别结果的测例:"
for f in "${UNKNOWN[@]}"; do
    echo "  $f"
done
echo "============================================"
