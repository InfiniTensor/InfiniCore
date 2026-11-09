#!/usr/bin/env bash
set -euo pipefail

PREFIX=${INFINI_ROOT:-/home/wuwei/.infini_fm9g70_port}
CORE_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CCL_SOURCE="$CORE_ROOT/submodules/InfiniCCL"
HCCL_SOURCE=${HCCL_SOURCE:-/root/.infini}
CXX=${CXX:-g++}
TMPDIR=${TMPDIR:-/root/.xmake-tmp}

mkdir -p "$PREFIX/include/infiniccl" "$PREFIX/lib" "$TMPDIR"
cp "$CCL_SOURCE/include/comm.h" "$PREFIX/include/infiniccl/infiniccl.h"
cp "$CCL_SOURCE/include/data_type.h" "$CCL_SOURCE/include/return_status.h" \
   "$PREFIX/include/infiniccl/"
cp "$HCCL_SOURCE/lib/libinfiniccl.so" "$PREFIX/lib/libinfiniccl_hccl.so"
cp "$HCCL_SOURCE/lib/libinfinirt.so" "$PREFIX/lib/libinfinirt_h"

python3 - "$PREFIX/lib/libinfiniccl_hccl.so" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
data = path.read_bytes()
needle = b"libinfinirt.so"
replacement = b"libinfinirt_h" + bytes([0])
if data.count(needle) != 1:
    raise SystemExit("expected exactly one libinfinirt.so dependency")
index = data.index(needle)
path.write_bytes(data[:index] + replacement + data[index + len(needle):])
PY

"$CXX" -shared -fPIC -O2 -std=c++17 \
    -I"$PREFIX/include" -I"$PREFIX/include/infiniccl" \
    "$CORE_ROOT/compat/ascend_ccl/infiniccl_hccl_compat.c" -ldl \
    -Wl,-rpath,"$PREFIX/lib" -o "$PREFIX/lib/libinfiniccl.so"
