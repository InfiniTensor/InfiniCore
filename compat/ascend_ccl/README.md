# Ascend HCCL CCL compatibility

The pinned upstream InfiniCCL source has no Ascend runtime backend, while the
validated FM9G70 baseline uses an HCCL-enabled InfiniCCL binary with the older
C API. This adapter exposes the current InfiniCCL API and forwards the calls
used by InfiniLM to that HCCL implementation.

Run inside the Ascend container with the coherent Infini prefix in
`INFINI_ROOT`, and make sure that prefix is in `LD_LIBRARY_PATH` before
starting InfiniLM. The script copies the HCCL binary and its legacy InfiniRT
dependency from `/root/.infini` by default; override `HCCL_SOURCE` when that
validated binary is stored elsewhere.
