# NVIDIA Operator Fix Plan: `erf`, `erfc`, `erfinv`, `matrix_power`, `pixel_shuffle`

## Goal Description
Fix and validate the NVIDIA CUDA implementations of five `infiniop` operators (`erf`, `erfc`, `erfinv`, `matrix_power`, `pixel_shuffle`) so the project:

1. Builds cleanly with `xmake` when configured with NVIDIA GPU support (`--nv-gpu=y`), and
2. Executes the official operator test/benchmark commands on an NVIDIA GPU (target: local RTX 5060 Ti) without crashes or correctness failures, and
3. Preserves all official test files (no edits or bypasses), and
4. Produces a branch-ready set of changes that can be pushed to `2025-autumn-LaiQuan-conquer-T1-1-41`.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: NVIDIA build + install succeeds (no compile/link failures)
  - Positive Tests (expected to PASS):
    - From `InfiniCore/`, `XMAKE_ROOT=y python scripts/install.py --cpu=y --omp=y --nv-gpu=y` completes with exit code `0`.
    - `xmake f -cv --cpu=y --omp=y --nv-gpu=y && xmake -v` completes with no compilation errors in any of:
      - `src/infiniop/ops/erf/**`
      - `src/infiniop/ops/erfc/**`
      - `src/infiniop/ops/erfinv/**`
      - `src/infiniop/ops/matrix_power/**`
      - `src/infiniop/ops/pixel_shuffle/**`
  - Negative Tests (expected to FAIL):
    - Any of the above commands fails due to CUDA compilation errors, missing headers, missing symbols, or type mismatches in the five target operators.

- AC-2: Official tests remain unmodified
  - Positive Tests (expected to PASS):
    - `git diff -- test/infinicore` shows no changes.
    - `git status --porcelain` contains no modified files under `test/infinicore/`.
  - Negative Tests (expected to FAIL):
    - Any change is detected under `test/infinicore/` (edits, deletions, skips, tolerances changed, “golden” outputs changed, or bypass logic added).

- AC-3: Operator descriptors validate inputs and reject invalid configurations (no UB)
  - Positive Tests (expected to PASS):
    - `infiniopCreate*Descriptor` succeeds for valid shapes/dtypes and parameters:
      - `erf`, `erfc`, `erfinv`: same-shape `x -> y`, supported dtypes, supports both contiguous and strided descriptors.
      - `matrix_power`: square 2D matrices (and any higher-rank batch semantics supported by the existing descriptor logic), `n >= 0`.
      - `pixel_shuffle`: 4D NCHW, `C_in % (upscale_factor^2) == 0`.
    - `infiniopGet*WorkspaceSize` returns a deterministic value; if workspace is required, it is non-zero and sufficient for the implementation.
  - Negative Tests (expected to FAIL):
    - `pixel_shuffle` descriptor creation returns an error for `C_in` not divisible by `upscale_factor^2`.
    - `matrix_power` descriptor creation returns an error for non-square matrices and/or `n < 0` (unless the repo already defines a different contract).
    - Any invalid input returns a non-success `infiniStatus_t` rather than crashing or launching a kernel that reads/writes out of bounds.

- AC-4: GPU correctness matches reference within defined tolerances
  - Positive Tests (expected to PASS):
    - A local validation harness (outside `test/infinicore/`) can run each operator on NVIDIA and compare against a reference:
      - `erf`, `erfc`: compare vs CUDA/PyTorch reference (`torch.erf`, `torch.erfc`) for `float16`, `bfloat16`, `float32` with the same tolerance envelopes used in `test/infinicore/ops/*.py`.
      - `erfinv`: generate inputs strictly within `(-1, 1)` and compare vs `torch.erfinv`.
      - `matrix_power`: compare vs `torch.matrix_power` for `n ∈ {0,1,2,3,5}` on small square matrices.
      - `pixel_shuffle`: compare vs `torch.nn.functional.pixel_shuffle` with multiple upscale factors and include at least one strided input layout.
  - Negative Tests (expected to FAIL):
    - The harness flags mismatches outside tolerance (numerical error, NaN/Inf handling differences), and the run is considered failed until corrected.

- AC-5: GPU runtime safety (no illegal access / deterministic completion)
  - Positive Tests (expected to PASS):
    - Running the local harness (AC-4) under a CUDA memory checker (e.g., `compute-sanitizer`) reports no out-of-bounds reads/writes, misaligned accesses, or illegal memory access.
    - Re-running the same harness inputs produces stable results (no sporadic failures).
  - Negative Tests (expected to FAIL):
    - Any `cudaErrorIllegalAddress`, “misaligned address”, or sanitizer-reported OOB access occurs for valid inputs.

- AC-6: Official benchmark runner completes on NVIDIA without failures
  - Positive Tests (expected to PASS):
    - `python test/infinicore/run.py --ops erf erfc erfinv matrix_power pixel_shuffle --nvidia --bench` completes with exit code `0` (no `FAILED` operators reported).
    - Benchmark output includes timing summaries (host and/or device) without crashing.
  - Negative Tests (expected to FAIL):
    - The runner exits non-zero, crashes, or reports any of the five operators as `FAILED`.

- AC-7: Deliverable branch is push-ready
  - Positive Tests (expected to PASS):
    - Changes are committed with a clear message and `git push origin 2025-autumn-LaiQuan-conquer-T1-1-41` succeeds.
  - Negative Tests (expected to FAIL):
    - Push is rejected due to wrong branch name, missing permissions, or non-fast-forward errors (must be resolved before “done”).

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
A CUDA implementation for all five operators that is both correct and performance-aware on NVIDIA GPUs:

- Correct strided-tensor support (honors descriptor stride metadata; does not assume contiguous inputs).
- Numerically robust `erfinv` (good initial approximation + refinement steps to meet tolerances across supported dtypes).
- `matrix_power` implemented with exponentiation-by-squaring and an efficient GEMM path (reusing existing kernels or CUDA libraries already used by the project).
- `pixel_shuffle` implemented as a bandwidth-efficient kernel with well-tuned launch parameters.
- Verified with a local harness and a CUDA memory checker for safety.

### Lower Bound (Minimum Acceptable Scope)
Minimal fixes that:

- Restore successful compilation and linking for NVIDIA (`--nv-gpu=y`), and
- Produce correct outputs within tolerance for the input ranges used by the official benchmarks, and
- Avoid any modifications to official tests.

Performance tuning beyond “not obviously slow” is optional in the minimum scope.

### Allowed Choices
- Can use:
  - Existing InfiniCore/InfiniOP utilities and device abstractions (descriptor metadata, stream handling, dtype utilities).
  - Standard CUDA C++ + CUDA math functions where applicable (`erff`, `erfcf`, double variants).
  - Mixed-precision strategies where the project already uses them (compute in `float` for `half/bfloat16`, then cast back).
  - Existing GEMM/matmul infrastructure already present in the repository for `matrix_power` (preferred over introducing new dependencies).
- Cannot use:
  - Any edits to official tests under `test/infinicore/` to manufacture a pass.
  - Closed-source third-party acceleration libraries not already part of the repository/toolchain.
  - “Silent fallback” behavior that masks incorrect execution (e.g., returning success while skipping GPU work).

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
1. **Reproduce build failures deterministically**:
   - Configure with `xmake f -cv --nv-gpu=y` and rebuild with verbose output to capture the first failing translation unit.
2. **Fix compilation first, then correctness**:
   - Resolve missing includes, wrong namespaces, and CUDA compilation constraints (`__host__`/`__device__`, `constexpr`, `std::` usage, etc.) before tuning kernels.
3. **Operator-by-operator bring-up**:
   - Start with `erf` and `erfc` (elementwise, easiest to validate).
   - Then implement/repair `pixel_shuffle` (indexing correctness + stride handling).
   - Then `matrix_power` (algorithm + workspace management).
   - Finish with `erfinv` (approximation + accuracy).
4. **Validation strategy**:
   - Because `test/infinicore/ops/*.py` currently uses PyTorch operators as the implemented path (InfiniCore calls are commented out), add a small, separate validation harness outside `test/infinicore/` that exercises the actual `infiniop` APIs on NVIDIA and compares to PyTorch.

### Relevant References
- `include/infiniop/ops/erf.h` / `src/infiniop/ops/erf/nvidia/erf_nvidia.cu` - NVIDIA erf entrypoints and kernels
- `include/infiniop/ops/erfc.h` / `src/infiniop/ops/erfc/nvidia/erfc_nvidia.cu` - NVIDIA erfc entrypoints and kernels
- `include/infiniop/ops/erfinv.h` / `src/infiniop/ops/erfinv/nvidia/erfinv_nvidia.cu` - NVIDIA erfinv entrypoints and kernels
- `include/infiniop/ops/matrix_power.h` / `src/infiniop/ops/matrix_power/nvidia/matrix_power_nvidia.cu` - matrix_power API + NVIDIA implementation
- `include/infiniop/ops/pixel_shuffle.h` / `src/infiniop/ops/pixel_shuffle/nvidia/pixel_shuffle_nvidia.cu` - pixel_shuffle API + NVIDIA implementation
- `src/infiniop/ops/*/operator.cc` - device dispatch, descriptor validation patterns
- `xmake.lua` - build flags (`--nv-gpu`, `--cuda_arch`, `--omp`, `--cpu`)

## Dependencies and Sequence

### Milestones
1. Environment + baseline reproduction
   - Phase A: Confirm toolchain (`xmake`, CUDA toolkit, driver, Python env) and configure with `--nv-gpu=y`.
   - Phase B: Run `python scripts/install.py ...` to capture the exact compile/link failures for the five operators.
2. Restore successful NVIDIA compilation for the five operators
   - Phase A: Fix compilation errors in the operator directories (headers, templates, CUDA compilation issues).
   - Phase B: Ensure descriptor creation + workspace sizing compiles and is consistent across CPU/NVIDIA builds.
3. Correctness bring-up with a local harness
   - Phase A: Add a small harness outside `test/infinicore/` to run the five operators on NVIDIA and compare to PyTorch.
   - Phase B: Iterate operator fixes until AC-4 and AC-5 pass.
4. Benchmark + final verification
   - Phase A: Run the official benchmark runner command (AC-6) and confirm no failures.
   - Phase B: Verify `git diff -- test/infinicore` is clean, then commit and push (AC-7).

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- Keep error handling explicit: invalid descriptors/args should return non-success `infiniStatus_t` rather than crashing.
- Favor local, operator-scoped fixes first; only touch shared utilities if multiple operators share the same root cause.

--- Original Design Draft Start ---

# Operator Development Plan (erf, erfc, erfinv, matrix_power, pixel_shuffle)

## Goal Description
Fix, optimize, and successfully execute the 5 currently broken operators (`erf`, `erfc`, `erfinv`, `matrix_power`, `pixel_shuffle`) on a local NVIDIA RTX 5060Ti GPU. The objective is to ensure the codebase compiles properly, passes all official benchmark tests without modifying any built-in test cases, and to push the final working modifications to the target remote repository and branch (`2025-autumn-LaiQuan-conquer-T1-1-41`).

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Successful Library and Operator Compilation
  - Positive Tests (expected to PASS):
    - Executing `XMAKE_ROOT=y python scripts/install.py --omp=y --cpu=y --nv-gpu=y` completes successfully with no syntax errors, undefined references, or fatal aborts in the terminal.
  - Negative Tests (expected to FAIL):
    - Compilation halts due to C++/CUDA syntax errors, missing headers, or type mismatches in any of the 5 targeted operator files.
- AC-2: Official Benchmark Tests Execution
  - Positive Tests:
    - Executing `python test/infinicore/run.py --ops erf,erfc,erfinv,matrix_power,pixel_shuffle --nv-gpu --bench` runs successfully, printing "PASS" and the benchmark performance metrics for all 5 operators.
  - Negative Tests:
    - The test script crashes due to runtime errors (e.g., CUDA out-of-bounds memory access, segmentation fault, illegal memory access) or fails the official assertions due to incorrect mathematical logic or precision limits.
- AC-3: Strict Preservation of Official Test Cases
  - Positive Tests:
    - Git status and diff show zero modifications, deletions, or bypasses to the official test cases located in the `test/infinicore/` directory.
  - Negative Tests:
    - Built-in test cases or the official test scripts are found to be modified to achieve a false positive pass.
- AC-4: Code Submission and Remote Push
  - Positive Tests:
    - Successfully committing and running `git push` to upload all local changes to the `2025-autumn-LaiQuan-conquer-T1-1-41` branch of the `git@github.com:LaiQuan-conquer/InfiniCore.git` repository.
  - Negative Tests:
    - Push gets rejected by the remote server due to incorrect branch naming, missing permissions, or non-fast-forward tracking errors.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
A highly optimized CUDA implementation for all five operators that fully utilizes the shared memory and parallel computing capabilities of the local RTX 5060Ti. The code gracefully handles complex index calculations and memory boundaries (especially for `pixel_shuffle` and `matrix_power`), uses robust numerical approximations for inverse error functions, achieves optimal computational performance in the benchmark tests, and features clean formatting with proper grid/block dimension tuning.

### Lower Bound (Minimum Acceptable Scope)
A fundamentally sound algorithmic implementation that resolves all existing syntax and compilation bugs, correctly computes the required mathematical outputs within acceptable error margins, and successfully passes the target test commands on the local GPU, satisfying the minimum requirements for the competition without over-engineering.

### Allowed Choices
- Can use: Standard CUDA C/C++ programming paradigms, intrinsic CUDA math functions (like `erff()`, `erfcf()`), existing mathematical helper functions/macros within the InfiniCore framework, and local profiling/debugging commands (e.g., `nvidia-smi`).
- Cannot use: Any modifications to the official test scripts (including `run.py` and its dependencies), alterations to the built-in test cases, or unauthorized closed-source third-party acceleration libraries.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
1. **Compilation Troubleshooting**: Address the immediate "cannot compile" issue by inspecting the terminal logs from `install.py`. Fix fundamental C++ issues such as missing header includes, uninitialized pointers, or kernel parameter mismatches.
2. **Operator-by-Operator Execution**:
   - `erf` / `erfc`: These are standard error functions. Ensure you are correctly leveraging the built-in CUDA math library functions mapped to the appropriate precision (float vs double) arrays to avoid precision loss.
   - `erfinv`: The inverse error function requires careful handling. If not provided directly by the target CUDA runtime version, you may need a robust rational polynomial approximation or to map it through inverse cumulative distribution functions.
   - `matrix_power`: This involves repeated matrix multiplication. Pay attention to memory management to avoid allocating excessive temporary buffers on the device. Consider implementing binary exponentiation (exponentiation by squaring) for performance if the power is large.
   - `pixel_shuffle`: This operation reshapes and rearranges elements. Focus heavily on index arithmetic to correctly map elements from the input tensor shape to the output tensor shape (handling the upscaling factor accurately).
3. **Iterative Testing**: Isolate the operators using the provided test script (e.g., test individually via `--ops pixel_shuffle`). Debug logic errors sequentially before proceeding to the combined full benchmark validation.

### Relevant References
- The source code directory of the kernel implementations to locate and refactor the currently non-functional logic.
- Framework-level common header files to utilize established memory access patterns.

## Dependencies and Sequence

### Milestones
1. Environment Configuration and Compilation Fixes
   - Phase A: Run the installation script and collect the initial compilation error logs for the 5 operators.
   - Phase B: Systematically patch syntax, template, and type errors until `install.py` executes successfully on the local environment.
2. Logic Correction and Individual Operator Verification
   - Phase A: Run the test command for each operator individually to debug and correct the mathematical kernels.
   - Phase B: Strictly verify via Git that the official built-in test case files remain untouched.
3. Benchmark Validation and Remote Submission
   - Phase A: Execute the full benchmark test command to confirm that the performance and outputs of all 5 operators pass.
   - Phase B: Commit the finalized code and push it to the designated Git repository and `2025-autumn-LaiQuan-conquer-T1-1-41` branch.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are strictly for plan documentation only.
- Use descriptive, mathematical, and domain-appropriate naming conventions within the actual C++/CUDA codebase.

--- Original Design Draft End ---
