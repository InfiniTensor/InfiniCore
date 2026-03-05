# InfiniCore Operator Fix Plan (NVIDIA): bitwise_right_shift, gaussian_nll_loss, interpolate, prelu, relu6

## Goal Description
Fix the five currently broken operators (`bitwise_right_shift`, `gaussian_nll_loss`, `interpolate`, `prelu`, `relu6`) so they compile and run correctly on the NVIDIA backend and pass the official InfiniCore benchmark/test runner, without modifying any official test files. Validate locally on an NVIDIA GPU (target hardware per draft: RTX 5060 Ti) and submit the final changes by pushing to the target branch `2025-autumn-LaiQuan-conquer-T1-1-30` of `git@github.com:LaiQuan-conquer/InfiniCore.git`.

> **Working directory assumption**: All commands below are intended to be run from `InfiniCore/` (the repo root that contains `scripts/`, `src/`, and `test/`).

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: NVIDIA build/install completes successfully
  - Positive Tests (expected to PASS):
    - `XMAKE_ROOT=y python scripts/install.py --omp=y --cpu=y --nv-gpu=y` exits with code 0 and produces the expected build artifacts (no compile/link errors).
    - Re-running the same command after a clean build (or after touching only one of the five ops) still succeeds.
  - Negative Tests (expected to FAIL):
    - The install step fails with compilation/linkage errors in any of the five operator implementations (e.g., `src/infiniop/ops/*/nvidia/*.cu`).
    - The install step succeeds only when disabling NVIDIA support (e.g., `--nv-gpu=n`), indicating the NVIDIA path is still broken.

- AC-2: Each operator passes the official runner individually on NVIDIA
  - Positive Tests (expected to PASS):
    - `python test/infinicore/run.py --ops bitwise_right_shift --nvidia --bench`
    - `python test/infinicore/run.py --ops gaussian_nll_loss --nvidia --bench`
    - `python test/infinicore/run.py --ops interpolate --nvidia --bench`
    - `python test/infinicore/run.py --ops prelu --nvidia --bench`
    - `python test/infinicore/run.py --ops relu6 --nvidia --bench`
  - Negative Tests (expected to FAIL):
    - Any single-op run crashes (CUDA error, illegal memory access, segfault) or reports a correctness failure for that operator.

- AC-3: All five operators pass together on NVIDIA
  - Positive Tests (expected to PASS):
    - `python test/infinicore/run.py --ops bitwise_right_shift gaussian_nll_loss interpolate prelu relu6 --nvidia --bench` completes successfully with all operators reported as passing.
  - Negative Tests (expected to FAIL):
    - The combined run fails even though the single-op runs pass, indicating cross-op state, build configuration, or registration issues.

- AC-4: No GPU runtime correctness/safety errors under debug execution
  - Positive Tests (expected to PASS):
    - `CUDA_LAUNCH_BLOCKING=1 python test/infinicore/run.py --ops bitwise_right_shift gaussian_nll_loss interpolate prelu relu6 --nvidia --bench` passes (useful for surfacing latent async CUDA errors).
  - Negative Tests (expected to FAIL):
    - The debug run reports kernel launch failures, out-of-bounds accesses, or other runtime errors that are otherwise masked by async execution.

- AC-5: Official tests remain unmodified (no bypassing)
  - Positive Tests (expected to PASS):
    - `git diff --name-only -- test/infinicore` returns empty output.
    - `git status --porcelain` shows no changes under `test/infinicore/`.
  - Negative Tests (expected to FAIL):
    - Any file under `test/infinicore/` is modified, added, deleted, or the runner is edited to special-case these five operators.

- AC-6: Submission branch is pushed successfully
  - Positive Tests (expected to PASS):
    - `git status` is clean after committing the intended changes.
    - `git push origin 2025-autumn-LaiQuan-conquer-T1-1-30` succeeds (or the configured remote equivalent).
  - Negative Tests (expected to FAIL):
    - Push is rejected due to wrong branch name, permissions, or non-fast-forward conflicts.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
A robust and performant NVIDIA implementation for all five operators:
- Correct for all supported dtypes/shapes described by the public APIs (including edge cases like empty tensors, boundary conditions, and broadcasting semantics where applicable).
- Uses safe and performant CUDA kernels (appropriate launch config, coalesced loads where possible, minimal divergent branches).
- Keeps CPU and NVIDIA implementations behaviorally consistent.

### Lower Bound (Minimum Acceptable Scope)
The minimal set of changes that:
- Fixes compilation/link issues and obvious runtime errors in the NVIDIA implementations.
- Produces correct outputs sufficient to pass the official runner/bench commands in AC-2 and AC-3 on an NVIDIA GPU.
- Avoids refactors outside the five operators unless required to restore correctness/buildability.

### Allowed Choices
- Can use:
  - Existing InfiniCore operator patterns in `src/infiniop/ops/*/operator.cc` and per-backend implementations.
  - CPU implementations as a correctness reference (`src/infiniop/ops/*/cpu/*`).
  - Existing shared CUDA utilities already present in the repo (e.g., `src/infiniop/ops/*/cuda/kernel.cuh`).
- Cannot use:
  - Any modifications to official test code under `test/infinicore/` (including `test/infinicore/run.py` and test cases under `test/infinicore/ops/`).
  - Hard-coded outputs, special-casing only the benchmark inputs, or bypassing correctness checks.
  - Closed-source third-party acceleration libraries or changes that require non-standard external dependencies.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
1. **Reproduce and localize failures**
   - Run AC-1 once to capture the first compile error (fix in strict order: the first fatal error usually unblocks the next).
   - Run AC-2 per-operator to determine whether issues are compile-time, registration/dispatch, or kernel correctness.
2. **Follow the established operator pattern**
   - For each operator, inspect `operator.cc` to understand descriptor parsing, type dispatch, and backend selection.
   - Use the CPU implementation as a reference for semantics and edge-case handling.
3. **Fix NVIDIA kernels with correctness first**
   - Use `CUDA_LAUNCH_BLOCKING=1` during debugging; consider CUDA sanitizers if available to catch OOB and race conditions.
   - Common pitfalls:
     - `bitwise_right_shift`: signed vs unsigned shift behavior; shift amounts outside bit-width; vectorization assumptions.
     - `gaussian_nll_loss`: numerical stability (variance/eps), avoiding NaNs/Infs, correct reduction semantics.
     - `interpolate`: coordinate mapping, align-corners behavior, bounds handling and off-by-one indices.
     - `prelu`: broadcasting slope parameters correctly across tensor shapes; datatype promotion rules.
     - `relu6`: clamping behavior and dtype handling; avoiding overflow/precision surprises.
4. **Validate end-to-end**
   - After all single-op runs pass, run the combined benchmark (AC-3) and the debug-mode run (AC-4).
   - Before committing, verify the test tree is unchanged (AC-5).

### Relevant References
- `include/infiniop/ops/bitwise_right_shift.h` - public API expectations for the operator
- `include/infiniop/ops/gaussian_nll_loss.h` - public API expectations for the operator
- `include/infiniop/ops/interpolate.h` - public API expectations for the operator
- `include/infiniop/ops/prelu.h` - public API expectations for the operator
- `include/infiniop/ops/relu6.h` - public API expectations for the operator
- `src/infiniop/ops/bitwise_right_shift/` - implementation (CPU + NVIDIA + shared kernel helpers)
- `src/infiniop/ops/gaussian_nll_loss/` - implementation (CPU + NVIDIA + shared kernel helpers)
- `src/infiniop/ops/interpolate/` - implementation (CPU + NVIDIA + shared kernel helpers)
- `src/infiniop/ops/prelu/` - implementation (CPU + NVIDIA + shared kernel helpers)
- `src/infiniop/ops/relu6/` - implementation (CPU + NVIDIA + shared kernel helpers)
- `test/infinicore/run.py` - official runner (read-only for this task)

## Dependencies and Sequence

### Milestones
1. **Baseline reproduction**
   - Phase A: Run AC-1 to confirm current build errors; capture logs.
   - Phase B: Run AC-2 for each operator to establish a pass/fail matrix.
2. **Per-operator fixes (correctness first)**
   - Phase A: Fix compilation/registration issues so each operator can execute on NVIDIA.
   - Phase B: Iterate on kernel logic until each single-op benchmark passes (AC-2).
3. **System validation and submission**
   - Phase A: Run combined benchmark (AC-3) and debug execution (AC-4).
   - Phase B: Verify test tree unchanged (AC-5), then commit and push (AC-6).

Describe dependencies as: build must pass (AC-1) before NVIDIA runtime validation (AC-2/3/4), and test immutability (AC-5) gates submission (AC-6).

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

--- Original Design Draft Start ---

# Operator Development Plan (bitwise_right_shift, gaussian_nll_loss, interpolate, prelu, relu6)

## Goal Description
Fix, optimize, and successfully execute the 5 currently broken operators (bitwise_right_shift, gaussian_nll_loss, interpolate, prelu, relu6) on a local NVIDIA RTX 5060Ti GPU. Ensure the codebase compiles properly, passes all official benchmark tests without modifying any built-in test cases, and push the final modifications to the target remote repository and branch (`2025-autumn-LaiQuan-conquer-T1-1-30`).

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Successful Library and Operator Compilation
  - Positive Tests (expected to PASS):
    - Executing `XMAKE_ROOT=y python scripts/install.py --omp=y --cpu=y --nv-gpu=y` completes successfully with no fatal errors in the terminal.
  - Negative Tests (expected to FAIL):
    - Compilation aborts due to C++/CUDA syntax errors, undefined references, or type mismatches in any of the 5 operator files.
- AC-2: Official Benchmark Tests Execution
  - Positive Tests:
    - Executing `python test/infinicore/run.py --ops bitwise_right_shift,gaussian_nll_loss,interpolate,prelu,relu6 --nv-gpu --bench` runs successfully, printing "PASS" and the benchmark performance metrics for all 5 operators.
  - Negative Tests:
    - The test script crashes due to runtime errors (e.g., CUDA out-of-bounds memory access, segmentation fault) or fails the official assertions due to incorrect calculation precision/logic.
- AC-3: Strict Preservation of Official Test Cases
  - Positive Tests:
    - Git status and diff show zero modifications, deletions, or additions to the official test cases located in the `test/infinicore/` directory.
  - Negative Tests:
    - Official test scripts or built-in test cases are found to be modified or bypassed to achieve a false pass.
- AC-4: Code Submission and Push
  - Positive Tests:
    - Successfully committing and running `git push` to upload all local changes to the `2025-autumn-LaiQuan-conquer-T1-1-30` branch of the `git@github.com:LaiQuan-conquer/InfiniCore.git` repository.
  - Negative Tests:
    - Push gets rejected by the remote server due to incorrect branch naming, permission issues, or non-fast-forward updates.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
A highly optimized CUDA implementation for all five operators that fully utilizes the shared memory and vectorized memory access instructions of the RTX 5060Ti. The code handles mathematical edge cases flawlessly, achieves optimal performance in the benchmark tests, and includes clean formatting with proper grid/block dimension setups.

### Lower Bound (Minimum Acceptable Scope)
A fundamental algorithmic implementation that resolves all existing syntax and compilation errors, correctly computes the mathematical results, and successfully passes the target test commands on the local GPU, satisfying the minimum competition requirements without over-engineering.

### Allowed Choices
- Can use: Standard CUDA C/C++ programming paradigms, existing helper functions/macros within the InfiniCore framework, and local profiling/debugging tools (e.g., `nvidia-smi`).
- Cannot use: Any modifications to the official test scripts (including `run.py` and its dependencies), built-in test cases, or unauthorized closed-source third-party acceleration libraries.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
1. **Compilation Troubleshooting**: Address the "cannot compile" issue by targeting the first fatal syntax error in the terminal logs. Fix basic C++ issues such as out-of-bounds pointers, missing includes, or kernel function parameter type mismatches.
2. **Operator-by-Operator Execution**:
   - `bitwise_right_shift`: Focus on correct bitwise operations for various integer types, taking care of logical vs. arithmetic shifts based on the data type.
   - `gaussian_nll_loss`: Ensure numerically stable implementations of logarithmic functions and variance handling to prevent NaN/Inf outputs.
   - `interpolate`: Pay close attention to index mapping, coordinate scaling, and boundary handling for different interpolation modes (e.g., nearest, linear).
   - `prelu` / `relu6`: Implement efficient element-wise activation bounds and weight parameter broadcasting.
3. **Iterative Testing**: Isolate the operators using the provided test script (e.g., test individually via `--ops prelu`). Once an operator passes individually, proceed to combined testing and full benchmark validation.

### Relevant References
- The source code directory of the kernel implementations for refactoring the currently broken logic.
- Framework-level common header files to check for encapsulated memory processing or math interfaces.

## Dependencies and Sequence

### Milestones
1. Environment Configuration and Compilation Fixes
   - Phase A: Run the installation script and collect the compilation error logs for the 5 operators.
   - Phase B: Systematically resolve syntax and type errors until `install.py` executes successfully.
2. Logic Correction and Individual Operator Verification
   - Phase A: Run the test command for each operator individually to debug mathematical logic errors.
   - Phase B: Strictly verify that the official built-in test case files remain untouched.
3. Benchmark Validation and Remote Submission
   - Phase A: Execute the full benchmark test command to confirm that the performance and results of all 5 operators pass.
   - Phase B: Commit the finalized code and push it to the designated Git repository and `2025-autumn-LaiQuan-conquer-T1-1-30` branch.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

--- Original Design Draft End ---
