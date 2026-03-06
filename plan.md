# Operator Fix & Benchmark Plan (diff, digamma, dist, logdet, pad)

## Goal Description
Fix, optimize where feasible, and successfully execute the five targeted operators (`diff`, `digamma`, `dist`, `logdet`, `pad`) on a local NVIDIA CUDA GPU (target hardware: RTX 5060 Ti or equivalent). The finished work must:

- Build cleanly with the NVIDIA backend enabled via xmake.
- Pass the official Python operator test runner for the targeted ops on NVIDIA (including benchmark mode).
- Preserve the integrity of the official test suite (no edits to checked-in tests to force a pass).
- Be ready to push to the target remote branch `2025-autumn-LaiQuan-conquer-T1-1-37`.

Important repo-specific detail:
- Build configuration uses the xmake option `--nv-gpu=y` (as defined in `InfiniCore/xmake.lua`).
- The Python test runner selects NVIDIA via `--nvidia` (in `InfiniCore/test/infinicore/run.py`), not `--nv-gpu`.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Successful NVIDIA build (library + operator tests)
  - Positive Tests (expected to PASS):
    - From repo root: `cd InfiniCore && python scripts/install.py --omp=y --cpu=y --nv-gpu=y` completes with exit code 0.
    - Re-running `cd InfiniCore && xmake -r` completes with exit code 0 (confirms the configured toolchain stays consistent).
  - Negative Tests (expected to FAIL):
    - Any C++/CUDA compile error, missing header, undefined reference, or xmake configuration failure occurs during the install/build process.

- AC-2: Correctness for `diff`, `digamma`, `dist`, `logdet` on NVIDIA via the official runner
  - Positive Tests (expected to PASS):
    - `cd InfiniCore && python test/infinicore/run.py --ops diff digamma dist logdet --nvidia` exits with code 0 and reports no failed/partial/skipped cases in the final summary.
    - `cd InfiniCore && python test/infinicore/run.py --ops diff digamma dist logdet --nvidia --verbose` exits with code 0 (helps ensure the run is stable when configured to stop on first error).
  - Negative Tests (expected to FAIL):
    - Any operator produces wrong shapes/values vs PyTorch outside the test tolerances, triggers NaN/Inf unexpectedly, or crashes (segfault / CUDA illegal memory access).

- AC-3: `pad` correctness on NVIDIA (requires clarifying the evaluation path)
  - Background / issue to resolve:
    - The checked-in test file `InfiniCore/test/infinicore/ops/pad.py` currently does not implement `infinicore_operator` (it is commented out), which causes a "partial" result and fails the overall run with the current framework logic.
  - Option A (if `pad.py` is part of the official evaluation suite and must pass in local-scan mode):
    - Positive Tests (expected to PASS):
      - `cd InfiniCore && python test/infinicore/run.py --ops pad --nvidia` exits with code 0 and reports no failed/partial/skipped cases.
    - Negative Tests (expected to FAIL):
      - Any "partial" test result (InfiniCore operator missing), output mismatch vs `torch.nn.functional.pad`, or runtime crash.
  - Option B (if checked-in tests must remain byte-for-byte unchanged and `pad.py` is intentionally incomplete):
    - Positive Tests (expected to PASS):
      - Provide JSON-based pad cases and run them via the existing dynamic mode:
        - `cd InfiniCore && python test/infinicore/run.py --load <path/to/pad_cases.json> --nvidia` exits with code 0.
    - Negative Tests (expected to FAIL):
      - Any mismatch vs PyTorch pad semantics for the supported modes (`constant`, `reflect`, `replicate`, `circular`) or any runtime crash.

- AC-4: Benchmark mode completes on NVIDIA for the targeted operators
  - Positive Tests (expected to PASS):
    - `cd InfiniCore && python test/infinicore/run.py --ops diff digamma dist logdet pad --nvidia --bench both` exits with code 0 and prints the benchmark summary totals.
  - Negative Tests (expected to FAIL):
    - Benchmark run fails due to runtime errors, hangs, or produces invalid timing outputs (e.g., missing device timing when CUDA is active).

- AC-5: No modifications to the official test suite
  - Positive Tests (expected to PASS):
    - `git diff -- InfiniCore/test/infinicore` is empty (no local changes).
  - Negative Tests (expected to FAIL):
    - Any file under `InfiniCore/test/infinicore/` is changed in a way that bypasses correctness or disables coverage.

- AC-6: Remote submission is ready and push succeeds
  - Positive Tests (expected to PASS):
    - Local changes are committed and `git push origin HEAD:2025-autumn-LaiQuan-conquer-T1-1-37` succeeds (or equivalent push command per local git remote configuration).
  - Negative Tests (expected to FAIL):
    - Push rejected due to permissions, wrong branch, or non-fast-forward history.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
A fully correct and performance-tuned CUDA/NVIDIA implementation for all five operators, including:

- Robust handling of edge cases and unusual shapes/strides that appear in the official test suite.
- Careful CUDA memory safety (bounds checks, correct indexing math, no race conditions).
- Sensible kernel launch configuration and use of shared memory or vectorization where appropriate.
- Benchmark runs complete successfully and show non-regressing performance vs the initial baseline run.

### Lower Bound (Minimum Acceptable Scope)
The smallest acceptable change set that still satisfies the acceptance criteria:

- Fixes compilation errors for the NVIDIA backend.
- Produces correct outputs within the framework’s tolerances for the official test cases.
- Avoids crashes/illegal memory accesses.
- Leaves optimization opportunities for later, as long as correctness and stability are met.

### Allowed Choices
- Can use:
  - Standard CUDA C/C++ and the existing InfiniCore operator/kernel patterns in `InfiniCore/src/infiniop/ops/**`.
  - Existing framework helpers/macros/utilities already used by other ops (e.g., reduction helpers, tensor access helpers, workspace APIs).
  - Local profiling/debugging tools (`cuda-memcheck`, `nsys`, `nvidia-smi`) for investigation.
- Cannot use:
  - Changes to checked-in test files under `InfiniCore/test/infinicore/` to "make tests pass" by bypassing assertions or reducing coverage.
  - Closed-source or externally downloaded acceleration libraries not already vendored in `InfiniCore/third_party/`.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
1. **Establish a baseline**:
   - Build with `--nv-gpu=y`, run the targeted ops on NVIDIA, and capture the first failing operator and stack trace.
2. **Fix compilation first, then runtime safety**:
   - Prioritize build errors and linker issues.
   - Then address CUDA memory safety (bounds checks, correct pointer math, correct grid/block mapping).
3. **Operator-by-operator correctness**:
   - `diff`: validate axis/stride handling, boundary conditions, and output shape math.
   - `digamma`: ensure numerically stable approximations and handle small/negative inputs per the expected semantics in tests.
   - `dist`: confirm p-norm definition, broadcasting/shape rules, and reduction correctness (avoid race conditions).
   - `logdet`: validate decomposition approach, workspace sizing, and numerical stability (avoid overflow/underflow when possible).
   - `pad`: confirm index mapping from output → input and implement the required modes (`constant`, `reflect`, `replicate`, `circular`) consistently with PyTorch.
4. **Benchmark last, after correctness**:
   - Treat benchmark numbers as informational unless the evaluation defines explicit performance thresholds.

### Relevant References
- `InfiniCore/xmake.lua` - build configuration options (including `nv-gpu`).
- `InfiniCore/scripts/install.py` - canonical build/install entrypoint used by the draft.
- `InfiniCore/test/infinicore/run.py` - official local runner (`--nvidia`, `--bench`, `--ops`, `--load`).
- Operator implementations (likely edit targets):
  - `InfiniCore/src/infiniop/ops/diff/`
  - `InfiniCore/src/infiniop/ops/digamma/`
  - `InfiniCore/src/infiniop/ops/dist/`
  - `InfiniCore/src/infiniop/ops/logdet/`
  - `InfiniCore/src/infiniop/ops/pad/`

## Dependencies and Sequence

### Milestones
1. Baseline build + failure reproduction
   - Phase A: Build with `python scripts/install.py --omp=y --cpu=y --nv-gpu=y` and record the first error.
   - Phase B: Run `python test/infinicore/run.py --ops diff digamma dist logdet pad --nvidia --verbose` and record the first failing operator and failure mode.
2. Compilation fixes (blocking)
   - Phase A: Resolve compilation/type issues in the targeted operator CUDA/NVIDIA sources.
   - Phase B: Confirm the full build is clean before debugging runtime behavior.
3. Correctness fixes (per operator)
   - Phase A: Fix one operator at a time, re-running only that operator in the test runner for fast iteration.
   - Phase B: After each operator passes, re-run the full targeted set to catch cross-op regressions.
4. Benchmark and polish
   - Phase A: Run benchmark mode to ensure it is stable and produces timing summaries.
   - Phase B: Optional tuning where it is low-risk (e.g., launch configuration), without sacrificing correctness.
5. Final validation and submission
   - Phase A: Ensure `git diff -- InfiniCore/test/infinicore` is empty (test suite unchanged).
   - Phase B: Commit and push to `2025-autumn-LaiQuan-conquer-T1-1-37`.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are strictly for plan documentation only.
- Use descriptive, mathematical, and domain-appropriate naming conventions within the actual C++/CUDA codebase.

--- Original Design Draft Start ---

# Operator Development Plan (diff, digamma, dist, logdet, pad)

## Goal Description
Fix, optimize, and successfully execute the 5 currently broken operators (`diff`, `digamma`, `dist`, `logdet`, `pad`) on a local NVIDIA RTX 5060Ti GPU. The objective is to ensure the codebase compiles properly, passes all official benchmark tests without modifying any built-in test cases, and to push the final working modifications to the target remote repository and branch (`2025-autumn-LaiQuan-conquer-T1-1-37`).

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Successful Library and Operator Compilation
  - Positive Tests (expected to PASS):
    - Executing `XMAKE_ROOT=y python scripts/install.py --omp=y --cpu=y --nv-gpu=y` completes successfully with no syntax errors, undefined references, or fatal aborts in the terminal.
  - Negative Tests (expected to FAIL):
    - Compilation halts due to C++/CUDA syntax errors, missing headers, or type mismatches in any of the 5 targeted operator files.
- AC-2: Official Benchmark Tests Execution
  - Positive Tests:
    - Executing `python test/infinicore/run.py --ops diff,digamma,dist,logdet,pad --nv-gpu --bench` runs successfully, printing "PASS" and the benchmark performance metrics for all 5 operators.
  - Negative Tests:
    - The test script crashes due to runtime errors (e.g., CUDA out-of-bounds memory access, segmentation fault, illegal memory access) or fails the official assertions due to incorrect mathematical logic.
- AC-3: Strict Preservation of Official Test Cases
  - Positive Tests:
    - Git status and diff show zero modifications, deletions, or bypasses to the official test cases located in the `test/infinicore/` directory.
  - Negative Tests:
    - Built-in test cases or the official test scripts are found to be modified to achieve a false positive pass.
- AC-4: Code Submission and Remote Push
  - Positive Tests:
    - Successfully committing and running `git push` to upload all local changes to the `2025-autumn-LaiQuan-conquer-T1-1-37` branch of the `git@github.com:LaiQuan-conquer/InfiniCore.git` repository.
  - Negative Tests:
    - Push gets rejected by the remote server due to incorrect branch naming, missing permissions, or non-fast-forward tracking errors.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
A highly optimized CUDA implementation for all five operators that fully utilizes the shared memory and parallel computing capabilities of the local RTX 5060Ti. The code gracefully handles complex index calculations and memory boundaries (especially for `pad` and `diff`), achieves optimal computational performance in the benchmark tests, and features clean formatting with proper grid/block dimension tuning.

### Lower Bound (Minimum Acceptable Scope)
A fundamentally sound algorithmic implementation that resolves all existing syntax and compilation bugs, correctly computes the required mathematical outputs, and successfully passes the target test commands on the local GPU, satisfying the minimum requirements for the competition without over-engineering.

### Allowed Choices
- Can use: Standard CUDA C/C++ programming paradigms, existing mathematical helper functions/macros within the InfiniCore framework, and local profiling/debugging commands (e.g., `nvidia-smi`).
- Cannot use: Any modifications to the official test scripts (including `run.py` and its dependencies), alterations to the built-in test cases, or unauthorized closed-source third-party acceleration libraries.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
1. **Compilation Troubleshooting**: Address the immediate "cannot compile" issue by inspecting the terminal logs from `install.py`. Fix fundamental C++ issues such as missing header includes, uninitialized pointers, or kernel parameter mismatches.
2. **Operator-by-Operator Execution**:
   - `diff`: Ensure correct stride and boundary checks when computing differences along specified dimensions.
   - `digamma`: Implement or correctly call stable numerical approximations for the logarithmic derivative of the gamma function to avoid NaN results.
   - `dist`: Focus on accurate norm calculations (e.g., p-norm) across vectors/matrices and ensure correct reduction implementation to prevent race conditions.
   - `logdet`: This may require a stable approach for determinant calculation (such as leveraging LU or Cholesky decomposition equivalents available in the framework or robust custom kernels) to prevent underflow/overflow.
   - `pad`: Pay close attention to index mapping between the padded output tensor and the original input tensor, handling various padding modes (e.g., constant, reflect, replicate).
3. **Iterative Testing**: Isolate the operators using the provided test script (e.g., test individually via `--ops pad`). Debug logic errors sequentially before proceeding to the combined full benchmark validation.

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
   - Phase B: Commit the finalized code and push it to the designated Git repository and `2025-autumn-LaiQuan-conquer-T1-1-37` branch.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are strictly for plan documentation only.
- Use descriptive, mathematical, and domain-appropriate naming conventions within the actual C++/CUDA codebase.

--- Original Design Draft End ---
