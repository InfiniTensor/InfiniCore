# Operator Development Plan: NVIDIA Fix + Bench (block_diag, hinge_embedding_loss, kron, selu, sinh)

## Goal Description
Repair the NVIDIA (CUDA) implementations of the five currently broken operators (`block_diag`, `hinge_embedding_loss`, `kron`, `selu`, `sinh`) in the `InfiniCore/` repository so that:

- The project builds and installs successfully via the official `scripts/install.py` flow
- The C++ and Python layers are installed (`xmake install _infinicore` and `pip install -e .`) so the test harness can import and execute operators
- The official operator test/benchmark harness passes for these operators on an NVIDIA GPU (target environment: local RTX 5060 Ti)
- No official tests or benchmark scripts are modified to achieve passing results
- The final changes are committed and pushed to `origin/2025-autumn-LaiQuan-conquer-T1-1-31`

All commands in this plan assume the working directory is `InfiniCore/` unless noted otherwise.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Build + install completes (runtime env + xmake + Python package)
  - Positive Tests (expected to PASS):
    - Running `source scripts/set_env_linux.sh` completes (or equivalent manual export of `INFINI_ROOT` + `LD_LIBRARY_PATH` as documented in `README.md`).
    - Running `python scripts/install.py --omp=y --cpu=y --nv-gpu=y` completes with exit code 0 (no compile/link errors).
    - Running `xmake build _infinicore && xmake install _infinicore` completes with exit code 0.
    - Running `pip install -e .` completes with exit code 0.
    - Running `python -c "import infinicore; print('import_ok')"` prints `import_ok`.
  - Negative Tests (expected to FAIL):
    - Any of the commands above fails, or `import infinicore` fails due to missing modules/shared libraries.
- AC-2: Targeted operator suite passes on NVIDIA with benchmarking enabled
  - Positive Tests (expected to PASS):
    - Running `python test/infinicore/run.py --ops block_diag hinge_embedding_loss kron selu sinh --nvidia --bench` exits with code 0.
    - The final summary reports `🎉 All tests passed!` and the five operators appear under `✅ PASSED OPERATORS` (and not under `❌ FAILED OPERATORS`).
  - Negative Tests (expected to FAIL):
    - The command exits non-zero, any of the five operators appear under `❌ FAILED OPERATORS`, or a CUDA runtime error occurs (e.g., illegal memory access).
- AC-3: Official tests and benchmark harness remain unmodified
  - Positive Tests (expected to PASS):
    - Running `git diff --name-only -- test/` returns no output.
    - Running `git status --porcelain` shows no modified files under `test/` (including `test/infinicore/`).
  - Negative Tests (expected to FAIL):
    - Any file under `test/` is modified, deleted, or bypassed to obtain a pass.
- AC-4: Each operator passes its single-op test entrypoint on NVIDIA (debugging gate)
  - Positive Tests (expected to PASS):
    - Running each command below exits with code 0:
      - `python test/infinicore/ops/block_diag.py --nvidia --bench`
      - `python test/infinicore/ops/hinge_embedding_loss.py --nvidia --bench`
      - `python test/infinicore/ops/kron.py --nvidia --bench`
      - `python test/infinicore/ops/selu.py --nvidia --bench`
      - `python test/infinicore/ops/sinh.py --nvidia --bench`
  - Negative Tests (expected to FAIL):
    - Any individual script fails, or produces mismatched outputs versus the reference implementation (typically PyTorch).
- AC-5: Changes are pushed to the target remote branch
  - Positive Tests (expected to PASS):
    - Running `git rev-parse --abbrev-ref HEAD` returns `2025-autumn-LaiQuan-conquer-T1-1-31`.
    - Running `git push` succeeds and the remote branch updates.
  - Negative Tests (expected to FAIL):
    - Push is rejected (permissions, wrong branch, non-fast-forward), or local changes are not reflected upstream.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
A correct and performance-aware CUDA implementation for all five operators that:
- Passes correctness tests for all covered shapes/dtypes in `test/infinicore/ops/*.py`
- Avoids unnecessary global memory traffic (coalesced reads/writes where feasible)
- Uses existing shared utilities (elementwise/reduction helpers) where appropriate
- Includes sensible kernel launch configuration and avoids obvious bottlenecks
- Maintains clean integration with the existing InfiniOP operator registration and dispatch

### Lower Bound (Minimum Acceptable Scope)
A correctness-first implementation that:
- Builds successfully with `python scripts/install.py --nv-gpu=y`
- Builds and installs the C++ layer via `xmake build _infinicore && xmake install _infinicore`
- Installs the Python package via `pip install -e .`
- Passes the official NVIDIA operator tests for the five targeted operators
- Does not modify tests or weaken assertions
- Keeps changes localized to the operator implementation/registration needed for correctness

### Allowed Choices
- Can use:
  - Existing helper utilities in `src/infiniop/elementwise/`, `src/infiniop/reduce/`, and device-specific utilities under `src/infiniop/devices/`
  - Edits within operator implementation folders such as `src/infiniop/ops/<op>/` and their device subfolders (e.g., `nvidia/`)
  - Small refactors that reduce duplication across these five operators when clearly beneficial and low-risk
- Cannot use:
  - Any modifications to official tests/bench harness under `test/` (including `test/infinicore/run.py` and `test/infinicore/ops/*.py`)
  - Closed-source or non-repo third-party acceleration libraries
  - Changes that alter public operator semantics relative to the reference implementation used in tests

> **Note on Deterministic Designs**: If the draft specifies a highly deterministic design with no choices (e.g., "must use JSON format", "must use algorithm X"), then the path boundaries should reflect this narrow constraint. In such cases, upper and lower bounds may converge to the same point, and "Allowed Choices" should explicitly state that the choice is fixed per the draft specification.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
1. Establish a failing baseline
   - Run `source scripts/set_env_linux.sh` (or manually export the documented env vars).
   - Run `python scripts/install.py --omp=y --cpu=y --nv-gpu=y` and record the first build error.
   - Build/install the C++ and Python layers (`xmake build _infinicore && xmake install _infinicore`, then `pip install -e .`) so tests can import `infinicore`.
   - Run the targeted suite `python test/infinicore/run.py --ops block_diag hinge_embedding_loss kron selu sinh --nvidia --bench` to capture the first runtime/correctness failure once it builds.
2. Fix build/registration issues first
   - Resolve missing includes, mismatched signatures, or dispatch/descriptor mismatches in `include/infiniop/ops/*.h` and `src/infiniop/ops/*/operator.cc` as needed.
3. Iterate operator-by-operator (prefer smallest kernels first)
   - `selu`, `sinh`: elementwise kernels; validate dtype handling and use correct math intrinsics.
   - `hinge_embedding_loss`: ensure correct margin/target behavior and reduction mode consistent with the test harness.
   - `block_diag`, `kron`: focus on correct shape math and index mapping; guard against out-of-bounds and ensure output layout matches reference.
4. Use the test harness to shorten feedback loops
   - Run each op script directly with `--verbose` when debugging: e.g. `python test/infinicore/ops/kron.py --nvidia --verbose`
   - Use `--debug` to get more detailed tensor comparisons when needed.

### Relevant References
- Build/install flow:
  - `README.md` - official setup, build, and test commands (including env vars)
  - `scripts/install.py` - official build and install entrypoint
  - `scripts/set_env_linux.sh` - recommended env var setup (`INFINI_ROOT`, `LD_LIBRARY_PATH`)
  - `xmake.lua` - build options (notably `option("nv-gpu")`)
- NVIDIA operator implementations:
  - `src/infiniop/ops/block_diag/`
  - `src/infiniop/ops/hinge_embedding_loss/`
  - `src/infiniop/ops/kron/`
  - `src/infiniop/ops/selu/`
  - `src/infiniop/ops/sinh/`
- Operator headers / C API surface:
  - `include/infiniop/ops/block_diag.h`
  - `include/infiniop/ops/hinge_embedding_loss.h`
  - `include/infiniop/ops/kron.h`
  - `include/infiniop/ops/selu.h`
  - `include/infiniop/ops/sinh.h`
- Official tests (do not modify):
  - `test/infinicore/run.py`
  - `test/infinicore/ops/` (individual operator test scripts)

## Dependencies and Sequence

### Milestones
1. Milestone 1: Environment and install readiness
   - Phase A: Configure env vars (`source scripts/set_env_linux.sh`)
   - Phase B: Install required layers (`python scripts/install.py …`, `xmake install _infinicore`, `pip install -e .`)
2. Milestone 2: Reproduce and isolate failures
   - Phase A: Capture compile/link errors for the five operators
   - Phase B: Run targeted NVIDIA tests to identify the first failing operator and failure mode
3. Milestone 3: Make the code build and load cleanly
   - Phase A: Fix compilation and linking errors for the five operators
   - Phase B: Ensure Python import and operator registration works end-to-end on NVIDIA
4. Milestone 4: Correctness pass for each operator
   - Phase A: Fix `selu` and `sinh` (elementwise)
   - Phase B: Fix `hinge_embedding_loss` (control flow + reduction)
   - Phase C: Fix `block_diag` and `kron` (index-heavy)
5. Milestone 5: Benchmark + submission readiness
   - Phase A: Run `python test/infinicore/run.py --ops block_diag hinge_embedding_loss kron selu sinh --nvidia --bench` and verify summary is clean
   - Phase B: Verify `git diff -- test/` is empty, then commit and push to the target branch

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead
- Keep changes minimal and localized; prefer aligning with existing InfiniCore/InfiniOP conventions
- Run formatting checks on touched files when practical (see `scripts/format.py`)

--- Original Design Draft Start ---

> Note: The section below is preserved verbatim from `draft.md`. Some command flags/syntax in the draft were normalized in the plan above (e.g., `--nvidia` vs `--nv-gpu`, and `--ops` expects space-separated names).

# Operator Development Plan (block_diag, hinge_embedding_loss, kron, selu, sinh)

## Goal Description
Fix, optimize, and successfully execute the 5 currently broken operators (`block_diag`, `hinge_embedding_loss`, `kron`, `selu`, `sinh`) on a local NVIDIA RTX 5060Ti GPU. The objective is to ensure the codebase compiles properly, passes all official benchmark tests without modifying any built-in test cases, and to push the final working modifications to the target remote repository and branch (`2025-autumn-LaiQuan-conquer-T1-1-31`).

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Successful Library and Operator Compilation
  - Positive Tests (expected to PASS):
    - Executing `XMAKE_ROOT=y python scripts/install.py --omp=y --cpu=y --nv-gpu=y` completes successfully with no syntax errors, undefined references, or fatal aborts in the terminal.
  - Negative Tests (expected to FAIL):
    - Compilation halts due to C++/CUDA syntax errors or type mismatches in any of the 5 targeted operator files.
- AC-2: Official Benchmark Tests Execution
  - Positive Tests:
    - Executing `python test/infinicore/run.py --ops block_diag,hinge_embedding_loss,kron,selu,sinh --nv-gpu --bench` runs successfully, printing "PASS" and the benchmark performance metrics for all 5 operators.
  - Negative Tests:
    - The test script crashes due to runtime errors (e.g., CUDA out-of-bounds memory access, segmentation fault) or fails the official assertions due to incorrect mathematical logic.
- AC-3: Strict Preservation of Official Test Cases
  - Positive Tests:
    - Git status and diff show zero modifications, deletions, or bypasses to the official test cases located in the `test/infinicore/` directory.
  - Negative Tests:
    - Built-in test cases or the official test scripts are found to be modified to achieve a false positive pass.
- AC-4: Code Submission and Remote Push
  - Positive Tests:
    - Successfully committing and running `git push` to upload all local changes to the `2025-autumn-LaiQuan-conquer-T1-1-31` branch of the `git@github.com:LaiQuan-conquer/InfiniCore.git` repository.
  - Negative Tests:
    - Push gets rejected by the remote server due to incorrect branch naming, missing permissions, or non-fast-forward tracking errors.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
A highly optimized CUDA implementation for all five operators that fully utilizes the shared memory and parallel computing capabilities of the local RTX 5060Ti. The code gracefully handles complex index calculations (especially for `block_diag` and `kron`), achieves optimal computational performance in the benchmark tests, and features clean formatting with proper grid/block dimension tuning.

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
   - `block_diag` / `kron`: These require precise multi-dimensional index mapping. Pay close attention to how thread IDs map to tensor coordinates to avoid out-of-bounds memory accesses.
   - `hinge_embedding_loss`: Ensure correct implementation of the conditional logic (margin handling) and proper reduction if a mean/sum is required over the batch.
   - `selu` / `sinh`: These are primarily element-wise operations. Focus on applying the correct mathematical formulas uniformly across the memory layout with efficient vectorized reads/writes if possible.
3. **Iterative Testing**: Isolate the operators using the provided test script (e.g., test individually via `--ops selu`). Debug logic errors sequentially before proceeding to the combined full benchmark validation.

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
   - Phase B: Commit the finalized code and push it to the designated Git repository and `2025-autumn-LaiQuan-conquer-T1-1-31` branch.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are strictly for plan documentation only.
- Use descriptive, mathematical, and domain-appropriate naming conventions within the actual C++/CUDA codebase.

--- Original Design Draft End ---
