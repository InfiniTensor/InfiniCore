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
