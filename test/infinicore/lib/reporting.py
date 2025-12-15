from .types import TestResult

class ConsoleReporter:
    def print_header(self, ops_dir, count):
        print(f"InfiniCore Operator Test Runner")
        print(f"Directory: {ops_dir}")
        print(f"Tests found: {count}\n")

    def print_live_result(self, result: TestResult, verbose=False):
        """实时打印单行结果"""
        print(f"{result.status_icon}  {result.name}: {result.status_text} (code: {result.return_code})")
        
        if verbose or not result.success:
            if result.stdout: print(result.stdout.rstrip())
            if result.stderr: print("\nSTDERR:", result.stderr.rstrip())
            if result.error_message: print(f"💥 Error: {result.error_message}")

    def print_summary(self, results, cumulative_timing, total_expected=0):
        print(f"\n{'='*80}\nCUMULATIVE TEST SUMMARY\n{'='*80}")
        
        passed = [r for r in results if r.return_code == 0]
        failed = [r for r in results if r.return_code == -1]
        skipped = [r for r in results if r.return_code == -2]
        
        print(f"Total: {len(results)} | Passed: {len(passed)} | Failed: {len(failed)}")
        
        # 打印 Benchmark 数据
        if cumulative_timing:
            self._print_timing(cumulative_timing)

        # 打印失败列表
        if failed:
            print(f"\n❌ FAILED ({len(failed)}):")
            for r in failed: print(f"  {r.name}")

        return len(failed) == 0

    def _print_timing(self, t):
        print(f"{'-'*40}\nBENCHMARK SUMMARY:")
        print(f"  PyTorch Host:    {t.torch_host:.3f} ms")
        print(f"  InfiniCore Host: {t.infini_host:.3f} ms")
        print(f"{'-'*40}")
