from dataclasses import dataclass, field
from typing import Any


@dataclass
class CaseResult:
    """Test case result data structure"""

    success: bool
    return_code: int  # 0: success, -1: failure, -2: skipped, -3: partial
    torch_host_time: float = 0.0
    torch_device_time: float = 0.0
    infini_host_time: float = 0.0
    infini_device_time: float = 0.0
    error_message: str = ""
    test_case: Any = None
    device: Any = None


@dataclass
class TestTiming:
    """Stores performance timing metrics."""

    torch_host: float = 0.0
    torch_device: float = 0.0
    infini_host: float = 0.0
    infini_device: float = 0.0
    # Added field to support the logic in your print_summary
    operators_tested: int = 0


@dataclass
class OperatorResult:
    """Stores the execution results of a single operator."""

    name: str
    success: bool = False
    return_code: int = -1
    error_message: str = ""
    stdout: str = ""
    stderr: str = ""
    timing: TestTiming = field(default_factory=TestTiming)

    @property
    def status_icon(self):
        if self.return_code == 0:
            return "✅"
        if self.return_code == -2:
            return "⏭️"
        if self.return_code == -3:
            return "⚠️"
        return "❌"

    @property
    def status_text(self):
        if self.return_code == 0:
            return "PASSED"
        if self.return_code == -2:
            return "SKIPPED"
        if self.return_code == -3:
            return "PARTIAL"
        return "FAILED"
