from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TestTiming:
    """存储性能测试时间"""
    torch_host: float = 0.0
    torch_device: float = 0.0
    infini_host: float = 0.0
    infini_device: float = 0.0

@dataclass
class TestResult:
    """存储单个测试文件的运行结果"""
    name: str
    success: bool = False
    return_code: int = -1
    error_message: str = ""
    stdout: str = ""
    stderr: str = ""
    timing: TestTiming = field(default_factory=TestTiming)

    @property
    def status_icon(self):
        if self.return_code == 0: return "✅"
        if self.return_code == -2: return "⏭️"  # Skipped
        if self.return_code == -3: return "⚠️"  # Partial
        return "❌"

    @property
    def status_text(self):
        if self.return_code == 0: return "PASSED"
        if self.return_code == -2: return "SKIPPED"
        if self.return_code == -3: return "PARTIAL"
        return "FAILED"
