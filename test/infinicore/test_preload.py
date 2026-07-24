import ctypes
import os
import unittest
from unittest import mock

from infinicore import _preload


class DevicePreloadTest(unittest.TestCase):
    def test_hpcc_environment_selects_only_mars(self):
        with mock.patch.dict(os.environ, {"HPCC_PATH": "/opt/hpcc"}, clear=True):
            self.assertTrue(_preload._should_preload_device("MARS"))
            self.assertFalse(_preload._should_preload_device("METAX"))

    def test_maca_environment_does_not_select_mars(self):
        with mock.patch.dict(os.environ, {"MACA_PATH": "/opt/maca"}, clear=True):
            self.assertFalse(_preload._should_preload_device("MARS"))
            self.assertFalse(_preload._should_preload_device("METAX"))

    def test_hpcc_home_is_a_supported_runtime_prefix(self):
        with mock.patch.dict(os.environ, {"HPCC_HOME": "/custom/hpcc"}, clear=True):
            with mock.patch.object(_preload, "_try_load") as try_load:
                _preload.preload_hpcc()

        self.assertEqual(try_load.call_count, 4)
        self.assertEqual(try_load.call_args_list[0].args[0], ["/custom/hpcc"])

    def test_runtime_loader_checks_lib64(self):
        def exists(path):
            return path == "/custom/hpcc/lib64/libhcruntime.so"

        with mock.patch.object(_preload.os.path, "exists", side_effect=exists):
            with mock.patch.object(_preload.ctypes, "CDLL") as cdll:
                self.assertTrue(
                    _preload._try_load(["/custom/hpcc"], "libhcruntime.so")
                )

        cdll.assert_called_once_with(
            "/custom/hpcc/lib64/libhcruntime.so", mode=ctypes.RTLD_GLOBAL
        )


if __name__ == "__main__":
    unittest.main()
