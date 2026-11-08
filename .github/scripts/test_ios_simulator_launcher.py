# Copyright 2025-present the zvec project
# SPDX-License-Identifier: Apache-2.0
"""Exercise the observed simctl failure with real temporary console FIFOs."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ios_simulator_launcher as launcher


class SimulatorLauncherTest(unittest.TestCase):
    device = "4A57FB15-6E7C-49CC-BE93-450B18BD286E"
    bundle = "com.zvec.visit_filter_test"
    pid = 53118

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        home_patch = mock.patch.dict(os.environ, HOME=str(self.home))
        home_patch.start()
        self.addCleanup(home_patch.stop)
        self.directory = (
            self.home
            / "Library/Developer/CoreSimulator/Devices"
            / self.device
            / "data/tmp"
        )
        self.directory.mkdir(parents=True)
        self.fifo = self.directory / f"launch_console-{self.pid}-1"
        self.log = self.home / "visit_filter_test.log"

    def collision(self, path=None):
        return (
            f"Unable to establish FIFO at {path or self.fifo}: Error 17\n"
            "An error was encountered processing the command "
            "(domain=NSPOSIXErrorDomain, code=17):\n"
            "The operation couldn\u2019t be completed. File exists\nFile exists\n"
        )

    def run_test(self, runner, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()) as console:
            result = launcher.run_test(
                self.device, self.bundle, self.log, runner=runner, **kwargs
            )
        assert console.getvalue() == self.log.read_text()
        return result

    def test_exact_failure_removes_only_its_fifo_and_requires_new_success(self):
        os.mkfifo(self.fifo)
        other = self.directory / "launch_console-123-1"
        os.mkfifo(other)
        passed = "[iOS] Starting visit_filter_test\n[  PASSED  ] 1 test.\n"
        runner = mock.Mock(
            side_effect=[(self.pid, 17, self.collision()), (53119, 0, passed)]
        )
        with mock.patch.object(launcher, "monotonic", side_effect=[100, 101, 102]):
            assert self.run_test(runner, timeout=600) == 0
        assert not self.fifo.exists()
        assert other.exists()
        assert runner.call_count == 2
        for call, remaining in zip(runner.call_args_list, [599, 598], strict=True):
            command, env, timeout = call.args
            assert command == [
                "xcrun",
                "simctl",
                "launch",
                "--console",
                self.device,
                self.bundle,
            ]
            assert env["SIMCTL_CHILD_GTEST_COLOR"] == "no"
            assert timeout == remaining
        output = self.log.read_text()
        assert self.collision() in output
        assert "::warning::" in output
        assert output.endswith(passed)

    def test_success_launches_once(self):
        runner = mock.Mock(return_value=(self.pid, 0, "[  PASSED  ] 1 test.\n"))
        assert self.run_test(runner) == 0
        runner.assert_called_once()

    def test_does_not_retry_actual_failure_crash_or_timeout(self):
        os.mkfifo(self.fifo)
        for status in [1, 3, 124, 139, -11]:
            with self.subTest(status=status):
                runner = mock.Mock(return_value=(self.pid, status, self.collision()))
                assert self.run_test(runner) == status
                runner.assert_called_once()
                assert self.fifo.exists()

    def test_does_not_retry_if_test_has_started(self):
        os.mkfifo(self.fifo)
        for output in [
            "[iOS] Starting visit_filter_test\n",
            "Running main() from gtest_main.cc\n",
            "[==========] Running 1 test from 1 test suite.\n",
            "[ RUN      ] VisitFilterTest.Check\n",
            "[       OK ] VisitFilterTest.Check (0 ms)\n",
            "[  PASSED  ] 1 test.\n",
            "[  FAILED  ] VisitFilterTest.Check\n",
            "Passed: 1\n",
            "Failed: 1\n",
            "C API memory admission and recovery passed\n",
        ]:
            with self.subTest(output=output):
                runner = mock.Mock(
                    return_value=(self.pid, 17, output + self.collision())
                )
                assert self.run_test(runner) == 17
                runner.assert_called_once()
                assert self.fifo.exists()

    def test_requires_both_exact_fifo_error_and_posix_domain(self):
        os.mkfifo(self.fifo)
        for output in [
            "File exists\n",
            self.collision().replace("Error 17", "Error 13"),
            self.collision().replace("NSPOSIXErrorDomain", "OtherErrorDomain"),
            self.collision().replace("code=17", "code=13"),
        ]:
            with self.subTest(output=output):
                assert not launcher.remove_stale_console_fifo(
                    self.device, self.pid, 17, output
                )
                assert self.fifo.exists()

    def test_does_not_remove_regular_file(self):
        self.fifo.write_text("preserve")
        assert not launcher.remove_stale_console_fifo(
            self.device, self.pid, 17, self.collision()
        )
        assert self.fifo.read_text() == "preserve"

    def test_does_not_remove_symlink(self):
        target = self.directory / "other-fifo"
        os.mkfifo(target)
        self.fifo.symlink_to(target)
        assert not launcher.remove_stale_console_fifo(
            self.device, self.pid, 17, self.collision()
        )
        assert self.fifo.is_symlink()
        assert target.exists()

    def test_requires_current_process_pid_and_requested_device(self):
        os.mkfifo(self.fifo)
        for device, pid in [
            (self.device, self.pid + 1),
            ("00000000-0000-0000-0000-000000000000", self.pid),
            ("../../../../../../tmp", self.pid),
            (self.device, 0),
        ]:
            with self.subTest(device=device, pid=pid):
                assert not launcher.remove_stale_console_fifo(
                    device, pid, 17, self.collision()
                )
                assert self.fifo.exists()

    def test_does_not_remove_fifo_outside_device_directory(self):
        outside = self.home / self.fifo.name
        os.mkfifo(outside)
        assert not launcher.remove_stale_console_fifo(
            self.device, self.pid, 17, self.collision(outside)
        )
        assert outside.exists()

    def test_missing_fifo_is_not_retried(self):
        runner = mock.Mock(return_value=(self.pid, 17, self.collision()))
        assert self.run_test(runner) == 17
        runner.assert_called_once()

    def test_requires_fifo_owned_by_current_user(self):
        os.mkfifo(self.fifo)
        owner = os.getuid()
        with mock.patch.object(launcher.os, "getuid", return_value=owner + 1):
            assert not launcher.remove_stale_console_fifo(
                self.device, self.pid, 17, self.collision()
            )
        assert self.fifo.exists()

    def test_unlink_error_is_not_retried(self):
        os.mkfifo(self.fifo)
        runner = mock.Mock(return_value=(self.pid, 17, self.collision()))
        with mock.patch.object(Path, "unlink", side_effect=PermissionError):
            assert self.run_test(runner) == 17
        runner.assert_called_once()
        assert self.fifo.exists()

    def test_second_collision_is_failure_without_further_cleanup_or_retry(self):
        os.mkfifo(self.fifo)
        second = self.directory / "launch_console-53119-1"
        os.mkfifo(second)
        runner = mock.Mock(
            side_effect=[
                (self.pid, 17, self.collision()),
                (53119, 17, self.collision(second)),
            ]
        )
        assert self.run_test(runner) == 17
        assert runner.call_count == 2
        assert not self.fifo.exists()
        assert second.exists()

    def test_recovered_launch_still_propagates_test_failure_crash_and_timeout(self):
        for status, output in [
            (1, "[  FAILED  ] VisitFilterTest.Check\n"),
            (139, "[ RUN      ] VisitFilterTest.Check\nSegmentation fault\n"),
            (124, "[ RUN      ] VisitFilterTest.Check\nTest timed out.\n"),
        ]:
            with self.subTest(status=status):
                os.mkfifo(self.fifo)
                runner = mock.Mock(
                    side_effect=[
                        (self.pid, 17, self.collision()),
                        (53119, status, output),
                    ]
                )
                assert self.run_test(runner) == status
                assert runner.call_count == 2
                assert self.log.read_text().endswith(output)

    def test_recovery_does_not_extend_total_timeout(self):
        os.mkfifo(self.fifo)
        runner = mock.Mock(return_value=(self.pid, 17, self.collision()))
        with mock.patch.object(launcher, "monotonic", side_effect=[100, 100, 110]):
            assert self.run_test(runner, timeout=10) == 124
        runner.assert_called_once()
        assert "Test timed out." in self.log.read_text()

    def test_real_child_exit_status_and_combined_output_are_preserved(self):
        pid, status, output = launcher.launch(
            [
                sys.executable,
                "-c",
                "import sys; print('started', flush=True); "
                "print('assertion', file=sys.stderr); sys.exit(7)",
            ],
            os.environ.copy(),
            10,
        )
        assert pid > 0
        assert status == 7
        assert output == "started\nassertion\n"

    def test_real_child_timeout_preserves_partial_output(self):
        _, status, output = launcher.launch(
            [
                sys.executable,
                "-c",
                "import time; print('started', flush=True); time.sleep(60)",
            ],
            os.environ.copy(),
            0.5,
        )
        assert status == 124
        assert "started\n" in output
        assert "Test timed out." in output


if __name__ == "__main__":
    unittest.main()
