"""Regression checks for a stale server training CLI; never run real training."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_bias_only_remaining as launcher


class LauncherCompatibilityTests(unittest.TestCase):
    def write_fake_cli(self, root, options):
        source = (
            "import argparse, sys\n"
            "from pathlib import Path\n"
            "if '--help' not in sys.argv:\n"
            "    Path(__file__).with_name('training_started').touch()\n"
            "p = argparse.ArgumentParser()\n"
            + "".join(f"p.add_argument({option!r}, action='store_true')\n" for option in options)
            + "p.parse_args()\n"
        )
        (root / "cancha_cub.py").write_text(source, encoding="utf-8")

    def test_compatible_entry_point_only_reads_help(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fake_cli(root, ["--bias_only", "--bias_only_out"])
            with patch.object(launcher, "ROOT", root):
                launcher.check_training_cli([["--bias_only", "--bias_only_out", "unused"]])
            self.assertFalse((root / "training_started").exists())

    def test_similar_option_does_not_hide_missing_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fake_cli(root, ["--bias_only_out"])
            with patch.object(launcher, "ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "does not support: --bias_only\\."):
                    launcher.check_training_cli([["--bias_only", "--bias_only_out", "unused"]])

    def test_stale_entry_point_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = launcher.build_command("BUS_UCLM", "decoded", "cpu", root / "result")
            options = [arg for arg in command if arg.startswith("--")
                       and arg not in ("--bias_only", "--bias_only_out")]
            self.write_fake_cli(root, options)
            errors = io.StringIO()
            with patch.object(launcher, "ROOT", root), \
                    patch.object(launcher, "check_inputs"), \
                    patch("sys.argv", ["launcher", "--datasets", "BUS_UCLM",
                                       "--output-root", "result"]), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(errors):
                with self.assertRaises(SystemExit) as stopped:
                    launcher.main()
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn("--bias_only, --bias_only_out", errors.getvalue())
            self.assertFalse((root / "result").exists())
            self.assertFalse((root / "training_started").exists())


if __name__ == "__main__":
    unittest.main()
