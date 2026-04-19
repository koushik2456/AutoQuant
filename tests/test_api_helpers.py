import os
import tempfile
import unittest

from autoquant.api_helpers import sanitize_output_dir, validate_model_name


class TestValidateModelName(unittest.TestCase):
    def test_simple_and_org_slash(self) -> None:
        for name in ("gpt2", "facebook/opt-125m", "EleutherAI/gpt-neo-125M"):
            ok, msg = validate_model_name(name)
            self.assertTrue(ok, msg)

    def test_rejects(self) -> None:
        for name in ("bad/name/extra", "../escape", "", "x" * 201):
            ok, _ = validate_model_name(name)
            self.assertFalse(ok)


class TestSanitizeOutputDir(unittest.TestCase):
    def test_under_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ok, msg, resolved = sanitize_output_dir("my_run", "default_x", tmp)
            self.assertTrue(ok)
            self.assertEqual(msg, "my_run")
            self.assertEqual(resolved, os.path.join(os.path.realpath(tmp), "my_run"))

    def test_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ok, _, _ = sanitize_output_dir("../outside", "d", tmp)
            self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
