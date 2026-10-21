"""Offline tests for the PR data-format/binding version gate."""

import unittest
import tempfile
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts import check_format_version_bumps
from scripts.check_format_version_bumps import BINDINGS, RELEASE_SOURCE, check_bumps, supported_formats


def tree(yamc="0.16.2", yani="0.21.2", formats="2..=2"):
    return {
        BINDINGS["yamc-core"]: f'[package]\nname = "yamc-python"\nversion = "{yamc}"\n',
        BINDINGS["yani-core"]: f'[package]\nname = "yani-python"\nversion = "{yani}"\n',
        RELEASE_SOURCE: f"pub const SUPPORTED_FORMAT_VERSIONS: RangeInclusive<u32> = {formats};",
    }


class FormatVersionBumps(unittest.TestCase):
    def test_unrelated_changes_need_no_bump(self):
        self.assertEqual(check_bumps({"README.md"}, {}, {}), [])

    def test_release_comment_changes_need_no_bump(self):
        before = tree()
        after = tree()
        after[RELEASE_SOURCE] += "\n// Clarify the release layout.\n"
        self.assertEqual(check_bumps({RELEASE_SOURCE}, before, after), [])

    def test_format_change_requires_both_versions(self):
        failures = check_bumps({RELEASE_SOURCE}, tree(), tree(formats="2..=3"))
        self.assertEqual(len(failures), 2)
        self.assertTrue(any("yamc-core" in failure for failure in failures))
        self.assertTrue(any("yani-core" in failure for failure in failures))

    def test_only_one_version_bumped_is_not_enough(self):
        failures = check_bumps({RELEASE_SOURCE}, tree(), tree(yamc="0.16.3", formats="2..=3"))
        self.assertEqual(len(failures), 1)
        self.assertIn("yani-core", failures[0])

    def test_both_versions_bumped_pass(self):
        self.assertEqual(
            check_bumps({RELEASE_SOURCE}, tree(), tree("0.16.3", "0.21.3", "2..=3")), []
        )

    def test_schema_change_requires_both_versions(self):
        failures = check_bumps({"crates/nuclear-data-schema/src/lib.rs"}, tree(), tree())
        self.assertEqual(len(failures), 2)

    def test_schema_change_with_bumps_passes(self):
        self.assertEqual(
            check_bumps(
                {"crates/nuclear-data-schema/src/reactions.rs"}, tree(), tree("0.16.3", "0.21.3")
            ),
            [],
        )

    def test_schema_readme_needs_no_bump(self):
        self.assertEqual(check_bumps({"crates/nuclear-data-schema/README.md"}, {}, {}), [])

    def test_multiline_format_range_is_read(self):
        self.assertEqual(
            supported_formats("pub const SUPPORTED_FORMAT_VERSIONS: RangeInclusive<u32> =\n 2 ..= 3;"),
            (2, 3),
        )

    def test_unrecognised_format_definition_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "update the format-bump check"):
            supported_formats("pub const SUPPORTED_FORMAT_VERSIONS: &[u32] = &[2, 3];")

    def test_real_git_base_and_head_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(check_format_version_bumps, "ROOT", root):
                check_format_version_bumps.git("init", "--quiet")
                for path, content in tree().items():
                    destination = root / path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(content)
                check_format_version_bumps.git("add", ".")
                check_format_version_bumps.git(
                    "-c", "user.name=shimwell", "-c", "user.email=mail@jshimwell.com",
                    "commit", "--quiet", "-m", "Original format",
                )
                base = check_format_version_bumps.git("rev-parse", "HEAD").strip()
                (root / RELEASE_SOURCE).write_text(tree(formats="2..=3")[RELEASE_SOURCE])
                check_format_version_bumps.git("add", ".")
                check_format_version_bumps.git(
                    "-c", "user.name=shimwell", "-c", "user.email=mail@jshimwell.com",
                    "commit", "--quiet", "-m", "New format without version bumps",
                )
                errors = StringIO()
                with patch("sys.argv", ["check_format_version_bumps.py", "--base", base]):
                    with redirect_stderr(errors):
                        self.assertEqual(check_format_version_bumps.main(), 1)
                self.assertIn("yamc-core", errors.getvalue())
                self.assertIn("yani-core", errors.getvalue())


if __name__ == "__main__":
    unittest.main()