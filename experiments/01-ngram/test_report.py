import json
from pathlib import Path
import re
import tempfile
import unittest
from urllib.parse import unquote, urlsplit

from run import write_report


RESULTS = Path(__file__).resolve().parent / "results"


def local_links(text):
    targets = re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", text)
    targets += re.findall(r'(?:href|src)="([^"]+)"', text)
    return [unquote(urlsplit(target).path) for target in targets
            if not urlsplit(target).scheme and not target.startswith("#")]


class ReportPathTests(unittest.TestCase):
    def test_published_report_links_resolve(self):
        for name in ("README.md", "index.html"):
            targets = local_links((RESULTS / name).read_text(encoding="utf-8"))
            self.assertTrue(targets)
            for target in targets:
                with self.subTest(report=name, target=target):
                    self.assertTrue((RESULTS / target).is_file())

    def test_generated_reports_match_published_reports(self):
        result = json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_report(result, output)
            for name in ("README.md", "index.html"):
                with self.subTest(report=name):
                    generated = (output / name).read_text(encoding="utf-8")
                    published = (RESULTS / name).read_text(encoding="utf-8")
                    self.assertEqual(generated.rstrip("\n"), published.rstrip("\n"))


if __name__ == "__main__":
    unittest.main()
