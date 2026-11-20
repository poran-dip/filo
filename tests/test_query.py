import unittest

from server.query import InvalidDriveQuery, validate_query


class QueryTests(unittest.TestCase):
    def test_accepts_supported_filters(self):
        for query in (
            "(name contains 'report' or fullText contains 'report') and mimeType != 'image/png'",
            "not (createdTime <= '2024-01-01' and modifiedTime > '2023-01-01')",
            r"name = 'sam\'s (trashed) files\\draft'",
        ):
            with self.subTest(query=query):
                validate_query(query)

    def test_rejects_invalid_filters(self):
        for query in (
            "x) or (name contains 'y",
            "name = 'x') or (name = 'y'",
            "name = 'x' or trashed = false",
            "'id' in parents",
            "mimeType has 'pdf'",
            "name = 'unterminated",
            "(name = 'x'",
            "name = 'x' and",
            "",
        ):
            with self.subTest(query=query), self.assertRaises(InvalidDriveQuery):
                validate_query(query)

    def test_reports_invalid_input_and_expected_syntax(self):
        for query, offending, expected in (
            ("not", "'not'", "predicate"),
            ("not not", "'not'", "field"),
            ("not not name = 'x'", "'not'", "field"),
            ("name = 'x' AND name = 'y'", "'AND'", "'and'"),
            ("name = 'x' OR name = 'y'", "'OR'", "'or'"),
            ("mimetype = 'text/plain'", "'mimetype'", "mimeType"),
            ("fullText = 'x'", "'='", "contains"),
            ("name = word", "'word'", "single-quoted"),
            (None, "NoneType", "string"),
        ):
            with self.subTest(query=query), self.assertRaises(
                InvalidDriveQuery
            ) as error:
                validate_query(query)
            self.assertIn(offending, str(error.exception))
            self.assertIn(expected, str(error.exception))
