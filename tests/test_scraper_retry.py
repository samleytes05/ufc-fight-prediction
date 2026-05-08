from __future__ import annotations

from http.client import IncompleteRead
from unittest import TestCase, mock

from src.scrapers.common import fetch_html


class FetchHtmlRetryTests(TestCase):
    def test_fetch_html_retries_incomplete_read_and_succeeds(self) -> None:
        success_response = mock.MagicMock()
        success_response.read.return_value = b"<html>ok</html>"
        success_response.__enter__.return_value = success_response
        success_response.__exit__.return_value = False

        with (
            mock.patch(
                "src.scrapers.common.urlopen",
                side_effect=[IncompleteRead(b"partial", 10), success_response],
            ) as mocked_urlopen,
            mock.patch("src.scrapers.common.time.sleep") as mocked_sleep,
        ):
            html = fetch_html("http://example.com", retries=3, backoff_seconds=0.01)

        self.assertEqual(html, "<html>ok</html>")
        self.assertEqual(mocked_urlopen.call_count, 2)
        mocked_sleep.assert_called_once()

    def test_fetch_html_does_not_swallow_non_network_errors(self) -> None:
        with mock.patch("src.scrapers.common.urlopen", side_effect=ValueError("parser-like failure")):
            with self.assertRaises(ValueError):
                fetch_html("http://example.com", retries=3, backoff_seconds=0.01)
