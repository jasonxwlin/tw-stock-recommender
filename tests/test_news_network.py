#!/usr/bin/env python3
"""Self-check for news.py's network-facing fetchers, fmt_report, and main().
No real network calls — requests/yfinance are mocked at the boundary."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest import mock

import pandas as pd

import news as n

GOOGLE_XML = """
<rss><channel><item>
<title>台積電新聞 - 天下雜誌</title>
<link>https://example.com/a</link>
<pubDate>Tue, 01 Sep 2026 12:00:00 GMT</pubDate>
</item></channel></rss>
"""

YAHOO_ITEMS = [
    {'content': {'title': 'TSMC news', 'pubDate': '2026-09-02T04:20:27Z',
                 'provider': {'displayName': 'Reuters'},
                 'canonicalUrl': {'url': 'https://example.com/c'}}},
]


# ── fetch_google_news ─────────────────────────────────────────────────────────

def test_fetch_google_news_returns_recent_items():
    resp = mock.Mock()
    resp.text = GOOGLE_XML
    with mock.patch.object(n.requests, "get", return_value=resp):
        items = n.fetch_google_news("台積電", days=3650)
    assert len(items) == 1
    assert items[0]['source'] == '天下雜誌'


def test_fetch_google_news_returns_empty_on_exception():
    with mock.patch.object(n.requests, "get", side_effect=Exception("boom")):
        assert n.fetch_google_news("台積電") == []


# ── fetch_yahoo_news ──────────────────────────────────────────────────────────

def test_fetch_yahoo_news_returns_recent_items():
    with mock.patch.object(n.yf, "Ticker") as m:
        m.return_value.news = YAHOO_ITEMS
        items = n.fetch_yahoo_news("2330.TW", days=3650)
    assert len(items) == 1
    assert items[0]['title'] == 'TSMC news'


def test_fetch_yahoo_news_handles_none_and_exception():
    with mock.patch.object(n.yf, "Ticker") as m:
        m.return_value.news = None
        assert n.fetch_yahoo_news("2330.TW") == []
    with mock.patch.object(n.yf, "Ticker", side_effect=Exception("boom")):
        assert n.fetch_yahoo_news("2330.TW") == []


# ── fetch_news ────────────────────────────────────────────────────────────────

def test_fetch_news_combines_zh_and_en_and_defaults_query_to_symbol():
    with mock.patch.object(n, "fetch_google_news", return_value=[{'title': 'a'}]) as m_zh, \
         mock.patch.object(n, "fetch_yahoo_news", return_value=[{'title': 'b'}]):
        result = n.fetch_news(company_name="", symbol="2330", ticker="2330.TW")
    assert result == {'zh': [{'title': 'a'}], 'en': [{'title': 'b'}]}
    assert m_zh.call_args.args[0] == "2330"  # falls back to symbol when no company_name


# ── fmt_report ────────────────────────────────────────────────────────────────

def test_fmt_report_renders_populated_sections():
    r = {
        'zh': [{'pub_date': '2026-09-01', 'title': '台積電新聞', 'source': '天下雜誌'}],
        'en': [{'pub_date': '2026-09-01', 'title': 'TSMC news', 'source': None}],
    }
    report = n.fmt_report('2330', '台積電', r)
    assert '台積電' in report
    assert '共1則' in report
    assert '[天下雜誌]' in report
    assert 'TSMC news' in report


def test_fmt_report_handles_no_headlines():
    report = n.fmt_report('2330', '', {'zh': [], 'en': []})
    assert report.count('無資料') == 2


# ── main() ────────────────────────────────────────────────────────────────────

def test_main_prints_usage_and_exits_when_no_symbols(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["news.py"])
    try:
        n.main()
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert e.code == 1
    assert "用法" in capsys.readouterr().out


def test_main_happy_path(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["news.py", "2330"])
    with mock.patch.object(n, "fetch_price_data", return_value=(pd.DataFrame(), "2330.TW")), \
         mock.patch.object(n, "fetch_company_name", return_value="台積電"), \
         mock.patch.object(n, "fetch_news", return_value={'zh': [], 'en': []}):
        n.main()
    assert "台積電" in capsys.readouterr().out


def test_main_reports_error_and_continues_on_exception(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["news.py", "BAD"])
    with mock.patch.object(n, "fetch_price_data", side_effect=ValueError("找不到股票")):
        n.main()  # must not raise
    assert "[錯誤]" in capsys.readouterr().out


if __name__ == "__main__":
    import inspect
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            params = inspect.signature(fn).parameters
            if params:
                print(f"SKIP {name} (needs pytest fixtures; run via `pytest`)")
                continue
            try:
                fn()
                print(f"OK  {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print("all checks passed" if not failures else f"{failures} test(s) failed")
    sys.exit(1 if failures else 0)
