#!/usr/bin/env python3
"""Self-check for news.py's parsing/filtering helpers (ponytail: non-trivial regex + date-cutoff logic)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date, datetime, timezone
from news import _parse_google_news_xml, _parse_yahoo_news_items

GOOGLE_XML_FIXTURE = """
<rss><channel>
<item>
<title>台積電宣布進駐這地！預期效率提升25%至50% - news.ebc.net.tw</title>
<link>https://example.com/a</link>
<pubDate>Tue, 01 Sep 2026 12:51:00 GMT</pubDate>
</item>
<item>
<title>舊聞標題沒有來源分隔符</title>
<link>https://example.com/b</link>
<pubDate>Tue, 01 Jan 2020 00:00:00 GMT</pubDate>
</item>
</channel></rss>
"""

YAHOO_ITEMS_FIXTURE = [
    {'content': {'title': 'Recent story', 'pubDate': '2026-09-02T04:20:27Z',
                 'provider': {'displayName': 'Reuters'},
                 'canonicalUrl': {'url': 'https://example.com/c'}}},
    {'content': {'title': 'Old story', 'pubDate': '2020-01-01T00:00:00Z',
                 'provider': {'displayName': 'AP'}}},
    {'content': {'title': 'No date, skipped'}},
]


def test_parse_google_news_filters_by_cutoff_and_splits_source():
    cutoff = datetime(2026, 8, 1, tzinfo=timezone.utc)
    items = _parse_google_news_xml(GOOGLE_XML_FIXTURE, cutoff, limit=10)
    assert len(items) == 1
    assert items[0]['title'] == '台積電宣布進駐這地！預期效率提升25%至50%'
    assert items[0]['source'] == 'news.ebc.net.tw'
    assert items[0]['pub_date'] == date(2026, 9, 1)


def test_parse_yahoo_news_filters_by_cutoff_and_skips_missing_date():
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    items = _parse_yahoo_news_items(YAHOO_ITEMS_FIXTURE, cutoff, limit=10)
    assert len(items) == 1
    assert items[0]['title'] == 'Recent story'
    assert items[0]['source'] == 'Reuters'


def test_limit_is_respected():
    cutoff = datetime(2020, 1, 1, tzinfo=timezone.utc)
    items = _parse_google_news_xml(GOOGLE_XML_FIXTURE, cutoff, limit=1)
    assert len(items) == 1
