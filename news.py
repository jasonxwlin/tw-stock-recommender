#!/usr/bin/env python3
"""
Taiwan Stock News Snapshot — reference only, no scoring, no sentiment score.
Pulls recent headlines (Traditional Chinese via Google News RSS, English via
yfinance) so a human (or the analyst commentary step in the news-analysis
skill) can read and judge for themselves. Does not feed into tech_analysis.py.
"""
from __future__ import annotations
import re
import sys
import warnings
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

# requests/yfinance (and tech_analysis, which imports them too) transitively
# trigger urllib3's NotOpenSSLWarning at import time; filterwarnings must run
# before importing any of them to suppress it, which unavoidably breaks
# import-block contiguity for these lines only.
warnings.filterwarnings('ignore')
import requests  # pylint: disable=wrong-import-position
import yfinance as yf  # pylint: disable=wrong-import-position

from tech_analysis import fetch_price_data, fetch_company_name, print_cli_error  # pylint: disable=wrong-import-position

_UA = {"User-Agent": "Mozilla/5.0"}


def _parse_google_news_xml(xml: str, cutoff: datetime, limit: int) -> list[dict]:
    items = []
    for raw in re.findall(r'<item>(.*?)</item>', xml, re.S):
        title_m = re.search(r'<title>(.*?)</title>', raw, re.S)
        pub_m   = re.search(r'<pubDate>(.*?)</pubDate>', raw)
        link_m  = re.search(r'<link>(.*?)</link>', raw)
        if not title_m or not pub_m:
            continue
        try:
            pub_dt = parsedate_to_datetime(pub_m.group(1).strip())
        except (TypeError, ValueError):
            continue
        if pub_dt < cutoff:
            continue
        title = title_m.group(1).strip()
        source = None
        if ' - ' in title:
            title, source = title.rsplit(' - ', 1)
        items.append({
            'title': title.strip(),
            'source': source,
            'pub_date': pub_dt.date(),
            'link': link_m.group(1).strip() if link_m else None,
        })
    items.sort(key=lambda x: x['pub_date'], reverse=True)
    return items[:limit]


def fetch_google_news(query: str, days: int = 7, limit: int = 15) -> list[dict]:
    """Best-effort: Traditional Chinese headlines from Google News RSS (public feed)."""
    try:
        xml = requests.get(
            "https://news.google.com/rss/search",
            params={'q': query, 'hl': 'zh-TW', 'gl': 'TW', 'ceid': 'TW:zh-Hant'},
            headers=_UA, timeout=10,
        ).text
    except Exception:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return _parse_google_news_xml(xml, cutoff, limit)


def _parse_yahoo_news_items(raw_items: list[dict], cutoff: datetime, limit: int) -> list[dict]:
    items = []
    for it in raw_items:
        c = it.get('content', {}) or {}
        pub_str = c.get('pubDate')
        if not pub_str:
            continue
        try:
            pub_dt = datetime.fromisoformat(pub_str.replace('Z', '+00:00'))
        except ValueError:
            continue
        if pub_dt < cutoff:
            continue
        items.append({
            'title': c.get('title', ''),
            'source': (c.get('provider') or {}).get('displayName'),
            'pub_date': pub_dt.date(),
            'link': (c.get('canonicalUrl') or {}).get('url'),
        })
    items.sort(key=lambda x: x['pub_date'], reverse=True)
    return items[:limit]


def fetch_yahoo_news(ticker: str, days: int = 7, limit: int = 10) -> list[dict]:
    """Best-effort: English headlines from yfinance (mostly US/global-investor angle)."""
    try:
        raw_items = yf.Ticker(ticker).news or []
    except Exception:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return _parse_yahoo_news_items(raw_items, cutoff, limit)


def fetch_news(company_name: str, symbol: str, ticker: str, days: int = 7) -> dict:
    """Recent Chinese (Google News) + English (yfinance) headlines — reference only, no score."""
    query = company_name or symbol
    return {
        'zh': fetch_google_news(query, days=days),
        'en': fetch_yahoo_news(ticker, days=days),
    }


def fmt_report(symbol: str, company_name: str, r: dict) -> str:
    """Render fetch_news()'s result dict into the raw headline-list text report."""
    W = 70
    lines: list[str] = []
    lines.append("=" * W)
    name_str = f"  {company_name}" if company_name else ""
    lines.append(f"  股票代號: {symbol}{name_str}  [消息面快照 - 原始新聞列表，僅供參考，不評分]")

    lines.append("-" * W)
    zh = r.get('zh') or []
    lines.append(f"  中文新聞（近7日，共{len(zh)}則）:")
    if zh:
        for it in zh:
            src = f"  [{it['source']}]" if it.get('source') else ""
            lines.append(f"    {it['pub_date']}  {it['title']}{src}")
    else:
        lines.append("    無資料")

    lines.append("-" * W)
    en = r.get('en') or []
    lines.append(f"  英文新聞（近7日，共{len(en)}則）:")
    if en:
        for it in en:
            src = f"  [{it['source']}]" if it.get('source') else ""
            lines.append(f"    {it['pub_date']}  {it['title']}{src}")
    else:
        lines.append("    無資料")

    lines.append("=" * W)
    return "\n".join(lines)


def main():
    """CLI entry point: print a news snapshot for each symbol in sys.argv."""
    if len(sys.argv) < 2:
        print("用法: python3 news.py <代號1> [代號2] ...")
        print("範例: python3 news.py 2330 2317")
        sys.exit(1)

    for sym in sys.argv[1:]:
        sym = sym.strip().upper()
        try:
            print(f"\n正在抓取 {sym} 消息面數據...")
            _, ticker = fetch_price_data(sym)
            is_otc = ticker.endswith('.TWO')
            company_name = fetch_company_name(sym, is_otc=is_otc)
            result = fetch_news(company_name, sym, ticker)
            print(fmt_report(sym, company_name, result))
        except Exception as e:
            print_cli_error(sym, e)


if __name__ == "__main__":
    main()
