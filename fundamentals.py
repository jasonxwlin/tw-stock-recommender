#!/usr/bin/env python3
"""
Taiwan Stock Fundamental Snapshot — reference only, no scoring.
Valuation percentile (P/E, P/B vs trailing 12mo), monthly revenue YoY trend,
and quarterly profitability trend (ROE, gross margin). Does not feed into
tech_analysis.py's recommendation — purely informational.
"""
from __future__ import annotations
import json
import re
import sys
import traceback
import warnings
from datetime import date
from pathlib import Path

# requests/yfinance (and tech_analysis, which imports them too) transitively
# trigger urllib3's NotOpenSSLWarning at import time; filterwarnings must run
# before importing any of them to suppress it, which unavoidably breaks
# import-block contiguity for these lines only.
warnings.filterwarnings('ignore')
import requests  # pylint: disable=wrong-import-position
import yfinance as yf  # pylint: disable=wrong-import-position

from tech_analysis import fetch_price_data, fetch_company_name  # pylint: disable=wrong-import-position

_VALUATION_CACHE_FILE = Path(__file__).parent / "valuation_cache.json"
_TWSE_HDR = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.twse.com.tw/'}
_TPEX_HDR = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.tpex.org.tw/'}
_MOPS_HDR = {'User-Agent': 'Mozilla/5.0'}


def _roc_year(d: date) -> int:
    return d.year - 1911


def _months_back(n: int) -> list[date]:
    """First-of-month dates for the past n months, newest first."""
    out = []
    y, m = date.today().year, date.today().month
    for _ in range(n):
        out.append(date(y, m, 1))
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Valuation percentile — P/E, P/B vs trailing 12mo history
# ─────────────────────────────────────────────────────────────────────────────
# ponytail: TWSE gives a whole month per request (one call/month); TPEx has no
# such per-stock-per-month endpoint, only whole-market-per-day, so we sample
# one day per month instead. Both fail silently and independently per month.
#
# Past months never change once published, so they're cached to disk forever
# (same pattern as tech_analysis.py's taifex_net_pos_cache.json) — a fresh run
# across N stocks only needs ~N fresh requests (this month) instead of ~12N,
# which is what was tripping TWSE's anti-flood protection (HTTP 428).

def _load_valuation_cache() -> dict:
    if _VALUATION_CACHE_FILE.exists():
        try:
            return json.loads(_VALUATION_CACHE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_valuation_cache(cache: dict) -> None:
    try:
        _VALUATION_CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=0))
    except OSError:
        pass


def _fetch_pe_pb_twse_month(symbol: str, first_of_month: date) -> list[dict]:
    try:
        r = requests.get(
            "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU",
            params={'date': first_of_month.strftime('%Y%m01'), 'stockNo': symbol, 'response': 'json'},
            headers=_TWSE_HDR, timeout=10,
        ).json()
    except Exception:
        return []
    rows = []
    for row in r.get('data', []):
        try:
            rows.append({'pe': float(row[3]), 'pb': float(row[4])})
        except (ValueError, IndexError):
            continue
    return rows


def _fetch_pe_pb_tpex_day(symbol: str, day: date) -> dict | None:
    date_str = f"{_roc_year(day)}/{day.month:02d}/{day.day:02d}"
    try:
        r = requests.get(
            "https://www.tpex.org.tw/web/stock/aftertrading/peratio_analysis/pera_result.php",
            params={'l': 'zh-tw', 'd': date_str, 'o': 'json'},
            headers=_TPEX_HDR, timeout=10,
        ).json()
        rows = r.get('tables', [{}])[0].get('data', [])
    except Exception:
        return None
    for row in rows:
        if row[0] == symbol:
            try:
                return {'pe': float(row[2]), 'pb': float(row[6])}
            except (ValueError, IndexError):
                return None
    return None


def fetch_valuation_snapshot(symbol: str, is_otc: bool, months: int = 12) -> dict:
    """Best-effort: current P/E, P/B only (no percentile — a partially-filled
    cache would make it misleading). Trailing `months` of history still builds
    up in the cache in the background: one new past month is fetched per call,
    plus the current month (always refreshed since it's still accumulating).
    Full history fills in gradually over ~`months` repeated invocations,
    staying well under TWSE's anti-flood threshold either way."""
    cache = _load_valuation_cache()
    stock_cache = cache.setdefault(symbol, {})
    current_ym = date.today().strftime('%Y-%m')
    dirty = False
    fetched_new_past_month = False

    def _fetch(d: date) -> list[dict]:
        if is_otc:
            row = _fetch_pe_pb_tpex_day(symbol, d.replace(day=5))
            return [row] if row else []
        return _fetch_pe_pb_twse_month(symbol, d)

    pe_hist, pb_hist = [], []
    for d in reversed(_months_back(months)):
        ym = d.strftime('%Y-%m')
        if ym == current_ym:
            rows = _fetch(d)
        elif ym in stock_cache:
            rows = stock_cache[ym]
        elif not fetched_new_past_month:
            rows = _fetch(d)
            fetched_new_past_month = True
            if rows:
                stock_cache[ym] = rows
                dirty = True
        else:
            rows = []
        for row in rows:
            pe_hist.append(row['pe'])
            pb_hist.append(row['pb'])

    if dirty:
        _save_valuation_cache(cache)
    if not pe_hist:
        return {}
    return {'pe': pe_hist[-1], 'pb': pb_hist[-1] if pb_hist else None}


# ─────────────────────────────────────────────────────────────────────────────
# Monthly revenue YoY trend — MOPS historical monthly revenue report
# ─────────────────────────────────────────────────────────────────────────────

_REV_ROW_RE = re.compile(
    r'<tr align=right><td align=center>(\d+)</td>'
    r'<td align=left>([^<]*)</td>'
    r'<td nowrap>\s*([\-\d,]+)</td>'
    r'<td nowrap>\s*([\-\d,]+)</td>'
    r'<td nowrap>\s*([\-\d,]+)</td>'
    r'<td nowrap>\s*([\-\d.]+)</td>'
    r'<td nowrap>\s*([\-\d.]+)</td>',
    re.I,
)


def _parse_revenue_yoy(html: str, symbol: str) -> float | None:
    for m in _REV_ROW_RE.finditer(html):
        if m.group(1) == symbol:
            try:
                return float(m.group(7))
            except ValueError:
                return None
    return None


def fetch_monthly_revenue(symbol: str, is_otc: bool, months: int = 6) -> list[dict]:
    """Best-effort: trailing `months` of YoY revenue growth, oldest first.
    Skips the current calendar month — MOPS publishes ~10 days after month end,
    so this month's figure usually isn't out yet."""
    market = 'otc' if is_otc else 'sii'
    out = []
    for d in reversed(_months_back(months + 1)[1:]):
        url = f"https://mopsov.twse.com.tw/nas/t21/{market}/t21sc03_{_roc_year(d)}_{d.month}_0.html"
        try:
            html = requests.get(url, headers=_MOPS_HDR, timeout=10).content.decode('big5', errors='ignore')
        except Exception:
            continue
        yoy = _parse_revenue_yoy(html, symbol)
        if yoy is not None:
            out.append({'year_month': f"{d.year}/{d.month:02d}", 'yoy_pct': yoy})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Quarterly profitability trend — ROE, gross margin (yfinance)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_quality_trend(ticker: str, quarters: int = 4) -> list[dict]:
    """Best-effort: trailing `quarters` of single-quarter ROE and gross margin.
    ROE here is quarterly net income / quarter-end equity (not annualized) —
    a trend indicator across quarters, not comparable to headline annual ROE."""
    try:
        t = yf.Ticker(ticker)
        qf, qbs = t.quarterly_financials, t.quarterly_balance_sheet
        if qf is None or qf.empty or qbs is None or qbs.empty:
            return []
        equity_row = 'Common Stock Equity' if 'Common Stock Equity' in qbs.index else 'Stockholders Equity'
        out = []
        for col in qf.columns[:quarters]:
            revenue = qf.at['Total Revenue', col] if 'Total Revenue' in qf.index else None
            gross   = qf.at['Gross Profit', col] if 'Gross Profit' in qf.index else None
            net     = qf.at['Net Income', col] if 'Net Income' in qf.index else None
            equity  = qbs.at[equity_row, col] if col in qbs.columns and equity_row in qbs.index else None
            out.append({
                'quarter': col.date().isoformat(),
                'gross_margin': (gross / revenue * 100) if gross and revenue else None,
                'roe': (net / equity * 100) if net and equity else None,
            })
        return list(reversed(out))
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Combine + report
# ─────────────────────────────────────────────────────────────────────────────

def analyze_fundamentals(symbol: str, is_otc: bool, ticker: str) -> dict:
    return {
        'valuation': fetch_valuation_snapshot(symbol, is_otc),
        'revenue':   fetch_monthly_revenue(symbol, is_otc),
        'quality':   fetch_quality_trend(ticker),
    }


def fmt_report(symbol: str, company_name: str, r: dict) -> str:
    W = 70
    lines: list[str] = []
    lines.append("=" * W)
    name_str = f"  {company_name}" if company_name else ""
    lines.append(f"  股票代號: {symbol}{name_str}  [基本面快照 - 僅供參考，不計分]")

    lines.append("-" * W)
    val = r.get('valuation') or {}
    if val:
        lines.append("  估值:")
        lines.append(f"    本益比: {val['pe']:.2f}")
        if val.get('pb') is not None:
            lines.append(f"    股價淨值比: {val['pb']:.2f}")
    else:
        lines.append("  估值: 資料無法取得")

    lines.append("-" * W)
    rev = r.get('revenue') or []
    if rev:
        lines.append("  月營收年增率趨勢:")
        for item in rev:
            lines.append(f"    {item['year_month']}: {item['yoy_pct']:+.2f}%")
    else:
        lines.append("  月營收趨勢: 資料無法取得")

    lines.append("-" * W)
    qual = r.get('quality') or []
    if qual:
        lines.append("  獲利能力趨勢（近四季，單季數字）:")
        for item in qual:
            roe_str = f"{item['roe']:.1f}%" if item['roe'] is not None else "N/A"
            gm_str  = f"{item['gross_margin']:.1f}%" if item['gross_margin'] is not None else "N/A"
            lines.append(f"    {item['quarter']}: 單季ROE={roe_str}  毛利率={gm_str}")
    else:
        lines.append("  獲利能力趨勢: 資料無法取得")

    lines.append("=" * W)
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("用法: python3 fundamentals.py <代號1> [代號2] ...")
        print("範例: python3 fundamentals.py 2330 2317")
        sys.exit(1)

    for sym in sys.argv[1:]:
        sym = sym.strip().upper()
        try:
            print(f"\n正在抓取 {sym} 基本面數據...")
            _, ticker = fetch_price_data(sym)
            is_otc = ticker.endswith('.TWO')
            company_name = fetch_company_name(sym, is_otc=is_otc)
            result = analyze_fundamentals(sym, is_otc, ticker)
            print(fmt_report(sym, company_name, result))
        except Exception as e:
            print(f"[錯誤] {sym}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
