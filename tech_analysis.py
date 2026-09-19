#!/usr/bin/env python3
"""
Taiwan Stock Technical Analysis — Data-Driven Edition
Instead of fixed if/else scoring, every technical signal is evaluated by its
actual historical hit rate over the past 6 months for the specific stock.
"""

from __future__ import annotations
import json
import os
import re
import sys
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# requests/yfinance transitively trigger urllib3's NotOpenSSLWarning at
# import time; filterwarnings must run before importing them to suppress it,
# which unavoidably breaks import-block contiguity for these two lines only.
warnings.filterwarnings('ignore')
import requests  # pylint: disable=wrong-import-position
import yfinance as yf  # pylint: disable=wrong-import-position

# Split out of this file to satisfy too-many-lines; re-exported here so every
# existing `tech_analysis.X` / `from tech_analysis import X` call site (tests,
# fundamentals.py, news.py, backtest_annual.py) keeps working unchanged.
from backtest_engine import (  # pylint: disable=wrong-import-position
    COND_GROUPS,
    HORIZON,
    MIN_SAMPLES,
    MOVE_THRESH,
    _group_of,
    backtest_conditions,
    baseline_stats,
    compute_outcomes,
)
from conditions import build_conditions  # pylint: disable=wrong-import-position
from fetchers import (  # pylint: disable=wrong-import-position
    _COMPANY_CACHE,
    calc_indicators,
    fetch_company_name,
    fetch_price_data,
    fetch_stock_earnings_date,
)
from market_indicators import fetch_buffett_indicator, fetch_vix  # pylint: disable=wrong-import-position
from report import (  # pylint: disable=wrong-import-position
    _REC_ICON,
    _REC_RANK,
    _direction_label,
    fmt_report,
    fmt_summary_table,
)

# Re-exported for backward compatibility (tests / sibling scripts reference
# these via `tech_analysis.X`) even though nothing below calls them directly —
# __all__ tells pylint's unused-import check that's intentional.
__all__ = [
    'COND_GROUPS', 'MIN_SAMPLES', '_group_of', '_COMPANY_CACHE',
    '_REC_ICON', '_REC_RANK', '_direction_label',
    'HORIZON', 'MOVE_THRESH', 'backtest_conditions', 'baseline_stats', 'compute_outcomes',
    'build_conditions', 'calc_indicators', 'fetch_company_name', 'fetch_price_data',
    'fetch_stock_earnings_date', 'fetch_buffett_indicator', 'fetch_vix',
    'fmt_report', 'fmt_summary_table',
    'analyze', 'analyze_institutional', 'fetch_institutional', 'fetch_market_regime',
    'fetch_macro_events', 'fetch_futures_sentiment', 'fetch_futures_net_position',
    'print_cli_error', 'main',
]


# ─────────────────────────────────────────────────────────────────────────────
# Institutional (三大法人) — TWSE / TPEx
# ─────────────────────────────────────────────────────────────────────────────

_TWSE_HDR = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.twse.com.tw/'}
_TPEX_HDR = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.tpex.org.tw/'}


def _parse_num(s: str) -> int:
    try:
        return int(str(s).replace(',', '').replace(' ', ''))
    except (ValueError, AttributeError):
        return 0


def _fetch_twse_day(date_str: str, symbol: str) -> dict | None:
    try:
        r = requests.get(
            'https://www.twse.com.tw/rwd/zh/fund/T86',
            params={'response': 'json', 'date': date_str, 'selectType': 'ALLBUT0999'},
            headers=_TWSE_HDR, timeout=12,
        )
        d = r.json()
        if d.get('stat') != 'OK':
            return None
        for row in d.get('data', []):
            if str(row[0]).strip() == symbol:
                return {
                    'date':    date_str,
                    'foreign': _parse_num(row[4]) + _parse_num(row[7]),
                    'trust':   _parse_num(row[10]),
                    'dealer':  _parse_num(row[11]),
                    'total':   _parse_num(row[18]),
                }
    except Exception:
        pass
    return None


def _fetch_tpex_day(date_str: str, symbol: str) -> dict | None:
    try:
        tpex_date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y/%m/%d')
        r = requests.get(
            'https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php',
            params={'l': 'zh-tw', 'o': 'json', 'se': 'EW', 't': 'D', 'd': tpex_date},
            headers=_TPEX_HDR, timeout=12,
        )
        d = r.json()
        for row in (d.get('aaData') or d.get('data', [])):
            if str(row[0]).strip() == symbol:
                return {
                    'date':    date_str,
                    'foreign': _parse_num(row[4]) + _parse_num(row[7]),
                    'trust':   _parse_num(row[10]),
                    'dealer':  _parse_num(row[13]),
                    'total':   _parse_num(row[17]),
                }
    except Exception:
        pass
    return None


def fetch_institutional(symbol: str, is_otc: bool = False, n_days: int = 20) -> pd.DataFrame:
    """Last `n_days` of 三大法人 net buy/sell, fetched in parallel across up to 54 calendar days."""
    fn       = _fetch_tpex_day if is_otc else _fetch_twse_day
    dates    = [(datetime.now() - timedelta(days=i)).strftime('%Y%m%d') for i in range(1, 55)]
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fn, dt, symbol): dt for dt in dates}
        for fut in as_completed(futures):
            row = fut.result()
            if row:
                results.append(row)
            if len(results) >= n_days:
                for f in futures:
                    f.cancel()
                break

    if not results:
        return pd.DataFrame()
    return (
        pd.DataFrame(results)
        .sort_values('date', ascending=False)
        .reset_index(drop=True)
        .head(n_days)
    )


def _consecutive(lst: list[int]) -> tuple[int, int]:
    buy = sell = 0
    for v in lst:
        if v > 0:
            if sell:
                break
            buy += 1
        elif v < 0:
            if buy:
                break
            sell += 1
        else:
            break
    return buy, sell


def _streak_signal(label: str, buy: int, sell: int, cum5: int) -> tuple[int, tuple | None]:
    """Score one party's (外資/投信) consecutive buy/sell streak.
    Returns (score_delta, signal_tuple_or_None)."""
    if buy >= 3:
        return 2, (label, f"連續買超 {buy} 日，近5日累積 {cum5:+,} 萬股")
    if buy >= 1:
        return 1, (label, f"買超 {buy} 日，近5日累積 {cum5:+,} 萬股")
    if sell >= 3:
        return -2, (label, f"連續賣超 {sell} 日，近5日累積 {cum5:+,} 萬股")
    if sell >= 1:
        return -1, (label, f"賣超 {sell} 日，近5日累積 {cum5:+,} 萬股")
    return 0, None


def _cum_n(lst: list[int], n: int) -> int:
    return sum(lst[:n]) // 10_000


def _co_signal(f_buy: int, f_sell: int, t_buy: int, t_sell: int) -> tuple[int, tuple | None]:
    """外資+投信 same-direction signal (同買/同賣)."""
    if f_buy >= 1 and t_buy >= 1:
        return 1, ("同買", "外資+投信同步買超，籌碼集中")
    if f_sell >= 1 and t_sell >= 1:
        return -1, ("同賣", "外資+投信同步賣超，賣壓沉重")
    return 0, None


def _f10_signal(f10: int | None) -> tuple | None:
    """外資 10-day cumulative direction signal, or None if not enough history."""
    if f10 is None:
        return None
    tag = "外資10日"
    if f10 > 0:
        return tag, f"近10日累積買超 {f10:,} 萬股，中期偏多"
    return tag, f"近10日累積賣超 {f10:,} 萬股，中期偏空"


def _institutional_summary(
    foreign: list[int], trust: list[int], total: list[int],
    cum5: tuple[int, int, int], streaks: tuple[int, int, int, int],
) -> dict:
    f5, t5, tot5 = cum5
    f_buy, f_sell, t_buy, t_sell = streaks
    return {
        'f5': f5, 't5': t5, 'tot5': tot5,
        'f_buy': f_buy, 'f_sell': f_sell,
        't_buy': t_buy, 't_sell': t_sell,
        'f_today':   foreign[0] // 10_000 if foreign else 0,
        't_today':   trust[0]   // 10_000 if trust   else 0,
        'tot_today': total[0]   // 10_000 if total   else 0,
    }


def _streak_score(streaks: tuple[int, int, int, int], f5: int, t5: int) -> tuple[int, list[tuple]]:
    """Combine 外資/投信 streak + co-signal scoring into (score, signal tuples)."""
    f_buy, f_sell, t_buy, t_sell = streaks
    score   = 0
    signals: list[tuple] = []
    for delta, sig in (
        _streak_signal("外資", f_buy, f_sell, f5),
        _streak_signal("投信", t_buy, t_sell, t5),
        _co_signal(f_buy, f_sell, t_buy, t_sell),
    ):
        score += delta
        if sig:
            signals.append(sig)
    return score, signals


def analyze_institutional(df_inst: pd.DataFrame) -> tuple[int, list[tuple], dict]:
    """Score 外資/投信 buy-sell streaks by fixed rule; returns (score, signal tuples, summary dict)."""
    if df_inst.empty:
        return 0, [], {}

    foreign = df_inst['foreign'].tolist()
    trust   = df_inst['trust'].tolist()
    total   = df_inst['total'].tolist()

    cum5 = (_cum_n(foreign, 5), _cum_n(trust, 5), _cum_n(total, 5))
    f_buy, f_sell = _consecutive(foreign)
    t_buy, t_sell = _consecutive(trust)

    streaks = (f_buy, f_sell, t_buy, t_sell)
    score, signals = _streak_score(streaks, cum5[0], cum5[1])

    f10_sig = _f10_signal(_cum_n(foreign, 10) if len(foreign) >= 10 else None)
    if f10_sig:
        signals.append(f10_sig)

    summary = _institutional_summary(foreign, trust, total, cum5, streaks)
    return score, signals, summary


# ─────────────────────────────────────────────────────────────────────────────
# Macro event calendar — FOMC / PCE / CPI, informational only (no score impact)
# ─────────────────────────────────────────────────────────────────────────────
# ponytail: each source is scraped from its official government calendar page
# (stable table/markup, no ToS issue) and fails silently on its own — one
# source breaking (page redesign, network hiccup) never blocks the others.

_UA = {"User-Agent": "Mozilla/5.0"}


def _parse_month_day(month_day: str, year: int) -> date | None:
    try:
        return datetime.strptime(f"{month_day.strip()} {year}", "%B %d %Y").date()
    except ValueError:
        return None


def _parse_fomc_events(html: str, today: date, horizon: date) -> list[dict]:
    events = []
    for year_str, block in re.findall(r'(\d{4}) FOMC Meetings</a></h4>(.*?)(?=<h4>|\Z)', html, re.S):
        year   = int(year_str)
        months = re.findall(r'fomc-meeting__month[^"]*"><strong>(\w+)</strong>', block)
        dates  = re.findall(r'fomc-meeting__date[^"]*">([^<]+)<', block)
        for month, day_range in zip(months, dates):
            last_day = re.sub(r'\*', '', day_range).split('-')[-1].strip()
            d = _parse_month_day(f"{month} {last_day}", year)
            if d and today <= d <= horizon:
                events.append({'date': d, 'name': 'FOMC利率決策會議'})
    return events


def _parse_bea_events(html: str, today: date, horizon: date) -> list[dict]:
    """BEA release schedule covers both PCE ('Personal Income and Outlays') and
    quarterly GDP estimates (titles starting 'GDP (...)' — excludes the unrelated
    'GDP by County' regional release, which isn't market-moving)."""
    events = []
    year_m = re.search(r'Year (\d{4})', html)
    year   = int(year_m.group(1)) if year_m else today.year
    for row in re.findall(r'<tr class="scheduled-releases-type-press">(.*?)</tr>', html, re.S):
        date_m  = re.search(r'release-date">([^<]+)<', row)
        title_m = re.search(r'release-title[^>]*>([^<]+)', row)
        if not date_m or not title_m:
            continue
        title = title_m.group(1).strip()
        if 'Personal Income and Outlays' in title:
            name = '美國PCE物價指數公布'
        elif title.startswith('GDP ('):
            name = '美國GDP公布'
        else:
            continue
        d = _parse_month_day(date_m.group(1), year)
        if d and today <= d <= horizon:
            events.append({'date': d, 'name': name})
    return events


def fetch_macro_events(days_ahead: int = 7) -> list[dict]:
    """Best-effort scrape of upcoming macro events within `days_ahead` days.
    Each source fails silently and independently — a missing FRED_API_KEY
    just means CPI is skipped, a federalreserve.gov redesign just means FOMC
    is skipped, etc.
    """
    today   = date.today()
    horizon = today + timedelta(days=days_ahead)
    events: list[dict] = []

    try:
        html = requests.get(
            "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
            headers=_UA, timeout=10,
        ).text
        events += _parse_fomc_events(html, today, horizon)
    except Exception:
        pass

    try:
        html = requests.get("https://www.bea.gov/news/schedule", headers=_UA, timeout=10).text
        events += _parse_bea_events(html, today, horizon)
    except Exception:
        pass

    api_key = os.environ.get('FRED_API_KEY')
    if api_key:
        for release_id, name in [(10, '美國CPI公布'), (50, '美國非農就業報告(NFP)')]:
            try:
                resp = requests.get(
                    "https://api.stlouisfed.org/fred/release/dates",
                    params={
                        'release_id': release_id,
                        'api_key': api_key,
                        'file_type': 'json',
                        'realtime_start': today.isoformat(),
                        'realtime_end': horizon.isoformat(),
                        # Upcoming releases have no data yet by definition —
                        # 'false' here silently strips every future date.
                        'include_release_dates_with_no_data': 'true',
                    },
                    timeout=10,
                ).json()
                for rd in resp.get('release_dates', []):
                    d = datetime.strptime(rd['date'], '%Y-%m-%d').date()
                    if today <= d <= horizon:
                        events.append({'date': d, 'name': name})
            except Exception:
                pass

    events.sort(key=lambda e: e['date'])
    return events


def fetch_market_regime() -> dict:
    """
    Determine market regime from Taiwan Weighted Index (^TWII) vs its MA200.
    A bonus is added to the combined score to favour staying long in bull markets
    and reducing exposure in bear markets.

    Regime table:
      TWII / MA200 ≥ 1.05  →  強多頭  bonus +0.25
      TWII / MA200 ≥ 1.00  →  多頭    bonus +0.15
      TWII / MA200 ≥ 0.95  →  中性    bonus  0.00
      TWII / MA200 <  0.95 →  空頭    bonus -0.15
    """
    try:
        df = yf.download("^TWII", period="400d", progress=False, auto_adjust=True)
        if df.empty:
            return {'regime': '中性', 'bonus': 0.0}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close    = df['Close']
        ma200    = close.rolling(200).mean()
        current  = float(close.iloc[-1])
        ma200_v  = float(ma200.iloc[-1])
        if pd.isna(ma200_v):
            return {'regime': '中性', 'bonus': 0.0}
        ratio = current / ma200_v
        if ratio >= 1.05:
            regime, bonus = '強多頭', +0.25
        elif ratio >= 1.00:
            regime, bonus = '多頭',   +0.15
        elif ratio >= 0.95:
            regime, bonus = '中性',    0.00
        else:
            regime, bonus = '空頭',   -0.15
        return {'regime': regime, 'bonus': bonus, 'twii': current, 'ma200': ma200_v, 'ratio': ratio}
    except Exception:
        return {'regime': '中性', 'bonus': 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Futures positioning (外資臺股期貨淨部位) — market-wide macro overlay
# ─────────────────────────────────────────────────────────────────────────────
# ponytail: TAIFEX only serves single-date queries and fronts them with Cloudflare
# rate-limiting — no bulk history download exists. So we cache one date per run
# and let the sample accumulate naturally over repeated invocations instead of
# ever bulk-scraping. Below FUTURES_MIN_SAMPLES the signal contributes 0.

FUTURES_CACHE_FILE  = Path(__file__).parent / "taifex_net_pos_cache.json"
FUTURES_MIN_SAMPLES = 12  # need >=3 per quartile bucket before trusting it at all


def _load_futures_cache() -> dict[str, float]:
    if FUTURES_CACHE_FILE.exists():
        try:
            return json.loads(FUTURES_CACHE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_futures_cache(cache: dict[str, float]) -> None:
    try:
        FUTURES_CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=0))
    except OSError:
        pass


def fetch_futures_net_position(date_str: str) -> tuple[float | None, float | None]:
    """
    POST TAIFEX 三大法人-依日期查詢 for TXF (臺股期貨), return (外資未平倉淨部位口數,
    外資未平倉空方口數) — both come off the same row of the same response, so
    capturing the second costs nothing extra.
    Single request, no retries — a 429/Cloudflare block or a non-trading day both
    just mean "no data today", handled the same as any other missing day.
    """
    try:
        r = requests.post(
            "https://www.taifex.com.tw/cht/3/futContractsDate",
            data={"queryType": "1", "queryDate": date_str, "commodityId": "TXF"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
    except requests.RequestException:
        return None, None
    if r.status_code == 429 or "Just a moment" in r.text:
        return None, None

    html = r.text
    table_start = html.find("table_f")
    idx = html.find("外資", table_start if table_start != -1 else 0)
    if idx == -1:
        return None, None
    block = html[idx: idx + 3000]
    text  = re.sub(r"<[^>]+>", " ", block)
    nums  = re.findall(r"-?[\d][\d,]*", text)
    if len(nums) < 11:
        return None, None
    # Row layout: [多方口數,多方金額,空方口數,空方金額,多空淨額口數(交易),多空淨額金額(交易),
    #  多方口數(未平倉,idx6),多方金額,空方口數(未平倉,idx8),空方金額,多空淨額口數(未平倉,idx10),多空淨額金額]
    try:
        short_oi = float(nums[8].replace(",", ""))
    except ValueError:
        short_oi = None
    try:
        net_pos = float(nums[10].replace(",", ""))
    except ValueError:
        net_pos = None
    return net_pos, short_oi


def _known_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Label each day's forward move as up(1)/flat(0)/down(-1), dropping unresolved rows."""
    fwd     = (df['close'].shift(-HORIZON) - df['close']) / df['close']
    outcome = pd.Series(np.nan, index=df.index)
    outcome[fwd >  MOVE_THRESH] =  1.0
    outcome[fwd < -MOVE_THRESH] = -1.0
    outcome[(fwd >= -MOVE_THRESH) & (fwd <= MOVE_THRESH)] = 0.0
    return df.assign(outcome=outcome).dropna(subset=['outcome'])


def _bucket_by_quartile(
    known: pd.DataFrame, today_val: float
) -> tuple[pd.DataFrame, pd.Interval | None, str]:
    """Quartile-bucket `known` by 'val' and find today's bucket. Returns (known, bucket, note-if-none)."""
    try:
        known = known.copy()
        known['quartile'] = pd.qcut(known['val'], 4, duplicates='drop')
    except ValueError:
        return known, None, "數值變化不足以分組，暫不計分"
    for interval in known['quartile'].cat.categories:
        if today_val in interval or today_val == interval.right:
            return known, interval, ""
    # today's value sits outside the historical range entirely — extrapolating
    # off the last known bucket would overstate confidence, so skip scoring.
    return known, None, "今日數值超出歷史分組範圍，暫不計分"


def _score_bucket(known: pd.DataFrame, bucket: pd.Interval, base_edge: float) -> dict:
    sub  = known[known['quartile'] == bucket]
    n    = len(sub)
    up   = float((sub['outcome'] == 1).mean())
    down = float((sub['outcome'] == -1).mean())
    excess = (up - down) - base_edge
    conf   = min(n / 20.0, 1.0)
    bonus  = excess * conf * 0.5  # halved: shared across every stock in the batch

    return {
        'bucket':     f"{bucket.left:,.0f} ~ {bucket.right:,.0f}",
        'bucket_n':   n,
        'up_rate':    up,
        'down_rate':  down,
        'excess':     excess,
        'confidence': conf,
        'bonus':      bonus,
        'note':       f"樣本{n}天 上漲{up:.0%} 下跌{down:.0%}",
    }


def _quartile_excess_bonus(values: pd.Series, close: pd.Series, today_val: float) -> dict:
    """
    Shared scoring core for a market-wide macro series: bucket `values` (date-
    indexed) into quartiles and score today's bucket by excess edge vs TWII's own
    baseline — same methodology as backtest_conditions(), confidence-weighted by
    sample size, halved because it applies identically to every stock in a batch.
    """
    df = pd.DataFrame({'val': values}).join(close.rename('close'), how='inner')
    if len(df) < FUTURES_MIN_SAMPLES:
        return {'bonus': 0.0, 'bucket': None, 'note': f"樣本僅 {len(df)} 天，未達 {FUTURES_MIN_SAMPLES} 天門檻，暫不計分"}

    known = _known_outcomes(df)
    if len(known) < FUTURES_MIN_SAMPLES:
        return {'bonus': 0.0, 'bucket': None, 'note': f"已知結果樣本僅 {len(known)} 天，暫不計分"}

    base_edge = float((known['outcome'] == 1).mean() - (known['outcome'] == -1).mean())
    known, bucket, note = _bucket_by_quartile(known, today_val)
    if bucket is None:
        return {'bonus': 0.0, 'bucket': None, 'note': note}

    return _score_bucket(known, bucket, base_edge)


def _dated_series(raw: dict[str, float]) -> pd.Series:
    s = pd.Series(raw)
    s.index = pd.to_datetime(s.index, format="%Y/%m/%d")
    return s.sort_index()


def _fetch_twii_close() -> pd.Series | None:
    try:
        twii = yf.download("^TWII", period="200d", progress=False, auto_adjust=True)
        if twii.empty:
            return None
        if isinstance(twii.columns, pd.MultiIndex):
            twii.columns = twii.columns.get_level_values(0)
        return twii['Close']
    except Exception:
        return None


def _get_or_fetch_futures_entry(cache: dict, date_str: str) -> dict | None:
    """Look up today's cached TAIFEX entry, fetching (and caching) it if missing."""
    entry = cache.get(date_str)
    if not isinstance(entry, dict):  # missing, or legacy plain-float cache format
        entry = None
    if entry is None:
        net_val, short_val = fetch_futures_net_position(date_str)
        if net_val is not None:
            entry = {'net': net_val, 'short': short_val}
            cache[date_str] = entry
            _save_futures_cache(cache)
    return entry


def _split_legacy_cache(cache: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Tolerate legacy cache entries that were a plain float (net-only, no short)."""
    net_raw:   dict[str, float] = {}
    short_raw: dict[str, float] = {}
    for d, v in cache.items():
        if isinstance(v, dict):
            if v.get('net') is not None:
                net_raw[d] = v['net']
            if v.get('short') is not None:
                short_raw[d] = v['short']
        elif isinstance(v, (int, float)):
            net_raw[d] = v
    return net_raw, short_raw


def _score_net_and_short(
    result: dict, close: pd.Series, net_raw: dict[str, float], short_raw: dict[str, float]
) -> None:
    """Fill `result` in place with net-position and short-OI quartile scoring."""
    if result['net_pos'] is not None:
        r_net = _quartile_excess_bonus(_dated_series(net_raw), close, result['net_pos'])
        result['bonus']    = r_net['bonus']
        result['bucket']   = r_net.get('bucket')
        result['bucket_n'] = r_net.get('bucket_n')
        result['note']     = r_net['note']
    else:
        result['note'] = "無淨部位資料"

    if result['short_oi'] is not None and len(short_raw) >= FUTURES_MIN_SAMPLES:
        r_short = _quartile_excess_bonus(_dated_series(short_raw), close, result['short_oi'])
        result['short_bonus']  = r_short['bonus']
        result['short_bucket'] = r_short.get('bucket')
        result['short_note']   = r_short['note']
    else:
        result['short_note'] = f"空單口數樣本僅 {len(short_raw)} 天，暫不計分"


def fetch_futures_sentiment() -> dict:
    """
    Market-wide macro overlay from 外資 TXF 未平倉部位: net position (direction)
    and gross short open interest (hedging/bearish pressure), each scored
    independently via _quartile_excess_bonus() and summed into one bonus.
    """
    close = _fetch_twii_close()
    if close is None:
        return {}

    date_str = close.index[-1].strftime("%Y/%m/%d")

    cache = _load_futures_cache()
    entry = _get_or_fetch_futures_entry(cache, date_str)

    result = {
        'net_pos':     entry.get('net')   if entry else None,
        'short_oi':    entry.get('short') if entry else None,
        'sample_n':    len(cache),
        'bonus':       0.0,
        'short_bonus': 0.0,
        'bucket':      None,
    }
    if entry is None:
        result['note'] = "今日資料無法取得（可能遭 TAIFEX 限速或休市）"
        return result

    net_raw, short_raw = _split_legacy_cache(cache)
    _score_net_and_short(result, close, net_raw, short_raw)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def _regime_thresholds(regime: dict | None) -> dict:
    """Per-regime add/reduce/strong thresholds and move_thresh: bull → hard to
    reduce; bear → easy to reduce."""
    regime_params: dict[str, dict] = {
        '強多頭': {"add": 0.25, "reduce": -0.40, "strong": -0.80, "move": 0.020},
        '多頭':   {"add": 0.25, "reduce": -0.25, "strong": -0.60, "move": 0.015},
        '中性':   {"add": 0.25, "reduce": -0.25, "strong": -0.60, "move": 0.015},
        '空頭':   {"add": 0.25, "reduce": -0.25, "strong": -0.60, "move": 0.015},
    }
    return regime_params.get((regime or {}).get('regime', '中性'), regime_params['中性'])


def _score_technical(df: pd.DataFrame, move_thresh: float) -> tuple[dict, list[dict], dict, float, dict]:
    """Backtest every condition, then pick one best-|weight| active signal per
    indicator group (prevents double-counting) and sum those into tech_score."""
    conditions = build_conditions(df)
    outcomes   = compute_outcomes(df, move_thresh=move_thresh)
    base       = baseline_stats(outcomes)
    bt         = backtest_conditions(conditions, outcomes, base)

    today_flags: dict[str, bool] = {
        name: bool(cond.iloc[-1]) for name, cond in conditions.items()
    }

    group_best: dict[str, dict] = {}  # group → best active condition stats
    active_bt:  list[dict]       = []
    for name, stats in bt.items():
        if not today_flags.get(name, False):
            continue
        entry  = {'name': name, **stats}
        active_bt.append(entry)
        grp    = stats['group']
        if grp not in group_best or abs(stats['weight']) > abs(group_best[grp]['weight']):
            group_best[grp] = entry

    tech_score = sum(s['weight'] for s in group_best.values())
    active_bt.sort(key=lambda x: abs(x['weight']), reverse=True)
    return bt, active_bt, group_best, tech_score, base


def _recommendation_tier(combined: float, rp: dict) -> str:
    if combined >= 0.6:
        return "強力加碼"
    if combined >= rp['add']:
        return "加碼"
    if combined <= rp['strong']:
        return "強力減碼"
    if combined <= rp['reduce']:
        return "減碼"
    return "持平"


def _price_snapshot(df: pd.DataFrame) -> dict:
    def _s(col):
        if col not in df.columns:
            return None
        v = df[col].iloc[-1]
        return None if pd.isna(v) else float(v)

    price     = _s('Close')
    prev_p    = float(df['Close'].iloc[-2])
    price_chg = (price - prev_p) if price else None
    price_pct = (price_chg / prev_p * 100) if price_chg else None

    return {
        'price':          price,
        'price_chg':      price_chg,
        'price_pct':      price_pct,
        'K':   _s('K'),   'D':          _s('D'),
        'RSI': _s('RSI'), 'MACD':       _s('MACD'),
        'MACD_Signal':    _s('MACD_Signal'),
        'MA5':  _s('MA5'),  'MA20': _s('MA20'),
        'MA60': _s('MA60'), 'MA200': _s('MA200'),
        'BIAS5': _s('BIAS5'), 'BIAS20': _s('BIAS20'), 'BIAS60': _s('BIAS60'),
    }


def _combine_and_recommend(tech_score: float, inst_score: float, macro: dict, rp: dict) -> dict:
    """Fold institutional/regime/futures adjustments into tech_score and pick a tier."""
    inst_normalized = inst_score * 0.12  # normalize (-5..+5) to tech_score's scale
    # TWII vs MA200 bull/bear tilt so the strategy doesn't fight the tape
    regime_bonus = (macro.get('regime') or {}).get('bonus', 0.0)
    # 外資 TXF 淨部位 + 空方口數，bucketed against its own historical excess edge;
    # 0 until enough cached samples exist (see fetch_futures_sentiment).
    futures = macro.get('futures')
    futures_bonus = (futures or {}).get('bonus', 0.0) + (futures or {}).get('short_bonus', 0.0)
    combined = tech_score + inst_normalized + regime_bonus + futures_bonus

    return {
        'regime_bonus':   regime_bonus,
        'futures_bonus':  futures_bonus,
        'combined':       combined,
        'recommendation': _recommendation_tier(combined, rp),
    }


def analyze(df: pd.DataFrame, symbol: str, df_inst: pd.DataFrame, macro: dict | None = None) -> dict:
    """Backtest every condition on `df`, score today's active ones, and combine
    with institutional/regime/futures adjustments into a recommendation dict.
    `macro` holds the optional market-wide overlays: vix/buffett/regime/futures."""
    if len(df) < 30:
        raise ValueError("歷史數據不足")
    macro = macro or {}

    rp = _regime_thresholds(macro.get('regime'))
    bt, active_bt, group_best, tech_score, base = _score_technical(df, rp['move'])
    inst_score, inst_signals, inst_summary = analyze_institutional(df_inst)
    scores = _combine_and_recommend(tech_score, inst_score, macro, rp)

    return {
        'symbol':         symbol,
        **_price_snapshot(df),
        'vix':            macro.get('vix') or {},
        'buffett':        macro.get('buffett') or {},
        'regime':         macro.get('regime') or {},
        'regime_bonus':   scores['regime_bonus'],
        'futures':        macro.get('futures') or {},
        'futures_bonus':  scores['futures_bonus'],
        'base':           base,
        'active_bt':      active_bt,
        'group_best':     group_best,
        'all_bt':         bt,
        'tech_score':     tech_score,
        'inst_score':     inst_score,
        'combined':       scores['combined'],
        'recommendation': scores['recommendation'],
        'inst_signals':   inst_signals,
        'inst_summary':   inst_summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def print_cli_error(sym: str, exc: Exception) -> None:
    """Shared per-symbol error report for the three CLI scripts' main() loops."""
    print(f"[錯誤] {sym}: {exc}")
    traceback.print_exc()


def main():
    """CLI entry point: analyze each symbol in sys.argv, print a report per stock,
    then a sorted summary table if more than one symbol was given."""
    if len(sys.argv) < 2:
        print("用法: python3 tech_analysis.py <代號1> [代號2] ...")
        print("範例: python3 tech_analysis.py 2330 2317 0050")
        sys.exit(1)

    print("正在抓取總體市場指標 (VIX / 巴菲特指標 / 台股趨勢 / 外資期貨淨部位)...")
    macro = {
        'vix':     fetch_vix(),
        'buffett': fetch_buffett_indicator(),
        'regime':  fetch_market_regime(),
        'futures': fetch_futures_sentiment(),
    }
    macro_events = fetch_macro_events()
    if macro_events:
        ev_str = "、".join(f"{e['date'].strftime('%-m/%-d')} {e['name']}" for e in macro_events)
        print(f"⚠ 重大事件提醒（未來7天）：{ev_str}")

    results = []
    for sym in sys.argv[1:]:
        sym = sym.strip().upper()
        try:
            print(f"\n正在抓取 {sym} 價格數據...")
            df, ticker = fetch_price_data(sym)
            is_otc = ticker.endswith('.TWO')
            df = calc_indicators(df)

            company_name = fetch_company_name(sym, is_otc=is_otc)
            print(f"正在抓取 {sym} 三大法人數據...")
            df_inst = fetch_institutional(sym, is_otc=is_otc)

            result = analyze(df, sym, df_inst, macro=macro)
            result['company_name'] = company_name
            result['earnings_date'] = fetch_stock_earnings_date(ticker)
            print(fmt_report(result))
            results.append(result)
        except Exception as e:
            print_cli_error(sym, e)

    if len(results) > 1:
        print()
        print(fmt_summary_table(results))


if __name__ == "__main__":
    main()
