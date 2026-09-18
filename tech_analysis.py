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
from ta.trend import MACD, SMAIndicator

# requests/yfinance transitively trigger urllib3's NotOpenSSLWarning at
# import time; filterwarnings must run before importing them to suppress it,
# which unavoidably breaks import-block contiguity for these two lines only.
warnings.filterwarnings('ignore')
import requests  # pylint: disable=wrong-import-position
import yfinance as yf  # pylint: disable=wrong-import-position


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────

_COMPANY_CACHE: dict[str, str] = {}  # symbol → 公司簡稱

def _load_company_cache() -> None:
    """Populate _COMPANY_CACHE from TWSE OpenAPI (all listed companies, one call)."""
    if _COMPANY_CACHE:
        return
    try:
        r = requests.get(
            'https://openapi.twse.com.tw/v1/opendata/t187ap03_L',
            headers=_TWSE_HDR, timeout=10,
        )
        for row in r.json():
            code  = str(row.get('公司代號', '')).strip()
            short = str(row.get('公司簡稱', '')).strip()
            if code and short:
                _COMPANY_CACHE[code] = short
    except Exception:
        pass


def fetch_company_name(symbol: str, is_otc: bool = False) -> str:  # pylint: disable=unused-argument
    """Return the Chinese short name for a stock code, or '' if not found.
    is_otc is unused here (the company cache isn't market-specific) but kept
    for call-site symmetry with fetch_institutional(symbol, is_otc=...)."""
    _load_company_cache()
    return _COMPANY_CACHE.get(symbol, '')


def fetch_price_data(symbol: str) -> tuple[pd.DataFrame, str]:
    end   = datetime.now()
    start = end - timedelta(days=400)  # ~285 trading days, enough for MA200
    for suffix in [".TW", ".TWO"]:
        ticker = f"{symbol}{suffix}"
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # Use unadjusted prices (match Taiwan stock app behavior)
            for col in ['Open', 'High', 'Low', 'Close']:
                if col not in df.columns and f'{col}' in df.columns:
                    pass
            return df, ticker
    raise ValueError(f"找不到股票 {symbol} 的數據")


def fetch_stock_earnings_date(ticker: str) -> date | None:
    """Best-effort: this stock's next earnings date via yfinance, or None if unavailable."""
    try:
        cal = yf.Ticker(ticker).calendar
        dates = (cal or {}).get('Earnings Date')
        if not dates:
            return None
        d = dates[0] if isinstance(dates, list) else dates
        return d if isinstance(d, date) else None
    except Exception:
        return None


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    vol   = df['Volume']

    for period in [5, 20, 60, 200, 240]:
        df[f'MA{period}'] = (
            SMAIndicator(close, window=period).sma_indicator()
            if len(df) >= period else np.nan
        )

    for period in [5, 20, 60]:
        ma_col = f'MA{period}'
        if not df[ma_col].isna().all():
            df[f'BIAS{period}'] = (close - df[ma_col]) / df[ma_col] * 100

    # Taiwan KD: RSV(9) with 1/3 EMA smoothing, initial K=D=50
    _low9  = low.rolling(9).min()
    _high9 = high.rolling(9).max()
    _rsv   = ((close - _low9) / (_high9 - _low9) * 100).fillna(50)
    _k, _d = 50.0, 50.0
    _ks, _ds = [], []
    for v in _rsv:
        _k = _k * (2/3) + v * (1/3)
        _d = _d * (2/3) + _k * (1/3)
        _ks.append(_k)
        _ds.append(_d)
    df['K'] = pd.Series(_ks, index=df.index)
    df['D'] = pd.Series(_ds, index=df.index)

    macd_obj = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df['MACD']        = macd_obj.macd()
    df['MACD_Signal'] = macd_obj.macd_signal()
    df['MACD_Hist']   = macd_obj.macd_diff()

    # SMA-based RSI (Cutler's RSI) — matches Taiwan stock app behavior
    _delta = close.diff()
    _gain  = _delta.clip(lower=0).rolling(14).mean()
    _loss  = (-_delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + _gain / _loss))
    df['Vol_MA20'] = vol.rolling(20).mean()
    df['Ret1']     = close.pct_change()     # daily return

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Condition definitions
# ─────────────────────────────────────────────────────────────────────────────

def _cross_above(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) <= b.shift(1)) & (a > b)

def _cross_below(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) >= b.shift(1)) & (a < b)


def build_conditions(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Define all candidate conditions as boolean Series indexed by df.index.
    Each condition is one candidate pattern we want to backtest.
    """
    c     = df['Close']
    k     = df['K']
    d     = df['D']
    macd  = df['MACD']
    msig  = df['MACD_Signal']
    mhist = df['MACD_Hist']
    rsi   = df['RSI']
    vol   = df['Volume']
    vma   = df['Vol_MA20']
    ret1  = df['Ret1']

    conds: dict[str, pd.Series] = {}

    # ── KD ──────────────────────────────────────────────────────────────────
    conds['KD黃金交叉']         = _cross_above(k, d)
    conds['KD死亡交叉']         = _cross_below(k, d)
    conds['KD黃金交叉(超賣區)'] = _cross_above(k, d) & (k < 25)
    conds['KD死亡交叉(超買區)'] = _cross_below(k, d) & (k > 75)
    conds['K值超賣(<20)']       = k < 20
    conds['K值超賣(<30)']       = (k >= 20) & (k < 30)
    conds['K值超買(>80)']       = k > 80
    conds['K值超買(70-80)']     = (k >= 70) & (k <= 80)
    conds['K低位回升(<40)']     = (k < 40) & (k > k.shift(1))
    conds['K高位回落(>60)']     = (k > 60) & (k < k.shift(1))

    # ── MACD ────────────────────────────────────────────────────────────────
    conds['MACD黃金交叉']         = _cross_above(macd, msig)
    conds['MACD死亡交叉']         = _cross_below(macd, msig)
    conds['MACD黃金交叉(零軸下)'] = _cross_above(macd, msig) & (macd < 0)
    conds['MACD死亡交叉(零軸上)'] = _cross_below(macd, msig) & (macd > 0)
    conds['MACD零軸上方']         = macd > 0
    conds['MACD零軸下方']         = macd < 0
    conds['MACD柱擴大(多頭)']     = (mhist > 0) & (mhist > mhist.shift(1))
    conds['MACD柱擴大(空頭)']     = (mhist < 0) & (mhist < mhist.shift(1))
    conds['MACD柱縮小(多轉弱)']   = (mhist > 0) & (mhist < mhist.shift(1))
    conds['MACD柱縮小(空轉弱)']   = (mhist < 0) & (mhist > mhist.shift(1))

    # ── RSI ─────────────────────────────────────────────────────────────────
    conds['RSI超賣(<30)']     = rsi < 30
    conds['RSI偏低(30-40)']   = (rsi >= 30) & (rsi < 40)
    conds['RSI中性(40-60)']   = (rsi >= 40) & (rsi <= 60)
    conds['RSI偏高(60-70)']   = (rsi > 60) & (rsi <= 70)
    conds['RSI超買(>70)']     = rsi > 70
    conds['RSI極度超買(>80)'] = rsi > 80
    conds['RSI低位回升']      = (rsi < 45) & (rsi > rsi.shift(1))
    conds['RSI高位回落']      = (rsi > 55) & (rsi < rsi.shift(1))

    # ── Moving Averages ──────────────────────────────────────────────────────
    for period, label in [(5, '5MA'), (20, '月線20MA'), (60, '季線60MA'), (200, '200MA')]:
        col = f'MA{period}'
        if col not in df.columns or df[col].isna().all():
            continue
        ma = df[col]
        conds[f'站上{label}']  = c > ma
        conds[f'跌破{label}']  = c < ma
        conds[f'突破{label}↑'] = _cross_above(c, ma)
        conds[f'跌破{label}↓'] = _cross_below(c, ma)

    # ── Volume ──────────────────────────────────────────────────────────────
    high_vol = vol > vma * 1.5
    conds['爆量上漲'] = high_vol & (ret1 > 0)
    conds['爆量下跌'] = high_vol & (ret1 < 0)
    conds['縮量上漲'] = (vol < vma * 0.7) & (ret1 > 0)
    conds['縮量下跌'] = (vol < vma * 0.7) & (ret1 < 0)

    # ── BIAS (乖離率) ─────────────────────────────────────────────────────────
    for _period, _hi in [(5, 5), (20, 10), (60, 15)]:
        _bcol = f'BIAS{_period}'
        if _bcol in df.columns:
            _b = df[_bcol]
            conds[f'BIAS{_period}偏高(>{_hi}%)']     = _b >  _hi
            conds[f'BIAS{_period}偏低(<-{_hi}%)']    = _b < -_hi
            conds[f'BIAS{_period}極高(>{_hi*2}%)']   = _b >  _hi * 2
            conds[f'BIAS{_period}極低(<-{_hi*2}%)']  = _b < -_hi * 2

    # ── Compound ─────────────────────────────────────────────────────────────
    conds['KD超賣+MACD金叉']  = (k < 30) & _cross_above(macd, msig)
    conds['KD超買+MACD死叉']  = (k > 70) & _cross_below(macd, msig)
    conds['跌破月線+MACD死叉'] = _cross_below(c, df.get('MA20', pd.Series(np.nan, index=df.index))) & _cross_below(macd, msig)
    conds['突破月線+MACD金叉'] = _cross_above(c, df.get('MA20', pd.Series(np.nan, index=df.index))) & _cross_above(macd, msig)

    # Cast all to bool, fill NaN → False
    return {name: s.fillna(False).astype(bool) for name, s in conds.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────────────────────────────────────

HORIZON     = 5     # trading days forward
MOVE_THRESH = 0.015 # ±1.5% counts as significant move
MIN_SAMPLES = 4     # minimum occurrences to trust a condition

# Condition → indicator group mapping (for deduplication)
COND_GROUPS: dict[str, str] = {}  # populated lazily by _group_of()

def _group_of(name: str) -> str:
    if name.startswith(('KD', 'K值', 'K低', 'K高')):
        return 'KD'
    if name.startswith('MACD'):
        return 'MACD'
    if name.startswith('RSI'):
        return 'RSI'
    if name.startswith('BIAS'):
        return 'BIAS'
    if '5MA' in name or '5ma' in name.lower():
        return 'MA5'
    if '月線' in name or '20MA' in name:
        return 'MA20'
    if '季線' in name or '60MA' in name:
        return 'MA60'
    if '200MA' in name:
        return 'MA200'
    if '量' in name:
        return 'Volume'
    return 'Compound'


def compute_outcomes(df: pd.DataFrame, move_thresh: float = MOVE_THRESH) -> pd.Series:
    """5-day forward return label: 1=up, -1=down, 0=flat, NaN=unknown."""
    close = df['Close']
    fwd   = (close.shift(-HORIZON) - close) / close
    out   = pd.Series(np.nan, index=df.index, dtype=float)
    out[fwd >  move_thresh] =  1.0
    out[fwd < -move_thresh] = -1.0
    out[(fwd >= -move_thresh) & (fwd <= move_thresh)] = 0.0
    return out


def backtest_conditions(
    conditions: dict[str, pd.Series],
    outcomes:   pd.Series,
    base:       dict,
) -> dict[str, dict]:
    """
    For each condition, compute historical up/down rates when it was active.
    Uses EXCESS edge (condition_edge − baseline_edge) so that bull-market bias
    does not inflate scores.
    Returns only conditions with >= MIN_SAMPLES occurrences.
    """
    known        = outcomes.dropna()
    base_edge    = base['up_rate'] - base['down_rate']
    results: dict[str, dict] = {}

    for name, cond in conditions.items():
        aligned = cond.reindex(known.index).fillna(False)
        hits    = known[aligned]
        n       = len(hits)
        if n < MIN_SAMPLES:
            continue
        up   = float((hits ==  1).mean())
        down = float((hits == -1).mean())
        flat = float((hits ==  0).mean())
        edge        = up - down
        excess_edge = edge - base_edge        # how much better/worse than doing nothing
        conf        = min(n / 20.0, 1.0)     # confidence saturates at 20 samples
        results[name] = {
            'count':       n,
            'up_rate':     up,
            'down_rate':   down,
            'flat_rate':   flat,
            'edge':        edge,
            'excess_edge': excess_edge,
            'confidence':  conf,
            'weight':      excess_edge * conf, # effective score contribution
            'group':       _group_of(name),
        }

    return results


def baseline_stats(outcomes: pd.Series) -> dict:
    known = outcomes.dropna()
    return {
        'count':     len(known),
        'up_rate':   float((known == 1).mean()),
        'down_rate': float((known == -1).mean()),
        'flat_rate': float((known == 0).mean()),
        'edge':      float((known == 1).mean()) - float((known == -1).mean()),
    }


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


def analyze_institutional(df_inst: pd.DataFrame) -> tuple[int, list[tuple], dict]:
    if df_inst.empty:
        return 0, [], {}

    foreign = df_inst['foreign'].tolist()
    trust   = df_inst['trust'].tolist()
    total   = df_inst['total'].tolist()

    def cum5(lst):
        return sum(lst[:5]) // 10_000

    f5, t5, tot5 = cum5(foreign), cum5(trust), cum5(total)
    f10 = sum(foreign[:10]) // 10_000 if len(foreign) >= 10 else None

    f_buy, f_sell = _consecutive(foreign)
    t_buy, t_sell = _consecutive(trust)

    score   = 0
    signals: list[tuple] = []

    if f_buy >= 3:
        score += 2
        signals.append(("外資", f"連續買超 {f_buy} 日，近5日累積 {f5:+,} 萬股"))
    elif f_buy >= 1:
        score += 1
        signals.append(("外資", f"買超 {f_buy} 日，近5日累積 {f5:+,} 萬股"))
    elif f_sell >= 3:
        score -= 2
        signals.append(("外資", f"連續賣超 {f_sell} 日，近5日累積 {f5:+,} 萬股"))
    elif f_sell >= 1:
        score -= 1
        signals.append(("外資", f"賣超 {f_sell} 日，近5日累積 {f5:+,} 萬股"))

    if t_buy >= 3:
        score += 2
        signals.append(("投信", f"連續買超 {t_buy} 日，近5日累積 {t5:+,} 萬股"))
    elif t_buy >= 1:
        score += 1
        signals.append(("投信", f"買超 {t_buy} 日，近5日累積 {t5:+,} 萬股"))
    elif t_sell >= 3:
        score -= 2
        signals.append(("投信", f"連續賣超 {t_sell} 日，近5日累積 {t5:+,} 萬股"))
    elif t_sell >= 1:
        score -= 1
        signals.append(("投信", f"賣超 {t_sell} 日，近5日累積 {t5:+,} 萬股"))

    if f_buy >= 1 and t_buy >= 1:
        score += 1
        signals.append(("同買", "外資+投信同步買超，籌碼集中"))
    elif f_sell >= 1 and t_sell >= 1:
        score -= 1
        signals.append(("同賣", "外資+投信同步賣超，賣壓沉重"))

    if f10 is not None:
        tag = "外資10日"
        if f10 > 0:
            signals.append((tag, f"近10日累積買超 {f10:,} 萬股，中期偏多"))
        else:
            signals.append((tag, f"近10日累積賣超 {f10:,} 萬股，中期偏空"))

    summary = {
        'f5': f5, 't5': t5, 'tot5': tot5,
        'f_buy': f_buy, 'f_sell': f_sell,
        't_buy': t_buy, 't_sell': t_sell,
        'f_today':   foreign[0] // 10_000 if foreign else 0,
        't_today':   trust[0]   // 10_000 if trust   else 0,
        'tot_today': total[0]   // 10_000 if total   else 0,
    }
    return score, signals, summary


# ─────────────────────────────────────────────────────────────────────────────
# Global market indicators — VIX & Buffett Indicator
# ─────────────────────────────────────────────────────────────────────────────

def fetch_vix() -> dict:
    """Fetch CBOE VIX fear index from Yahoo Finance."""
    try:
        df = yf.download("^VIX", period="5d", progress=False, auto_adjust=True)
        if df.empty:
            return {}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        val = float(df['Close'].iloc[-1])
        if val < 15:
            level, desc = "極度貪婪", "市場情緒極度樂觀，恐慌指數極低"
        elif val < 20:
            level, desc = "樂觀", "市場情緒良好，波動率偏低"
        elif val < 25:
            level, desc = "中性", "市場波動率適中"
        elif val < 30:
            level, desc = "謹慎", "市場不確定性上升，注意風險"
        elif val < 40:
            level, desc = "恐慌", "市場恐慌情緒，歷史上常現買點"
        else:
            level, desc = "極度恐慌", "市場極度恐慌，可能是逢低布局機會"
        return {'value': val, 'level': level, 'desc': desc}
    except Exception:
        return {}


def fetch_buffett_indicator() -> dict:
    """
    Taiwan Buffett Indicator = 台灣上市市值 / 台灣GDP
    GDP: DGBAS 2023 (NT$ billion); market cap fetched live from TWSE or estimated via ^TWII.
    """
    taiwan_gdp_bn = 23_599  # NT$ billion (2023, DGBAS 行政院主計總處)
    gdp_year      = 2023

    market_cap_bn = None
    note = ""

    # Attempt 1: TWSE MI_INDEX market summary
    try:
        r = requests.get(
            'https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX',
            params={'response': 'json', 'type': 'MS'},
            headers=_TWSE_HDR,
            timeout=10,
        )
        data = r.json()
        for row in data.get('data', []):
            if isinstance(row, list):
                row_text = ' '.join(str(c) for c in row)
                if '總市值' in row_text or '上市市值' in row_text:
                    for cell in reversed(row):
                        try:
                            val = float(str(cell).replace(',', ''))
                            if val > 1_000_000:  # 億元 level
                                market_cap_bn = val / 10  # 億 → billion NT$
                                note = "TWSE實時"
                                break
                        except ValueError:
                            continue
                if market_cap_bn:
                    break
    except Exception:
        pass

    # Attempt 2: estimate from ^TWII index value
    if market_cap_bn is None:
        try:
            twii = yf.download("^TWII", period="5d", progress=False, auto_adjust=True)
            if not twii.empty:
                if isinstance(twii.columns, pd.MultiIndex):
                    twii.columns = twii.columns.get_level_values(0)
                idx = float(twii['Close'].iloc[-1])
                # Empirical: TWII ≈ 18000 ↔ market cap ≈ NT$54T → coefficient ≈ 3.0 billion/point
                market_cap_bn = idx * 3.0
                note = f"估算(加權指數{idx:.0f}點)"
        except Exception:
            pass

    if market_cap_bn is None:
        return {}

    ratio = market_cap_bn / taiwan_gdp_bn * 100

    if ratio < 80:
        level, desc = "嚴重低估", "台股估值極低，長線布局機會"
    elif ratio < 120:
        level, desc = "合理", "台股估值處於合理區間"
    elif ratio < 160:
        level, desc = "略偏高", "台股估值略偏高，宜審慎操作"
    elif ratio < 200:
        level, desc = "偏高", "台股估值偏高，注意風險控管"
    else:
        level, desc = "高估", "台股估值明顯過高，系統性風險較大"

    return {
        'ratio':         ratio,
        'market_cap_bn': market_cap_bn,
        'gdp_bn':        taiwan_gdp_bn,
        'gdp_year':      gdp_year,
        'level':         level,
        'desc':          desc,
        'note':          note,
    }


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

    fwd     = (df['close'].shift(-HORIZON) - df['close']) / df['close']
    outcome = pd.Series(np.nan, index=df.index)
    outcome[fwd >  MOVE_THRESH] =  1.0
    outcome[fwd < -MOVE_THRESH] = -1.0
    outcome[(fwd >= -MOVE_THRESH) & (fwd <= MOVE_THRESH)] = 0.0
    known = df.assign(outcome=outcome).dropna(subset=['outcome'])
    if len(known) < FUTURES_MIN_SAMPLES:
        return {'bonus': 0.0, 'bucket': None, 'note': f"已知結果樣本僅 {len(known)} 天，暫不計分"}

    base_edge = float((known['outcome'] == 1).mean() - (known['outcome'] == -1).mean())
    try:
        known = known.copy()
        known['quartile'] = pd.qcut(known['val'], 4, duplicates='drop')
    except ValueError:
        return {'bonus': 0.0, 'bucket': None, 'note': "數值變化不足以分組，暫不計分"}

    bucket = None
    for interval in known['quartile'].cat.categories:
        if today_val in interval or today_val == interval.right:
            bucket = interval
            break
    if bucket is None:
        # today's value sits outside the historical range entirely — extrapolating
        # off the last known bucket would overstate confidence, so skip scoring.
        return {'bonus': 0.0, 'bucket': None, 'note': "今日數值超出歷史分組範圍，暫不計分"}

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


def _dated_series(raw: dict[str, float]) -> pd.Series:
    s = pd.Series(raw)
    s.index = pd.to_datetime(s.index, format="%Y/%m/%d")
    return s.sort_index()


def fetch_futures_sentiment() -> dict:
    """
    Market-wide macro overlay from 外資 TXF 未平倉部位: net position (direction)
    and gross short open interest (hedging/bearish pressure), each scored
    independently via _quartile_excess_bonus() and summed into one bonus.
    """
    try:
        twii = yf.download("^TWII", period="200d", progress=False, auto_adjust=True)
        if twii.empty:
            return {}
        if isinstance(twii.columns, pd.MultiIndex):
            twii.columns = twii.columns.get_level_values(0)
        close = twii['Close']
    except Exception:
        return {}

    latest_date = close.index[-1]
    date_str    = latest_date.strftime("%Y/%m/%d")

    cache = _load_futures_cache()
    entry = cache.get(date_str)
    if not isinstance(entry, dict):  # missing, or legacy plain-float cache format
        entry = None
    if entry is None:
        net_val, short_val = fetch_futures_net_position(date_str)
        if net_val is not None:
            entry = {'net': net_val, 'short': short_val}
            cache[date_str] = entry
            _save_futures_cache(cache)

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

    # Tolerate legacy cache entries that were a plain float (net-only, no short).
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

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze(
    df:       pd.DataFrame,
    symbol:   str,
    df_inst:  pd.DataFrame,
    vix:      dict | None = None,
    buffett:  dict | None = None,
    regime:   dict | None = None,
    futures:  dict | None = None,
) -> dict:
    if len(df) < 30:
        raise ValueError("歷史數據不足")

    # ── Build conditions & run backtest ─────────────────────────────────────
    # Per-regime thresholds: bull → hard to reduce; bear → easy to reduce
    _regime_params: dict[str, dict] = {
        '強多頭': {"add": 0.25, "reduce": -0.40, "strong": -0.80, "move": 0.020},
        '多頭':   {"add": 0.25, "reduce": -0.25, "strong": -0.60, "move": 0.015},
        '中性':   {"add": 0.25, "reduce": -0.25, "strong": -0.60, "move": 0.015},
        '空頭':   {"add": 0.25, "reduce": -0.25, "strong": -0.60, "move": 0.015},
    }
    _rp = _regime_params.get((regime or {}).get('regime', '中性'), _regime_params['中性'])

    conditions = build_conditions(df)
    outcomes   = compute_outcomes(df, move_thresh=_rp['move'])
    base       = baseline_stats(outcomes)
    bt         = backtest_conditions(conditions, outcomes, base)

    # ── Which conditions are active today (last row) ─────────────────────────
    today_flags: dict[str, bool] = {
        name: bool(cond.iloc[-1]) for name, cond in conditions.items()
    }

    # ── Score: one best signal per indicator group (prevents double-counting) ─
    # For each group, pick the active condition with the highest |weight|.
    # Sum those per-group bests → tech_score.
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

    # ── Institutional signals ────────────────────────────────────────────────
    inst_score, inst_signals, inst_summary = analyze_institutional(df_inst)
    # Normalize inst_score (-5..+5) to similar scale as tech_score
    inst_normalized = inst_score * 0.12

    # ── Market regime adjustment ─────────────────────────────────────────────
    # TWII vs MA200 adds a bull/bear tilt so the strategy doesn't fight the tape
    regime_bonus = (regime or {}).get('bonus', 0.0)

    # ── Futures positioning adjustment ───────────────────────────────────────
    # 外資 TXF 淨部位 + 空方口數，each bucketed against its own historical excess
    # edge; 0 until enough cached samples exist (see fetch_futures_sentiment).
    futures_bonus = (futures or {}).get('bonus', 0.0) + (futures or {}).get('short_bonus', 0.0)

    combined = tech_score + inst_normalized + regime_bonus + futures_bonus

    # ── Recommendation ───────────────────────────────────────────────────────
    if combined >= 0.6:
        recommendation = "強力加碼"
    elif combined >= _rp['add']:
        recommendation = "加碼"
    elif combined <= _rp['strong']:
        recommendation = "強力減碼"
    elif combined <= _rp['reduce']:
        recommendation = "減碼"
    else:
        recommendation = "持平"

    # ── Snapshot values ──────────────────────────────────────────────────────
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
        'symbol':         symbol,
        'price':          price,
        'price_chg':      price_chg,
        'price_pct':      price_pct,
        'K':   _s('K'),   'D':          _s('D'),
        'RSI': _s('RSI'), 'MACD':       _s('MACD'),
        'MACD_Signal':    _s('MACD_Signal'),
        'MA5':  _s('MA5'),  'MA20': _s('MA20'),
        'MA60': _s('MA60'), 'MA200': _s('MA200'),
        'BIAS5': _s('BIAS5'), 'BIAS20': _s('BIAS20'), 'BIAS60': _s('BIAS60'),
        'vix':            vix or {},
        'buffett':        buffett or {},
        'regime':         regime or {},
        'regime_bonus':   regime_bonus,
        'futures':        futures or {},
        'futures_bonus':  futures_bonus,
        'base':           base,
        'active_bt':      active_bt,
        'group_best':     group_best,
        'all_bt':         bt,
        'tech_score':     tech_score,
        'inst_score':     inst_score,
        'combined':       combined,
        'recommendation': recommendation,
        'inst_signals':   inst_signals,
        'inst_summary':   inst_summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report formatting
# ─────────────────────────────────────────────────────────────────────────────

_REC_ICON = {"強力加碼": "🚀", "加碼": "↑", "持平": "→", "減碼": "↓", "強力減碼": "⚠"}
_REC_RANK = {"強力加碼": 5, "加碼": 4, "持平": 3, "減碼": 2, "強力減碼": 1}


def _direction_label(excess: float, n: int) -> str:
    """Label a condition based on its excess edge vs baseline."""
    # Require minimum statistical confidence: penalise small samples
    effective = excess * min(n / 10.0, 1.0)
    if effective >= +0.20:
        return "強多 ↑↑"
    if effective >= +0.10:
        return "多  ↑ "
    if effective <= -0.20:
        return "強空 ↓↓"
    if effective <= -0.10:
        return "空  ↓ "
    return "中性 → "


def fmt_report(r: dict) -> str:
    W = 70
    lines: list[str] = []

    def _v(x, fmt='.2f'):
        return f"{x:{fmt}}" if x is not None else "N/A"

    lines.append("=" * W)
    name_str = f"  {r['company_name']}" if r.get('company_name') else ""
    lines.append(f"  股票代號: {r['symbol']}{name_str}")
    chg = f"  ({r['price_chg']:+.2f}, {r['price_pct']:+.2f}%)" if r['price_chg'] else ""
    lines.append(f"  最新收盤: {_v(r['price'])}{chg}")
    ed = r.get('earnings_date')
    if ed and date.today() <= ed <= date.today() + timedelta(days=7):
        lines.append(f"  ⚠ 財報將於 {ed.strftime('%-m/%-d')} 公布，財報前後波動放大，建議觀望")

    # ── Indicator snapshot ────────────────────────────────────────────────────
    lines.append("-" * W)
    lines.append("  技術指標快照:")
    lines.append(f"    KD:   K={_v(r['K'],'.1f')}  D={_v(r['D'],'.1f')}")
    lines.append(f"    RSI:  {_v(r['RSI'],'.1f')}")
    lines.append(f"    MACD: {_v(r['MACD'],'.3f')}  Signal: {_v(r['MACD_Signal'],'.3f')}")
    ma_parts = []
    for key, lbl in [('MA5','5MA'),('MA20','20MA'),('MA60','60MA'),('MA200','200MA')]:
        if r[key] is not None:
            ma_parts.append(f"{lbl}={_v(r[key])}")
    lines.append(f"    均線: {'  '.join(ma_parts)}")
    bias_parts = []
    for key, lbl in [('BIAS5','5日'),('BIAS20','20日'),('BIAS60','60日')]:
        v = r.get(key)
        if v is not None:
            bias_parts.append(f"{lbl}={v:+.1f}%")
    if bias_parts:
        lines.append(f"    乖離率: {'  '.join(bias_parts)}")

    # ── Global market indicators ──────────────────────────────────────────────
    vix     = r.get('vix',     {})
    buffett = r.get('buffett', {})
    regime  = r.get('regime',  {})
    futures = r.get('futures', {})
    if vix or buffett or regime or futures:
        lines.append("-" * W)
        lines.append("  總體市場指標:")
        if regime:
            twii_str = f"TWII {regime['twii']:.0f} / MA200 {regime['ma200']:.0f}" if regime.get('twii') else ""
            bonus_str = f"  分數調整 {regime['bonus']:+.2f}" if regime.get('bonus') is not None else ""
            lines.append(f"    台股趨勢: {twii_str}  [{regime['regime']}]{bonus_str}")
        if vix:
            lines.append(
                f"    恐慌指數(VIX): {vix['value']:.1f}  "
                f"[{vix['level']}]  {vix['desc']}"
            )
        if buffett:
            cap_t  = buffett['market_cap_bn'] / 1_000
            gdp_t  = buffett['gdp_bn']        / 1_000
            lines.append(
                f"    巴菲特指標:  {buffett['ratio']:.1f}%"
                f"  (市值{cap_t:.1f}兆 / GDP{gdp_t:.1f}兆 {buffett['gdp_year']})"
                f"  [{buffett['level']}]"
            )
            note = buffett.get('note', '')
            suffix = f"  ({note})" if note else ""
            lines.append(f"    {buffett['desc']}{suffix}")
        if futures:
            net = futures.get('net_pos')
            net_str = f"{net:+,.0f}口" if net is not None else "N/A"
            bonus_str = f"  分數調整 {futures['bonus']:+.2f}" if futures.get('bonus') else ""
            lines.append(
                f"    外資期貨淨部位(TXF): {net_str}{bonus_str}  ({futures.get('note', '')})"
            )
            short = futures.get('short_oi')
            if short is not None:
                short_str = f"{short:+,.0f}口"
                short_bonus_str = f"  分數調整 {futures['short_bonus']:+.2f}" if futures.get('short_bonus') else ""
                lines.append(
                    f"    外資期貨空方口數(TXF): {short_str}{short_bonus_str}  ({futures.get('short_note', '')})"
                )

    # ── Institutional snapshot ────────────────────────────────────────────────
    inst = r.get('inst_summary', {})
    if inst:
        lines.append("-" * W)
        lines.append("  三大法人 (近5日累積，萬股):")

        def streak(b, s):
            return f"連買{b}日" if b else (f"連賣{s}日" if s else "持平")

        lines.append(
            f"    外資: 今日{inst['f_today']:+,}萬股  近5日{inst['f5']:+,}萬股"
            f"  [{streak(inst['f_buy'],inst['f_sell'])}]"
        )
        lines.append(
            f"    投信: 今日{inst['t_today']:+,}萬股  近5日{inst['t5']:+,}萬股"
            f"  [{streak(inst['t_buy'],inst['t_sell'])}]"
        )
        lines.append(
            f"    三大合計: 今日{inst['tot_today']:+,}萬股  近5日{inst['tot5']:+,}萬股"
        )

    # ── Backtested technical signals table ───────────────────────────────────
    base = r['base']
    lines.append("-" * W)
    lines.append(
        f"  歷史回測 ({base['count']} 個交易日 | 目標: 5日後漲跌>{MOVE_THRESH*100:.0f}%"
        f" | 基準: 上漲{base['up_rate']*100:.0f}% 下跌{base['down_rate']*100:.0f}%)"
    )
    lines.append("")

    # Determine which conditions actually contributed to the score (one per group)
    scoring_names = {s['name'] for s in r['group_best'].values()}

    col_w = [27, 5, 5, 7, 7, 8, 10]
    header = (
        f"  {'條件':<{col_w[0]}} {'次數':>{col_w[1]}} {'觸發':>{col_w[2]}}"
        f" {'上漲率':>{col_w[3]}} {'下跌率':>{col_w[4]}} {'超額邊際':>{col_w[5]}} {'結論':<{col_w[6]}}"
    )
    lines.append(header)
    lines.append("  " + "─" * (W - 2))

    all_bt       = r['all_bt']
    active_names = {s['name'] for s in r['active_bt']}

    def sort_key(item):
        name, stats = item
        # Active conditions first (sorted by |excess_edge|), then inactive
        return (0 if name in active_names else 1, -abs(stats['excess_edge']))

    for name, stats in sorted(all_bt.items(), key=sort_key):
        is_active  = name in active_names
        is_scoring = name in scoring_names
        if is_active and is_scoring:
            marker = " ★"   # active + scored
        elif is_active:
            marker = " ✓"   # active but group already covered by a better condition
        else:
            marker = "  "
        up_pct  = stats['up_rate']  * 100
        down_pct = stats['down_rate'] * 100
        excess  = stats['excess_edge']
        direction = _direction_label(excess, stats['count'])
        lines.append(
            f"  {name:<{col_w[0]}} {stats['count']:>{col_w[1]}} {marker:>{col_w[2]}}"
            f" {up_pct:>{col_w[3]}.0f}% {down_pct:>{col_w[4]}.0f}%"
            f" {excess:>+{col_w[5]}.3f} {direction:<{col_w[6]}}"
        )

    lines.append(f"  (★=計入評分 ✓=觸發但同組已有更強信號  超額邊際=條件邊際−基準{base['edge']:+.2f})")

    # ── Institutional signals ─────────────────────────────────────────────────
    if r['inst_signals']:
        lines.append("")
        lines.append("  [籌碼面]")
        for tag, desc in r['inst_signals']:
            lines.append(f"    [{tag}] {desc}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    lines.append("-" * W)
    reg_bonus = r.get('regime_bonus', 0.0)
    reg_name  = r.get('regime', {}).get('regime', '')
    reg_str   = f"  市場{reg_name}({reg_bonus:+.2f})" if reg_bonus != 0 else ""
    fut_bonus = r.get('futures_bonus', 0.0)
    fut_str   = f"  期貨籌碼({fut_bonus:+.2f})" if fut_bonus != 0 else ""
    lines.append(
        f"  技術分數: {r['tech_score']:+.3f}  "
        f"籌碼分數: {r['inst_score']:+d}(×0.12={r['inst_score']*0.12:+.2f})"
        f"{reg_str}{fut_str}  合計: {r['combined']:+.3f}"
    )
    rec  = r['recommendation']
    icon = _REC_ICON.get(rec, "")
    lines.append(f"  ▶ 未來一週建議: 【{rec}】 {icon}")
    lines.append("=" * W)
    return "\n".join(lines)


def fmt_summary_table(results: list[dict]) -> str:
    """Multi-stock comparison table, sorted 強力加碼 → 加碼 → 持平 → 減碼 → 強力減碼
    (ties broken by combined score, descending)."""
    W = 70
    lines: list[str] = []
    lines.append("=" * W)
    lines.append("  綜合比較表（依建議排序）")
    lines.append("-" * W)
    col_w = [6, 8, 9, 8, 8, 8]
    lines.append(
        f"  {'股票':<{col_w[0]}}{'公司名':<{col_w[1]}}{'收盤價':>{col_w[2]}}"
        f"{'技術分':>{col_w[3]}}{'籌碼分':>{col_w[4]}}{'合計':>{col_w[5]}}  建議"
    )
    ordered = sorted(results, key=lambda r: (-_REC_RANK.get(r['recommendation'], 0), -r['combined']))
    for r in ordered:
        rec = r['recommendation']
        icon = _REC_ICON.get(rec, "")
        lines.append(
            f"  {r['symbol']:<{col_w[0]}}{(r.get('company_name') or ''):<{col_w[1]}}"
            f"{r['price']:>{col_w[2]}.2f}{r['tech_score']:>+{col_w[3]}.3f}"
            f"{r['inst_score']*0.12:>+{col_w[4]}.2f}{r['combined']:>+{col_w[5]}.3f}"
            f"  【{rec}】{icon}"
        )
    lines.append("=" * W)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("用法: python3 tech_analysis.py <代號1> [代號2] ...")
        print("範例: python3 tech_analysis.py 2330 2317 0050")
        sys.exit(1)

    print("正在抓取總體市場指標 (VIX / 巴菲特指標 / 台股趨勢 / 外資期貨淨部位)...")
    vix     = fetch_vix()
    buffett = fetch_buffett_indicator()
    regime  = fetch_market_regime()
    futures = fetch_futures_sentiment()
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

            result = analyze(df, sym, df_inst, vix=vix, buffett=buffett, regime=regime, futures=futures)
            result['company_name'] = company_name
            result['earnings_date'] = fetch_stock_earnings_date(ticker)
            print(fmt_report(result))
            results.append(result)
        except Exception as e:
            print(f"[錯誤] {sym}: {e}")
            traceback.print_exc()

    if len(results) > 1:
        print()
        print(fmt_summary_table(results))


if __name__ == "__main__":
    main()
