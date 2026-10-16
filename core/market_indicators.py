"""
Global market sentiment/valuation indicators for tech_analysis.py's regime
overlay: CBOE VIX and the Taiwan Buffett Indicator (market cap / GDP).
"""
from __future__ import annotations

import warnings

import pandas as pd

# requests/yfinance transitively trigger urllib3's NotOpenSSLWarning at
# import time; filterwarnings must run before importing them to suppress it,
# which unavoidably breaks import-block contiguity for these two lines only.
warnings.filterwarnings('ignore')
import requests  # pylint: disable=wrong-import-position
import yfinance as yf  # pylint: disable=wrong-import-position

_TWSE_HDR = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.twse.com.tw/'}


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


def _find_market_cap_cell(row: list) -> float | None:
    """Scan a TWSE MI_INDEX row (reversed) for the first plausible 總市值 cell value."""
    for cell in reversed(row):
        try:
            val = float(str(cell).replace(',', ''))
        except ValueError:
            continue
        if val > 1_000_000:  # 億元 level
            return val / 10  # 億 → billion NT$
    return None


def _fetch_twse_market_cap() -> tuple[float | None, str]:
    """Attempt 1: TWSE MI_INDEX market summary. Returns (market_cap_bn, note)."""
    try:
        r = requests.get(
            'https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX',
            params={'response': 'json', 'type': 'MS'},
            headers=_TWSE_HDR,
            timeout=10,
        )
        for row in r.json().get('data', []):
            if not isinstance(row, list):
                continue
            row_text = ' '.join(str(c) for c in row)
            if '總市值' not in row_text and '上市市值' not in row_text:
                continue
            cap = _find_market_cap_cell(row)
            if cap:
                return cap, "TWSE實時"
    except Exception:
        pass
    return None, ""


def _estimate_market_cap_from_twii() -> tuple[float | None, str]:
    """Attempt 2 (fallback): estimate market cap from the ^TWII index value."""
    try:
        twii = yf.download("^TWII", period="5d", progress=False, auto_adjust=True)
        if twii.empty:
            return None, ""
        if isinstance(twii.columns, pd.MultiIndex):
            twii.columns = twii.columns.get_level_values(0)
        idx = float(twii['Close'].iloc[-1])
        # Empirical: TWII ≈ 18000 ↔ market cap ≈ NT$54T → coefficient ≈ 3.0 billion/point
        return idx * 3.0, f"估算(加權指數{idx:.0f}點)"
    except Exception:
        return None, ""


def _classify_buffett_ratio(ratio: float) -> tuple[str, str]:
    if ratio < 80:
        return "嚴重低估", "台股估值極低，長線布局機會"
    if ratio < 120:
        return "合理", "台股估值處於合理區間"
    if ratio < 160:
        return "略偏高", "台股估值略偏高，宜審慎操作"
    if ratio < 200:
        return "偏高", "台股估值偏高，注意風險控管"
    return "高估", "台股估值明顯過高，系統性風險較大"


def fetch_buffett_indicator() -> dict:
    """
    Taiwan Buffett Indicator = 台灣上市市值 / 台灣GDP
    GDP: DGBAS 2023 (NT$ billion); market cap fetched live from TWSE or estimated via ^TWII.
    """
    taiwan_gdp_bn = 23_599  # NT$ billion (2023, DGBAS 行政院主計總處)
    gdp_year      = 2023

    market_cap_bn, note = _fetch_twse_market_cap()
    if market_cap_bn is None:
        market_cap_bn, note = _estimate_market_cap_from_twii()
    if market_cap_bn is None:
        return {}

    ratio = market_cap_bn / taiwan_gdp_bn * 100
    level, desc = _classify_buffett_ratio(ratio)

    return {
        'ratio':         ratio,
        'market_cap_bn': market_cap_bn,
        'gdp_bn':        taiwan_gdp_bn,
        'gdp_year':      gdp_year,
        'level':         level,
        'desc':          desc,
        'note':          note,
    }
