"""
Price/company/earnings data fetching and indicator calculation for
tech_analysis.py. Pulls OHLCV + company names from Yahoo Finance/TWSE and
derives MA/BIAS/KD/MACD/RSI/volume columns.
"""
from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from ta.trend import MACD, SMAIndicator

# requests/yfinance transitively trigger urllib3's NotOpenSSLWarning at
# import time; filterwarnings must run before importing them to suppress it,
# which unavoidably breaks import-block contiguity for these two lines only.
warnings.filterwarnings('ignore')
import requests  # pylint: disable=wrong-import-position
import yfinance as yf  # pylint: disable=wrong-import-position

_TWSE_HDR = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.twse.com.tw/'}

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
    """400 days of OHLCV for `symbol`, trying .TW then .TWO. Raises if neither has data."""
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


def _calc_kd(close: pd.Series, high: pd.Series, low: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Taiwan KD: RSV(9) with 1/3 EMA smoothing, initial K=D=50."""
    low9  = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv   = ((close - low9) / (high9 - low9) * 100).fillna(50)
    k, d = 50.0, 50.0
    ks, ds = [], []
    for v in rsv:
        k = k * (2/3) + v * (1/3)
        d = d * (2/3) + k * (1/3)
        ks.append(k)
        ds.append(d)
    return pd.Series(ks, index=close.index), pd.Series(ds, index=close.index)


def _calc_rsi(close: pd.Series) -> pd.Series:
    """SMA-based RSI (Cutler's RSI) — matches Taiwan stock app behavior."""
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    return 100 - (100 / (1 + gain / loss))


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA/BIAS/KD/MACD/RSI/volume columns to `df` in place (and return it)."""
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

    df['K'], df['D'] = _calc_kd(close, high, low)

    macd_obj = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df['MACD']        = macd_obj.macd()
    df['MACD_Signal'] = macd_obj.macd_signal()
    df['MACD_Hist']   = macd_obj.macd_diff()

    df['RSI']      = _calc_rsi(close)
    df['Vol_MA20'] = vol.rolling(20).mean()
    df['Ret1']     = close.pct_change()     # daily return

    return df
