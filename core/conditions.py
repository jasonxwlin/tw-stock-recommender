"""
Technical condition definitions for tech_analysis.py's backtest engine.
Each function returns boolean pd.Series indexed by df.index — one candidate
pattern per key, later evaluated for historical hit rate by backtest_conditions().
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cross_above(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) <= b.shift(1)) & (a > b)

def _cross_below(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) >= b.shift(1)) & (a < b)


def _kd_conditions(k: pd.Series, d: pd.Series) -> dict[str, pd.Series]:
    return {
        'KD黃金交叉':         _cross_above(k, d),
        'KD死亡交叉':         _cross_below(k, d),
        'KD黃金交叉(超賣區)': _cross_above(k, d) & (k < 25),
        'KD死亡交叉(超買區)': _cross_below(k, d) & (k > 75),
        'K值超賣(<20)':       k < 20,
        'K值超賣(<30)':       (k >= 20) & (k < 30),
        'K值超買(>80)':       k > 80,
        'K值超買(70-80)':     (k >= 70) & (k <= 80),
        'K低位回升(<40)':     (k < 40) & (k > k.shift(1)),
        'K高位回落(>60)':     (k > 60) & (k < k.shift(1)),
    }


def _macd_conditions(macd: pd.Series, msig: pd.Series, mhist: pd.Series) -> dict[str, pd.Series]:
    return {
        'MACD黃金交叉':         _cross_above(macd, msig),
        'MACD死亡交叉':         _cross_below(macd, msig),
        'MACD黃金交叉(零軸下)': _cross_above(macd, msig) & (macd < 0),
        'MACD死亡交叉(零軸上)': _cross_below(macd, msig) & (macd > 0),
        'MACD零軸上方':         macd > 0,
        'MACD零軸下方':         macd < 0,
        'MACD柱擴大(多頭)':     (mhist > 0) & (mhist > mhist.shift(1)),
        'MACD柱擴大(空頭)':     (mhist < 0) & (mhist < mhist.shift(1)),
        'MACD柱縮小(多轉弱)':   (mhist > 0) & (mhist < mhist.shift(1)),
        'MACD柱縮小(空轉弱)':   (mhist < 0) & (mhist > mhist.shift(1)),
    }


def _rsi_conditions(rsi: pd.Series) -> dict[str, pd.Series]:
    return {
        'RSI超賣(<30)':     rsi < 30,
        'RSI偏低(30-40)':   (rsi >= 30) & (rsi < 40),
        'RSI中性(40-60)':   (rsi >= 40) & (rsi <= 60),
        'RSI偏高(60-70)':   (rsi > 60) & (rsi <= 70),
        'RSI超買(>70)':     rsi > 70,
        'RSI極度超買(>80)': rsi > 80,
        'RSI低位回升':      (rsi < 45) & (rsi > rsi.shift(1)),
        'RSI高位回落':      (rsi > 55) & (rsi < rsi.shift(1)),
    }


def _ma_conditions(df: pd.DataFrame, c: pd.Series) -> dict[str, pd.Series]:
    conds: dict[str, pd.Series] = {}
    for period, label in [(5, '5MA'), (20, '月線20MA'), (60, '季線60MA'), (200, '200MA')]:
        col = f'MA{period}'
        if col not in df.columns or df[col].isna().all():
            continue
        ma = df[col]
        conds[f'站上{label}']  = c > ma
        conds[f'跌破{label}']  = c < ma
        conds[f'突破{label}↑'] = _cross_above(c, ma)
        conds[f'跌破{label}↓'] = _cross_below(c, ma)
    return conds


def _volume_conditions(vol: pd.Series, vma: pd.Series, ret1: pd.Series) -> dict[str, pd.Series]:
    high_vol = vol > vma * 1.5
    return {
        '爆量上漲': high_vol & (ret1 > 0),
        '爆量下跌': high_vol & (ret1 < 0),
        '縮量上漲': (vol < vma * 0.7) & (ret1 > 0),
        '縮量下跌': (vol < vma * 0.7) & (ret1 < 0),
    }


def _bias_conditions(df: pd.DataFrame) -> dict[str, pd.Series]:
    conds: dict[str, pd.Series] = {}
    for period, hi in [(5, 5), (20, 10), (60, 15)]:
        bcol = f'BIAS{period}'
        if bcol in df.columns:
            b = df[bcol]
            conds[f'BIAS{period}偏高(>{hi}%)']    = b >  hi
            conds[f'BIAS{period}偏低(<-{hi}%)']   = b < -hi
            conds[f'BIAS{period}極高(>{hi*2}%)']  = b >  hi * 2
            conds[f'BIAS{period}極低(<-{hi*2}%)'] = b < -hi * 2
    return conds


def _compound_conditions(
    df: pd.DataFrame, c: pd.Series, k: pd.Series, macd: pd.Series, msig: pd.Series,
) -> dict[str, pd.Series]:
    ma20 = df.get('MA20', pd.Series(np.nan, index=df.index))
    return {
        'KD超賣+MACD金叉':  (k < 30) & _cross_above(macd, msig),
        'KD超買+MACD死叉':  (k > 70) & _cross_below(macd, msig),
        '跌破月線+MACD死叉': _cross_below(c, ma20) & _cross_below(macd, msig),
        '突破月線+MACD金叉': _cross_above(c, ma20) & _cross_above(macd, msig),
    }


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

    conds: dict[str, pd.Series] = {
        **_kd_conditions(k, d),
        **_macd_conditions(macd, msig, mhist),
        **_rsi_conditions(rsi),
        **_ma_conditions(df, c),
        **_volume_conditions(vol, vma, ret1),
        **_bias_conditions(df),
        **_compound_conditions(df, c, k, macd, msig),
    }

    # Cast all to bool, fill NaN → False
    return {name: s.fillna(False).astype(bool) for name, s in conds.items()}
