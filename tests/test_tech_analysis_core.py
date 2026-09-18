#!/usr/bin/env python3
"""Self-check for tech_analysis.py's pure indicator/condition/scoring/report
logic (no network). Covers calc_indicators, build_conditions, the full
analyze() pipeline run end-to-end on synthetic OHLCV, fmt_report, and the
small helpers analyze_institutional/_direction_label/_parse_num."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import tech_analysis as ta


def _make_ohlcv(n: int = 260, seed: int = 42) -> pd.DataFrame:
    """Deterministic, realistic-shaped synthetic OHLCV (seeded random walk)."""
    rng    = np.random.default_rng(seed)
    rets   = rng.normal(0.0005, 0.015, n)
    close  = 100 * np.cumprod(1 + rets)
    high   = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low    = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    openp  = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(1_000, 100_000, n).astype(float)
    idx    = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame(
        {'Open': openp, 'High': high, 'Low': low, 'Close': close, 'Volume': volume},
        index=idx,
    )


def _make_df_inst(pattern: list[int] | None = None) -> pd.DataFrame:
    pattern = pattern or [1000, 2000, 1500, -500, 3000, 800, 1200, -300, -900, -1100]
    n = len(pattern)
    return pd.DataFrame({
        'date':    [f'2026{(i % 12) + 1:02d}{(i % 27) + 1:02d}' for i in range(n)],
        'foreign': pattern,
        'trust':   [v // 2 for v in pattern],
        'dealer':  [0] * n,
        'total':   [v + v // 2 for v in pattern],
    })


# ── calc_indicators ──────────────────────────────────────────────────────────

def test_calc_indicators_adds_expected_columns_within_valid_ranges():
    df = ta.calc_indicators(_make_ohlcv())
    for col in ('MA5', 'MA20', 'MA60', 'MA200', 'K', 'D', 'MACD', 'MACD_Signal',
                'MACD_Hist', 'RSI', 'Vol_MA20', 'Ret1', 'BIAS5', 'BIAS20', 'BIAS60'):
        assert col in df.columns, col
    assert df['K'].between(0, 100).all()
    assert df['D'].between(0, 100).all()
    assert df['RSI'].dropna().between(0, 100).all()


def test_calc_indicators_skips_long_mas_on_short_history():
    df = ta.calc_indicators(_make_ohlcv(n=50))
    assert df['MA200'].isna().all()
    assert not df['MA20'].isna().all()


# ── build_conditions ─────────────────────────────────────────────────────────

def test_build_conditions_returns_bool_series_covering_full_index():
    df    = ta.calc_indicators(_make_ohlcv())
    conds = ta.build_conditions(df)
    assert len(conds) > 30
    for name, s in conds.items():
        assert s.dtype == bool, name
        assert not s.isna().any(), name
        assert len(s) == len(df), name


def test_build_conditions_ma_conditions_absent_without_enough_history():
    # n=50 has no MA200 column data → the corresponding 站上/跌破/突破/跌破 200MA
    # conditions must simply be skipped, not crash.
    df    = ta.calc_indicators(_make_ohlcv(n=50))
    conds = ta.build_conditions(df)
    assert '站上200MA' not in conds
    assert '站上5MA' in conds


# ── analyze_institutional / _consecutive interplay ───────────────────────────

def test_analyze_institutional_empty_df_returns_zeroed_result():
    score, signals, summary = ta.analyze_institutional(pd.DataFrame())
    assert score == 0
    assert signals == []
    assert summary == {}


def test_analyze_institutional_buy_streak_and_co_buy_signal():
    df_inst = _make_df_inst([1000, 2000, 1500])  # 3-day foreign buy streak
    score, signals, summary = ta.analyze_institutional(df_inst)
    tags = [t for t, _ in signals]
    assert "外資" in tags
    assert "投信" in tags
    assert "同買" in tags  # foreign and trust both buying
    assert score > 0
    assert summary['f_buy'] == 3


def test_analyze_institutional_sell_streak_and_co_sell_signal():
    df_inst = _make_df_inst([-1000, -2000, -1500])
    score, signals, summary = ta.analyze_institutional(df_inst)
    tags = [t for t, _ in signals]
    assert "同賣" in tags
    assert score < 0
    assert summary['f_sell'] == 3


def test_analyze_institutional_ten_day_cumulative_present_with_enough_rows():
    df_inst = _make_df_inst()  # 10 rows
    _, signals, summary = ta.analyze_institutional(df_inst)
    assert "外資10日" in [t for t, _ in signals]
    assert summary['tot5'] is not None


def _df_inst_isolated(foreign: list[int], trust: list[int]) -> pd.DataFrame:
    """Independent foreign/trust patterns (unlike _make_df_inst's foreign//2 coupling),
    for exercising each single-day/co-signal branch of analyze_institutional in isolation."""
    n = len(foreign)
    return pd.DataFrame({
        'date':    [f'2026{(i % 12) + 1:02d}{(i % 27) + 1:02d}' for i in range(n)],
        'foreign': foreign,
        'trust':   trust,
        'dealer':  [0] * n,
        'total':   [f + t for f, t in zip(foreign, trust)],
    })


def test_analyze_institutional_single_day_foreign_buy():
    df_inst = _df_inst_isolated([1000, -1, 0], [0, 0, 0])
    _, signals, _ = ta.analyze_institutional(df_inst)
    assert "買超 1 日" in dict(signals)["外資"]


def test_analyze_institutional_single_day_foreign_sell():
    df_inst = _df_inst_isolated([-1000, 1, 0], [0, 0, 0])
    _, signals, _ = ta.analyze_institutional(df_inst)
    assert "賣超 1 日" in dict(signals)["外資"]


def test_analyze_institutional_single_day_trust_buy():
    df_inst = _df_inst_isolated([0, 0, 0], [500, -1, 0])
    _, signals, _ = ta.analyze_institutional(df_inst)
    assert "買超 1 日" in dict(signals)["投信"]


def test_analyze_institutional_single_day_trust_sell():
    df_inst = _df_inst_isolated([0, 0, 0], [-500, 1, 0])
    _, signals, _ = ta.analyze_institutional(df_inst)
    assert "賣超 1 日" in dict(signals)["投信"]


def test_analyze_institutional_ten_day_positive_cumulative_message():
    df_inst = _df_inst_isolated([5000] * 10, [0] * 10)
    _, signals, summary = ta.analyze_institutional(df_inst)
    assert summary['f_buy'] == 10
    assert "近10日累積買超" in dict(signals)["外資10日"]


# ── _direction_label ──────────────────────────────────────────────────────────

def test_direction_label_covers_all_five_buckets():
    assert ta._direction_label(0.25, 10) == "強多 ↑↑"
    assert ta._direction_label(0.12, 10) == "多  ↑ "
    assert ta._direction_label(0.00, 10) == "中性 → "
    assert ta._direction_label(-0.12, 10) == "空  ↓ "
    assert ta._direction_label(-0.25, 10) == "強空 ↓↓"


def test_direction_label_small_sample_dampens_toward_neutral():
    # n=1 → confidence multiplier min(1/10, 1)=0.1, so even a big excess edge
    # is dampened to "中性" rather than "強多".
    assert ta._direction_label(0.90, 1) == "中性 → "


# ── _parse_num ────────────────────────────────────────────────────────────────

def test_parse_num_strips_commas_and_spaces():
    assert ta._parse_num("1,234") == 1234
    assert ta._parse_num(" 500 ") == 500
    assert ta._parse_num("-42") == -42


def test_parse_num_returns_zero_on_garbage():
    assert ta._parse_num("N/A") == 0
    assert ta._parse_num(None) == 0


# ── analyze() full integration (real build_conditions/backtest_conditions) ──

def test_analyze_full_integration_returns_expected_shape():
    df      = ta.calc_indicators(_make_ohlcv())
    df_inst = _make_df_inst()
    regime  = {'regime': '多頭', 'bonus': 0.15, 'twii': 20000.0, 'ma200': 18000.0, 'ratio': 1.11}
    vix     = {'value': 18.0, 'level': '樂觀', 'desc': 'x'}
    buffett = {'ratio': 150.0, 'level': '略偏高', 'desc': 'x',
               'market_cap_bn': 50_000.0, 'gdp_bn': 23_599.0, 'gdp_year': 2023, 'note': ''}
    futures = {'bonus': 0.0, 'short_bonus': 0.0}

    result = ta.analyze(df, '2330', df_inst, vix=vix, buffett=buffett, regime=regime, futures=futures)

    assert result['symbol'] == '2330'
    assert result['recommendation'] in ta._REC_RANK
    assert isinstance(result['combined'], float)
    assert isinstance(result['tech_score'], float)
    assert 'active_bt' in result and 'group_best' in result and 'all_bt' in result
    # combined must equal the documented formula
    expected = result['tech_score'] + result['inst_score'] * 0.12 + 0.15 + 0.0
    assert abs(result['combined'] - expected) < 1e-9


def test_analyze_raises_on_insufficient_history():
    df = ta.calc_indicators(_make_ohlcv(n=10))
    with pytest.raises(ValueError):
        ta.analyze(df, 'X', pd.DataFrame())


def test_analyze_handles_missing_optional_inputs_gracefully():
    df      = ta.calc_indicators(_make_ohlcv(n=40))
    result  = ta.analyze(df, 'X', pd.DataFrame())  # no vix/buffett/regime/futures
    assert result['vix'] == {}
    assert result['regime_bonus'] == 0.0
    assert result['futures_bonus'] == 0.0


# ── fmt_report ────────────────────────────────────────────────────────────────

def _full_result(earnings_date=None):
    df      = ta.calc_indicators(_make_ohlcv())
    df_inst = _make_df_inst()
    regime  = {'regime': '強多頭', 'bonus': 0.25, 'twii': 20000.0, 'ma200': 18000.0, 'ratio': 1.11}
    vix     = {'value': 14.0, 'level': '極度貪婪', 'desc': '市場情緒極度樂觀'}
    buffett = {'ratio': 190.0, 'level': '偏高', 'desc': '注意風險控管',
               'market_cap_bn': 55_000.0, 'gdp_bn': 23_599.0, 'gdp_year': 2023, 'note': '估算'}
    futures = {'bonus': 0.1, 'short_bonus': -0.05, 'net_pos': -1000.0, 'short_oi': 2000.0,
               'note': '樣本20天', 'short_note': '樣本20天'}
    result  = ta.analyze(df, '2330', df_inst, vix=vix, buffett=buffett, regime=regime, futures=futures)
    result['company_name']  = '台積電'
    result['earnings_date'] = earnings_date
    return result


def test_fmt_report_renders_all_sections():
    report = ta.fmt_report(_full_result())
    for must_have in ('2330', '台積電', '技術指標快照', '總體市場指標', '三大法人',
                       '歷史回測', '▶ 未來一週建議'):
        assert must_have in report


def test_fmt_report_flags_upcoming_earnings_date():
    soon   = date.today() + timedelta(days=3)
    report = ta.fmt_report(_full_result(earnings_date=soon))
    assert "財報將於" in report


def test_fmt_report_handles_missing_optional_sections():
    df     = ta.calc_indicators(_make_ohlcv(n=40))
    result = ta.analyze(df, 'X', pd.DataFrame())
    report = ta.fmt_report(result)
    assert "X" in report
    assert "總體市場指標" not in report  # no vix/buffett/regime/futures supplied


if __name__ == "__main__":
    import inspect
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"OK  {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print("all checks passed" if not failures else f"{failures} test(s) failed")
    sys.exit(1 if failures else 0)
