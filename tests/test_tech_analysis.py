#!/usr/bin/env python3
"""Self-check for tech_analysis.py's multi-stock summary sort (ponytail: non-trivial multi-key sort)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import tech_analysis as ta
from tech_analysis import (
    fmt_summary_table, _REC_RANK, _group_of, _consecutive,
    compute_outcomes, baseline_stats, backtest_conditions, MIN_SAMPLES,
)

FAKE_RESULTS = [
    {'symbol': 'A', 'company_name': 'AA', 'price': 100.0, 'tech_score': 0.1,
     'inst_score': 0, 'combined': -0.1, 'recommendation': '持平'},
    {'symbol': 'B', 'company_name': 'BB', 'price': 200.0, 'tech_score': 0.5,
     'inst_score': 3, 'combined': 0.9, 'recommendation': '強力加碼'},
    {'symbol': 'C', 'company_name': 'CC', 'price': 300.0, 'tech_score': 0.2,
     'inst_score': 0, 'combined': 0.3, 'recommendation': '加碼'},
    {'symbol': 'D', 'company_name': 'DD', 'price': 50.0, 'tech_score': -0.8,
     'inst_score': -3, 'combined': -0.7, 'recommendation': '強力減碼'},
    {'symbol': 'E', 'company_name': 'EE', 'price': 60.0, 'tech_score': 0.6,
     'inst_score': 3, 'combined': 1.2, 'recommendation': '強力加碼'},
]


def test_sorted_by_recommendation_tier_then_score_descending():
    table = fmt_summary_table(FAKE_RESULTS)
    order = [line.split()[0] for line in table.splitlines() if line.strip()[:1].isalpha()
             and line.strip()[0] in 'ABCDE']
    # Two 強力加碼 rows: E (combined 1.2) must outrank B (combined 0.9); then 加碼 C;
    # then 持平 A; then 強力減碼 D last.
    assert order == ['E', 'B', 'C', 'A', 'D'], order


def test_rec_rank_is_strictly_descending_bull_to_bear():
    tiers = ['強力加碼', '加碼', '持平', '減碼', '強力減碼']
    ranks = [_REC_RANK[t] for t in tiers]
    assert ranks == sorted(ranks, reverse=True)


def test_group_of_maps_known_prefixes_and_fallback():
    assert _group_of("跌破5MA") == "MA5"
    assert _group_of("KD黃金交叉") == "KD"
    assert _group_of("K值超買(>80)") == "KD"
    assert _group_of("MACD黃金交叉") == "MACD"
    assert _group_of("RSI超買(>70)") == "RSI"
    assert _group_of("BIAS20偏高(>10%)") == "BIAS"
    assert _group_of("站上月線20MA") == "MA20"
    assert _group_of("站上季線60MA") == "MA60"
    assert _group_of("站上200MA") == "MA200"
    assert _group_of("爆量上漲") == "Volume"
    assert _group_of("某個未知條件") == "Compound"


def test_consecutive_counts_streak_and_stops_at_sign_change():
    assert _consecutive([1, 1, 1, -1]) == (3, 0)
    assert _consecutive([-1, -1, 1]) == (0, 2)
    assert _consecutive([0, 1, 1]) == (0, 0)
    assert _consecutive([]) == (0, 0)


def test_compute_outcomes_labels_up_down_flat_and_nan_tail():
    # HORIZON=5, MOVE_THRESH=0.015 (defaults). 12 rows: idx0/1/2 have a known
    # forward-5 target; last 5 rows (idx7..11) have no forward data → NaN.
    close = [100.0] * 12
    close[5]  = 103.0   # idx0 forward: +3.0%  → up
    close[6]  = 97.0    # idx1 forward: -3.0%  → down
    close[7]  = 100.5   # idx2 forward: +0.5%  → flat
    df = pd.DataFrame({'Close': close})

    outcomes = compute_outcomes(df)

    assert outcomes.iloc[0] == 1.0
    assert outcomes.iloc[1] == -1.0
    assert outcomes.iloc[2] == 0.0
    assert outcomes.iloc[7:].isna().all()


def test_baseline_stats_and_backtest_conditions_excess_edge():
    # 10 known outcomes: 4 up, 3 down, 3 flat → base edge = 0.4 - 0.3 = 0.1
    outcomes = pd.Series([1, 1, 1, 1, -1, -1, -1, 0, 0, 0], dtype=float)
    base = baseline_stats(outcomes)
    assert abs(base['up_rate'] - 0.4) < 1e-9
    assert abs(base['down_rate'] - 0.3) < 1e-9
    assert abs(base['edge'] - 0.1) < 1e-9

    # ALWAYS is active on the first 5 rows ([1,1,1,1,-1] → up=0.8, down=0.2,
    # n=5 ≥ MIN_SAMPLES). RARE is active on only 2 rows (< MIN_SAMPLES) and
    # must be dropped from the results entirely.
    conditions = {
        'ALWAYS': pd.Series([True] * 5 + [False] * 5),
        'RARE':   pd.Series([True, True] + [False] * 8),
    }
    assert 2 < MIN_SAMPLES <= 5, "test assumes MIN_SAMPLES is between 3 and 5"

    bt = backtest_conditions(conditions, outcomes, base)

    assert 'RARE' not in bt
    stats = bt['ALWAYS']
    assert stats['count'] == 5
    assert abs(stats['up_rate'] - 0.8) < 1e-9
    assert abs(stats['down_rate'] - 0.2) < 1e-9
    assert abs(stats['excess_edge'] - 0.5) < 1e-9  # (0.8-0.2) - 0.1
    assert abs(stats['confidence'] - 0.25) < 1e-9  # 5/20
    assert abs(stats['weight'] - 0.125) < 1e-9     # 0.5 * 0.25


def _analyze_with_fixed_tech_score(tech_score: float, regime: dict | None = None) -> str:
    """Monkeypatch analyze()'s inputs so `combined` == tech_score exactly,
    then return the resulting recommendation tier for that regime."""
    df = pd.DataFrame({'Close': [100.0] * 30})
    orig = (ta.build_conditions, ta.compute_outcomes, ta.baseline_stats,
            ta.backtest_conditions, ta.analyze_institutional)
    try:
        ta.build_conditions       = lambda df: {'X': pd.Series([True])}
        ta.compute_outcomes       = lambda df, move_thresh=None: pd.Series([0.0])
        ta.baseline_stats         = lambda outcomes: {
            'count': 1, 'up_rate': 0.0, 'down_rate': 0.0, 'flat_rate': 1.0, 'edge': 0.0,
        }
        ta.backtest_conditions    = lambda conditions, outcomes, base: {
            'X': {'count': 10, 'up_rate': 0.5, 'down_rate': 0.3, 'flat_rate': 0.2,
                  'edge': 0.2, 'excess_edge': tech_score, 'confidence': 1.0,
                  'weight': tech_score, 'group': 'Compound'},
        }
        ta.analyze_institutional  = lambda df_inst: (0, [], {})
        macro  = {'regime': regime} if regime is not None else None
        result = ta.analyze(df, 'TEST', df_inst=pd.DataFrame(), macro=macro)
    finally:
        (ta.build_conditions, ta.compute_outcomes, ta.baseline_stats,
         ta.backtest_conditions, ta.analyze_institutional) = orig
    return result['recommendation']


def test_analyze_recommendation_thresholds_neutral_regime():
    # 中性 (no regime passed → bonus=0): add=0.25, reduce=-0.25, strong=-0.60.
    # 強力加碼's >=0.6 cutoff is hardcoded, not regime-dependent.
    assert _analyze_with_fixed_tech_score(0.60)  == '強力加碼'
    assert _analyze_with_fixed_tech_score(0.59)  == '加碼'
    assert _analyze_with_fixed_tech_score(0.25)  == '加碼'
    assert _analyze_with_fixed_tech_score(0.24)  == '持平'
    assert _analyze_with_fixed_tech_score(-0.25) == '減碼'
    assert _analyze_with_fixed_tech_score(-0.60) == '強力減碼'
    assert _analyze_with_fixed_tech_score(-0.59) == '減碼'


def test_analyze_recommendation_depends_on_regime():
    # Same combined score (-0.70): 中性's strong=-0.60 catches it as 強力減碼,
    # but 強多頭's looser strong=-0.80 only downgrades it to 減碼.
    bull = {'regime': '強多頭', 'bonus': 0.0}
    assert _analyze_with_fixed_tech_score(-0.70, regime=None) == '強力減碼'
    assert _analyze_with_fixed_tech_score(-0.70, regime=bull) == '減碼'
