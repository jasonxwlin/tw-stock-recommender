"""
Walk-forward backtest engine for tech_analysis.py: given a dict of boolean
condition Series (from conditions.py) and 5-day-forward outcome labels,
compute each condition's historical up/down rate and excess edge vs baseline.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

HORIZON     = 5     # trading days forward
MOVE_THRESH = 0.015 # ±1.5% counts as significant move
MIN_SAMPLES = 4     # minimum occurrences to trust a condition

# Condition → indicator group mapping (for deduplication)
COND_GROUPS: dict[str, str] = {}  # populated lazily by _group_of()

# (predicate, group) rules for _group_of(), checked in order; first match wins.
_GROUP_RULES: list[tuple[Callable[[str], bool], str]] = [
    (lambda n: n.startswith(('KD', 'K值', 'K低', 'K高')), 'KD'),
    (lambda n: n.startswith('MACD'), 'MACD'),
    (lambda n: n.startswith('RSI'), 'RSI'),
    (lambda n: n.startswith('BIAS'), 'BIAS'),
    (lambda n: '5MA' in n or '5ma' in n.lower(), 'MA5'),
    (lambda n: '月線' in n or '20MA' in n, 'MA20'),
    (lambda n: '季線' in n or '60MA' in n, 'MA60'),
    (lambda n: '200MA' in n, 'MA200'),
    (lambda n: '量' in n, 'Volume'),
]


def _group_of(name: str) -> str:
    """Which indicator group `name` belongs to, for one-winner-per-group scoring."""
    for predicate, group in _GROUP_RULES:
        if predicate(name):
            return group
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


def _condition_stats(cond: pd.Series, known: pd.Series, base_edge: float) -> dict | None:
    """One condition's historical up/down/excess-edge stats, or None if < MIN_SAMPLES."""
    aligned = cond.reindex(known.index).fillna(False)
    hits    = known[aligned]
    n       = len(hits)
    if n < MIN_SAMPLES:
        return None
    up   = float((hits ==  1).mean())
    down = float((hits == -1).mean())
    flat = float((hits ==  0).mean())
    edge        = up - down
    excess_edge = edge - base_edge        # how much better/worse than doing nothing
    conf        = min(n / 20.0, 1.0)     # confidence saturates at 20 samples
    return {
        'count':       n,
        'up_rate':     up,
        'down_rate':   down,
        'flat_rate':   flat,
        'edge':        edge,
        'excess_edge': excess_edge,
        'confidence':  conf,
        'weight':      excess_edge * conf, # effective score contribution
    }


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
    known     = outcomes.dropna()
    base_edge = base['up_rate'] - base['down_rate']
    results: dict[str, dict] = {}

    for name, cond in conditions.items():
        stats = _condition_stats(cond, known, base_edge)
        if stats is None:
            continue
        stats['group'] = _group_of(name)
        results[name] = stats

    return results


def baseline_stats(outcomes: pd.Series) -> dict:
    """Unconditional up/down/flat rates — the "do nothing" edge every condition is compared against."""
    known = outcomes.dropna()
    return {
        'count':     len(known),
        'up_rate':   float((known == 1).mean()),
        'down_rate': float((known == -1).mean()),
        'flat_rate': float((known == 0).mean()),
        'edge':      float((known == 1).mean()) - float((known == -1).mean()),
    }
