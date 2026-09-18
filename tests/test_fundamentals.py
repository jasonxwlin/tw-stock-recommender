#!/usr/bin/env python3
"""Self-check for fundamentals.py's parsing/date-math/cache-throttling helpers (ponytail: non-trivial regex + branching logic)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile
from datetime import date
import fundamentals as f
from fundamentals import _parse_revenue_yoy, _months_back, _roc_year

REVENUE_FIXTURE = (
    '<tr align=right><td align=center>2330</td><td align=left>台積電</td>'
    '<td nowrap>            467,580,548</td><td nowrap>            442,679,969</td>'
    '<td nowrap>            323,165,707</td><td nowrap>                  5.62</td>'
    '<Td nowrap>                 44.68</td><td nowrap>          2,872,064,238</td>'
    '<td nowrap>          2,096,211,240</td><td nowrap>                 37.01</td>'
    '<td align=center>-</td></tr>'
    '<tr align=right><td align=center>2317</td><td align=left>鴻海</td>'
    '<td nowrap>            1</td><td nowrap>            1</td>'
    '<td nowrap>            1</td><td nowrap>                  1.00</td>'
    '<td nowrap>                 -3.50</td><td nowrap>          1</td>'
    '<td nowrap>          1</td><td nowrap>                 1.00</td>'
    '<td align=center>-</td></tr>'
)


def test_parse_revenue_yoy_finds_target_stock_only():
    assert _parse_revenue_yoy(REVENUE_FIXTURE, '2330') == 44.68
    assert _parse_revenue_yoy(REVENUE_FIXTURE, '2317') == -3.50
    assert _parse_revenue_yoy(REVENUE_FIXTURE, '9999') is None


def test_months_back_walks_backward_across_year_boundary():
    months = _months_back(3)
    assert len(months) == 3
    assert months[0] == date.today().replace(day=1)
    for a, b in zip(months, months[1:]):
        assert b < a
        assert (a.year * 12 + a.month) - (b.year * 12 + b.month) == 1


def test_roc_year():
    assert _roc_year(date(2026, 1, 1)) == 115


def test_valuation_cache_backfills_one_past_month_per_call():
    """Regression check for the TWSE HTTP 428 flood-protection incident: a cold
    run must not burst-fetch every month at once. Only one new past month (plus
    the always-fresh current month) is fetched per call; the rest of the
    trailing window fills in gradually over repeated calls."""
    orig_cache_file = f._VALUATION_CACHE_FILE
    orig_fetch = f._fetch_pe_pb_twse_month
    f._VALUATION_CACHE_FILE = Path(tempfile.mkdtemp()) / "test_valuation_cache.json"
    calls = []
    f._fetch_pe_pb_twse_month = lambda symbol, d: calls.append(d) or [{'pe': 10.0 + d.month, 'pb': 1.0}]
    try:
        # months=3 → 2 past months + the current month.
        f.fetch_valuation_snapshot('TEST', is_otc=False, months=3)
        assert len(calls) == 2  # 1 new past month + current month, not all 3 at once
        assert len(f._load_valuation_cache()['TEST']) == 1

        f.fetch_valuation_snapshot('TEST', is_otc=False, months=3)
        assert len(calls) == 4  # the other past month + current month again
        assert len(f._load_valuation_cache()['TEST']) == 2  # both past months now cached

        f.fetch_valuation_snapshot('TEST', is_otc=False, months=3)
        assert len(calls) == 5  # both past months cached — only current month re-fetched
        assert len(f._load_valuation_cache()['TEST']) == 2
    finally:
        f._fetch_pe_pb_twse_month = orig_fetch
        f._VALUATION_CACHE_FILE = orig_cache_file


def test_valuation_snapshot_has_no_percentile_field():
    orig_cache_file = f._VALUATION_CACHE_FILE
    orig_fetch = f._fetch_pe_pb_twse_month
    f._VALUATION_CACHE_FILE = Path(tempfile.mkdtemp()) / "test_valuation_cache.json"
    f._fetch_pe_pb_twse_month = lambda symbol, d: [{'pe': 12.3, 'pb': 4.5}]
    try:
        result = f.fetch_valuation_snapshot('TEST', is_otc=False, months=1)
        assert result == {'pe': 12.3, 'pb': 4.5}
    finally:
        f._fetch_pe_pb_twse_month = orig_fetch
        f._VALUATION_CACHE_FILE = orig_cache_file
