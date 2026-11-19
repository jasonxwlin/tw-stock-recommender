#!/usr/bin/env python3
"""Self-check for fundamentals.py's network-facing fetchers, fmt_report, and
main(). No real network calls — requests/yfinance are mocked at the boundary."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date
from unittest import mock

import pandas as pd

import fundamentals as f


def _resp(json_data=None, content=b""):
    r = mock.Mock()
    r.json.return_value = json_data
    r.content = content
    return r


# ── _fetch_pe_pb_twse_month / _fetch_pe_pb_tpex_day ──────────────────────────

def test_fetch_pe_pb_twse_month_parses_rows_and_skips_bad_ones():
    data = {'data': [['2330', 'x', 'x', '28.5', '9.9'], ['bad', 'row']]}
    with mock.patch.object(f.requests, "get", return_value=_resp(json_data=data)):
        rows = f._fetch_pe_pb_twse_month('2330', date(2026, 1, 1))
    assert rows == [{'pe': 28.5, 'pb': 9.9}]


def test_fetch_pe_pb_twse_month_returns_empty_on_exception():
    with mock.patch.object(f.requests, "get", side_effect=Exception("boom")):
        assert f._fetch_pe_pb_twse_month('2330', date(2026, 1, 1)) == []


def test_fetch_pe_pb_tpex_day_parses_matching_row():
    data = {'tables': [{'data': [['1234', 'x', '12.3', 'x', 'x', 'x', '4.5']]}]}
    with mock.patch.object(f.requests, "get", return_value=_resp(json_data=data)):
        row = f._fetch_pe_pb_tpex_day('1234', date(2026, 1, 5))
    assert row == {'pe': 12.3, 'pb': 4.5}


def test_fetch_pe_pb_tpex_day_returns_none_when_no_match_or_exception():
    data = {'tables': [{'data': [['9999', 'x', '1.0', 'x', 'x', 'x', '1.0']]}]}
    with mock.patch.object(f.requests, "get", return_value=_resp(json_data=data)):
        assert f._fetch_pe_pb_tpex_day('1234', date(2026, 1, 5)) is None
    with mock.patch.object(f.requests, "get", side_effect=Exception("boom")):
        assert f._fetch_pe_pb_tpex_day('1234', date(2026, 1, 5)) is None


# ── fetch_valuation_snapshot: official OpenAPI snapshot ──────────────────────

def test_openapi_snapshot_twse_and_tpex(monkeypatch):
    monkeypatch.setattr(f, "_OPENAPI_PE_CACHE", {})
    twse = [{'Date': '1151117', 'Code': '2382', 'Name': '廣達', 'PEratio': '10.71', 'DividendYield': '5.94', 'PBratio': '3.13'},
            {'Date': '1151117', 'Code': '9999', 'Name': '虧損股', 'PEratio': '', 'DividendYield': '', 'PBratio': ''}]
    tpex = [{'Date': '1151117', 'SecuritiesCompanyCode': '6274', 'PriceEarningRatio': '27.70', 'YieldRatio': '0.95', 'PriceBookRatio': '8.41'}]
    with mock.patch.object(f.requests, "get", side_effect=[_resp(json_data=twse), _resp(json_data=tpex)]) as m:
        assert f.fetch_valuation_snapshot('2382', is_otc=False) == {
            'pe': 10.71, 'pb': 3.13, 'dy': 5.94, 'as_of': '2026-11-17', 'stale': False}
        assert f.fetch_valuation_snapshot('6274', is_otc=True)['pe'] == 27.70
        assert f._fetch_pe_pb_openapi('9999', is_otc=False) is None   # no PE/PB at all → skipped
        f.fetch_valuation_snapshot('2382', is_otc=False)               # served from the per-process table
    assert m.call_count == 2   # one request per market, not per stock


def test_openapi_failure_falls_back_to_monthly_history_marked_stale(monkeypatch):
    monkeypatch.setattr(f, "_OPENAPI_PE_CACHE", {})
    fallback = {'pe': 46.01, 'pb': 14.05, 'as_of': '2026-01', 'stale': True}
    with mock.patch.object(f.requests, "get", side_effect=Exception("boom")), \
         mock.patch.object(f, "_valuation_from_monthly_history", return_value=fallback):
        assert f.fetch_valuation_snapshot('2383', is_otc=False) == fallback


def test_fmt_report_marks_stale_and_shows_date_and_yield():
    stale = f.fmt_report('2383', '台光電', {'valuation': {'pe': 46.0, 'pb': 14.0, 'as_of': '2026-01', 'stale': True}, 'revenue': [], 'quality': []})
    assert '舊資料' in stale and '2026-01' in stale
    live = f.fmt_report('2382', '廣達', {'valuation': {'pe': None, 'pb': 3.1, 'dy': 5.9, 'as_of': '2026-11-17', 'stale': False}, 'revenue': [], 'quality': []})
    assert '資料日期 2026-11-17' in live and 'N/A' in live and '殖利率: 5.90%' in live


# ── fetch_valuation_snapshot: OTC path + empty-history path ─────────────────

def test_fetch_valuation_snapshot_otc_path_uses_tpex_fetcher(tmp_path):
    orig_cache_file = f._VALUATION_CACHE_FILE
    f._VALUATION_CACHE_FILE = tmp_path / "cache.json"
    try:
        with mock.patch.object(f, "_fetch_pe_pb_tpex_day", return_value={'pe': 20.0, 'pb': 2.0}) as m:
            result = f._valuation_from_monthly_history('1234', is_otc=True, months=1)
        assert (result['pe'], result['pb'], result['stale']) == (20.0, 2.0, False)
        assert m.called
    finally:
        f._VALUATION_CACHE_FILE = orig_cache_file


def test_fetch_valuation_snapshot_returns_empty_when_no_history(tmp_path):
    orig_cache_file = f._VALUATION_CACHE_FILE
    f._VALUATION_CACHE_FILE = tmp_path / "cache.json"
    try:
        with mock.patch.object(f, "_fetch_pe_pb_twse_month", return_value=[]):
            assert f._valuation_from_monthly_history('2330', is_otc=False, months=1) == {}
    finally:
        f._VALUATION_CACHE_FILE = orig_cache_file


# ── fetch_monthly_revenue ────────────────────────────────────────────────────

REVENUE_HTML = (
    '<tr align=right><td align=center>2330</td><td align=left>台積電</td>'
    '<td nowrap>            1</td><td nowrap>            1</td>'
    '<td nowrap>            1</td><td nowrap>                  1.00</td>'
    '<Td nowrap>                 12.34</td><td nowrap>          1</td>'
    '<td nowrap>          1</td><td nowrap>                 1.00</td>'
    '<td align=center>-</td></tr>'
)


def test_fetch_monthly_revenue_decodes_big5_and_parses_target_symbol():
    resp = mock.Mock()
    resp.content = REVENUE_HTML.encode('big5')
    with mock.patch.object(f.requests, "get", return_value=resp):
        rows = f.fetch_monthly_revenue('2330', is_otc=False, months=1)
    assert len(rows) == 1
    assert rows[0]['yoy_pct'] == 12.34


def test_fetch_monthly_revenue_skips_months_with_request_failures():
    with mock.patch.object(f.requests, "get", side_effect=Exception("boom")):
        assert f.fetch_monthly_revenue('2330', is_otc=False, months=2) == []


def test_fetch_monthly_revenue_skips_months_with_no_match():
    resp = mock.Mock()
    resp.content = REVENUE_HTML.encode('big5')
    with mock.patch.object(f.requests, "get", return_value=resp):
        assert f.fetch_monthly_revenue('9999', is_otc=False, months=1) == []


# ── fetch_quality_trend ──────────────────────────────────────────────────────

def _fake_ticker(with_data=True):
    idx_fin = ['Total Revenue', 'Gross Profit', 'Net Income']
    idx_bs  = ['Common Stock Equity']
    cols    = pd.to_datetime(['2026-06-30', '2026-03-31'])
    t = mock.Mock()
    if with_data:
        t.quarterly_financials = pd.DataFrame(
            [[1000.0, 900.0], [400.0, 350.0], [100.0, 90.0]], index=idx_fin, columns=cols)
        t.quarterly_balance_sheet = pd.DataFrame([[2000.0, 1900.0]], index=idx_bs, columns=cols)
    else:
        t.quarterly_financials = pd.DataFrame()
        t.quarterly_balance_sheet = pd.DataFrame()
    return t


def test_fetch_quality_trend_computes_roe_and_gross_margin_oldest_first():
    with mock.patch.object(f.yf, "Ticker", return_value=_fake_ticker()):
        rows = f.fetch_quality_trend('2330.TW', quarters=2)
    assert [r['quarter'] for r in rows] == ['2026-03-31', '2026-06-30']
    assert abs(rows[1]['roe'] - 5.0) < 1e-9          # 100/2000*100
    assert abs(rows[1]['gross_margin'] - 40.0) < 1e-9  # 400/1000*100


def test_fetch_quality_trend_returns_empty_when_no_financial_data():
    with mock.patch.object(f.yf, "Ticker", return_value=_fake_ticker(with_data=False)):
        assert f.fetch_quality_trend('2330.TW') == []


def test_fetch_quality_trend_returns_empty_on_exception():
    with mock.patch.object(f.yf, "Ticker", side_effect=Exception("boom")):
        assert f.fetch_quality_trend('2330.TW') == []


# ── analyze_fundamentals ─────────────────────────────────────────────────────

def test_analyze_fundamentals_combines_all_three_sections():
    with mock.patch.object(f, "fetch_valuation_snapshot", return_value={'pe': 1.0}), \
         mock.patch.object(f, "fetch_monthly_revenue", return_value=[{'year_month': '2026/01', 'yoy_pct': 1.0}]), \
         mock.patch.object(f, "fetch_quality_trend", return_value=[{'quarter': 'q', 'roe': 1.0, 'gross_margin': 1.0}]):
        result = f.analyze_fundamentals('2330', False, '2330.TW')
    assert set(result) == {'valuation', 'revenue', 'quality'}


# ── fmt_report ────────────────────────────────────────────────────────────────

def test_fmt_report_renders_populated_sections():
    r = {
        'valuation': {'pe': 28.4, 'pb': 9.9},
        'revenue':   [{'year_month': '2026/08', 'yoy_pct': 53.3}],
        'quality':   [{'quarter': '2026-06-30', 'roe': 11.0, 'gross_margin': 67.7}],
    }
    report = f.fmt_report('2330', '台積電', r)
    assert '台積電' in report
    assert '本益比: 28.40' in report
    assert '股價淨值比: 9.90' in report
    assert '2026/08: +53.30%' in report
    assert '單季ROE=11.0%' in report


def test_fmt_report_handles_missing_data_and_none_roe():
    r = {
        'valuation': {},
        'revenue': [],
        'quality': [{'quarter': '2026-06-30', 'roe': None, 'gross_margin': None}],
    }
    report = f.fmt_report('2330', '', r)
    assert '資料無法取得' in report
    assert 'N/A' in report


# ── main() ────────────────────────────────────────────────────────────────────

def test_main_prints_usage_and_exits_when_no_symbols(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fundamentals.py"])
    try:
        f.main()
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert e.code == 1
    assert "用法" in capsys.readouterr().out


def test_main_happy_path(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fundamentals.py", "2330"])
    with mock.patch.object(f, "fetch_price_data", return_value=(pd.DataFrame(), "2330.TW")), \
         mock.patch.object(f, "fetch_company_name", return_value="台積電"), \
         mock.patch.object(f, "analyze_fundamentals", return_value={'valuation': {}, 'revenue': [], 'quality': []}):
        f.main()
    out = capsys.readouterr().out
    assert "台積電" in out
    assert out.rstrip().endswith(f.DISCLAIMER)


def test_main_reports_error_and_continues_on_exception(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fundamentals.py", "BAD"])
    with mock.patch.object(f, "fetch_price_data", side_effect=ValueError("找不到股票")):
        f.main()  # must not raise
    assert "[錯誤]" in capsys.readouterr().out


if __name__ == "__main__":
    import inspect
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            params = inspect.signature(fn).parameters
            if params and 'tmp_path' not in params:
                print(f"SKIP {name} (needs pytest fixtures; run via `pytest`)")
                continue
            try:
                if 'tmp_path' in params:
                    import tempfile
                    fn(Path(tempfile.mkdtemp()))
                else:
                    fn()
                print(f"OK  {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print("all checks passed" if not failures else f"{failures} test(s) failed")
    sys.exit(1 if failures else 0)
