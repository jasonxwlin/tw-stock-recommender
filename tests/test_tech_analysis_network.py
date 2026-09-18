#!/usr/bin/env python3
"""Self-check for tech_analysis.py's network-facing fetch_* functions and
main(). No real network calls — requests/yfinance are mocked at the boundary
so we can exercise each function's parsing/branching/error-handling logic."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from datetime import date
from unittest import mock

import numpy as np
import pandas as pd
import requests as real_requests

import tech_analysis as ta


def _resp(json_data=None, text="", status_code=200):
    r = mock.Mock()
    r.json.return_value = json_data
    r.text = text
    r.status_code = status_code
    return r


# ── company name cache ───────────────────────────────────────────────────────

def test_load_company_cache_populates_from_twse_openapi():
    ta._COMPANY_CACHE.clear()
    fake = [{'公司代號': '2330', '公司簡稱': '台積電'}, {'公司代號': '', '公司簡稱': 'x'}]
    with mock.patch.object(ta.requests, "get", return_value=_resp(json_data=fake)):
        assert ta.fetch_company_name('2330') == '台積電'
    ta._COMPANY_CACHE.clear()


def test_load_company_cache_swallows_exceptions():
    ta._COMPANY_CACHE.clear()
    with mock.patch.object(ta.requests, "get", side_effect=Exception("boom")):
        assert ta.fetch_company_name('9999') == ''
    ta._COMPANY_CACHE.clear()


def test_load_company_cache_is_a_noop_once_populated():
    ta._COMPANY_CACHE.clear()
    ta._COMPANY_CACHE['1101'] = '台泥'
    with mock.patch.object(ta.requests, "get") as m:
        assert ta.fetch_company_name('1101') == '台泥'
        m.assert_not_called()
    ta._COMPANY_CACHE.clear()


# ── fetch_price_data ─────────────────────────────────────────────────────────

def _ohlcv_df(n=5):
    idx = pd.bdate_range("2026-01-01", periods=n)
    return pd.DataFrame({'Open': 1.0, 'High': 1.0, 'Low': 1.0, 'Close': 1.0, 'Volume': 1.0}, index=idx)


def test_fetch_price_data_falls_back_to_two_suffix():
    with mock.patch.object(ta.yf, "download", side_effect=[pd.DataFrame(), _ohlcv_df()]):
        df, ticker = ta.fetch_price_data("1234")
    assert ticker == "1234.TWO"
    assert not df.empty


def test_fetch_price_data_flattens_multiindex_columns():
    df = _ohlcv_df()
    df.columns = pd.MultiIndex.from_product([df.columns, ["1234.TW"]])
    with mock.patch.object(ta.yf, "download", return_value=df):
        result, ticker = ta.fetch_price_data("1234")
    assert ticker == "1234.TW"
    assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']


def test_fetch_price_data_raises_when_both_suffixes_empty():
    with mock.patch.object(ta.yf, "download", return_value=pd.DataFrame()):
        try:
            ta.fetch_price_data("0000")
            assert False, "expected ValueError"
        except ValueError:
            pass


# ── fetch_stock_earnings_date ────────────────────────────────────────────────

def test_fetch_stock_earnings_date_returns_first_date():
    fake_cal = {'Earnings Date': [date(2026, 10, 1), date(2026, 10, 2)]}
    with mock.patch.object(ta.yf, "Ticker") as m:
        m.return_value.calendar = fake_cal
        assert ta.fetch_stock_earnings_date("2330.TW") == date(2026, 10, 1)


def test_fetch_stock_earnings_date_returns_none_when_missing_or_wrong_type():
    with mock.patch.object(ta.yf, "Ticker") as m:
        m.return_value.calendar = {}
        assert ta.fetch_stock_earnings_date("2330.TW") is None
    with mock.patch.object(ta.yf, "Ticker") as m:
        m.return_value.calendar = {'Earnings Date': "not-a-date"}
        assert ta.fetch_stock_earnings_date("2330.TW") is None


def test_fetch_stock_earnings_date_swallows_exceptions():
    with mock.patch.object(ta.yf, "Ticker", side_effect=Exception("boom")):
        assert ta.fetch_stock_earnings_date("2330.TW") is None


# ── _fetch_twse_day / _fetch_tpex_day ────────────────────────────────────────

def _twse_row(symbol='2330'):
    row = [symbol] + ['0'] * 20
    row[4], row[7], row[10], row[11], row[18] = '1,000', '500', '200', '10', '99,000'
    return row


def test_fetch_twse_day_parses_matching_row():
    data = {'stat': 'OK', 'data': [_twse_row('9999'), _twse_row('2330')]}
    with mock.patch.object(ta.requests, "get", return_value=_resp(json_data=data)):
        row = ta._fetch_twse_day("20260101", "2330")
    assert row == {'date': '20260101', 'foreign': 1500, 'trust': 200, 'dealer': 10, 'total': 99000}


def test_fetch_twse_day_returns_none_when_stat_not_ok_or_symbol_missing():
    with mock.patch.object(ta.requests, "get", return_value=_resp(json_data={'stat': 'ERROR'})):
        assert ta._fetch_twse_day("20260101", "2330") is None
    with mock.patch.object(ta.requests, "get", return_value=_resp(json_data={'stat': 'OK', 'data': []})):
        assert ta._fetch_twse_day("20260101", "2330") is None


def test_fetch_twse_day_swallows_exceptions():
    with mock.patch.object(ta.requests, "get", side_effect=Exception("boom")):
        assert ta._fetch_twse_day("20260101", "2330") is None


def _tpex_row(symbol='1234'):
    row = [symbol] + ['0'] * 20
    row[4], row[7], row[10], row[13], row[17] = '800', '200', '300', '5', '50,000'
    return row


def test_fetch_tpex_day_parses_matching_row_from_aadata():
    data = {'aaData': [_tpex_row('1234')]}
    with mock.patch.object(ta.requests, "get", return_value=_resp(json_data=data)):
        row = ta._fetch_tpex_day("20260101", "1234")
    assert row == {'date': '20260101', 'foreign': 1000, 'trust': 300, 'dealer': 5, 'total': 50000}


def test_fetch_tpex_day_falls_back_to_data_key_and_handles_no_match():
    data = {'data': [_tpex_row('1234')]}
    with mock.patch.object(ta.requests, "get", return_value=_resp(json_data=data)):
        assert ta._fetch_tpex_day("20260101", "9999") is None
        assert ta._fetch_tpex_day("20260101", "1234") is not None


def test_fetch_tpex_day_swallows_exceptions():
    with mock.patch.object(ta.requests, "get", side_effect=Exception("boom")):
        assert ta._fetch_tpex_day("20260101", "1234") is None


# ── fetch_institutional ──────────────────────────────────────────────────────

def test_fetch_institutional_aggregates_and_caps_at_n_days():
    rows = [{'date': f"2026{i:04d}", 'foreign': i, 'trust': i, 'dealer': 0, 'total': i}
            for i in range(1, 30)]
    with mock.patch.object(ta, "_fetch_twse_day", side_effect=lambda dt, sym: rows.pop() if rows else None):
        df = ta.fetch_institutional("2330", is_otc=False, n_days=5)
    assert len(df) == 5
    assert list(df.columns) >= ['date', 'foreign', 'trust', 'dealer', 'total']


def test_fetch_institutional_uses_tpex_fetcher_when_otc():
    with mock.patch.object(ta, "_fetch_tpex_day", return_value=None) as m_tpex, \
         mock.patch.object(ta, "_fetch_twse_day", return_value=None) as m_twse:
        df = ta.fetch_institutional("1234", is_otc=True, n_days=3)
    assert df.empty
    assert m_tpex.called
    assert not m_twse.called


# ── fetch_vix ─────────────────────────────────────────────────────────────────

def _vix_df(value, multiindex=False):
    df = pd.DataFrame({'Close': [value]}, index=pd.bdate_range("2026-01-01", periods=1))
    if multiindex:
        df.columns = pd.MultiIndex.from_product([df.columns, ["^VIX"]])
    return df


def test_fetch_vix_classifies_each_level():
    cases = [(10.0, "極度貪婪"), (17.0, "樂觀"), (22.0, "中性"), (27.0, "謹慎"),
             (35.0, "恐慌"), (50.0, "極度恐慌")]
    for value, expected_level in cases:
        with mock.patch.object(ta.yf, "download", return_value=_vix_df(value)):
            result = ta.fetch_vix()
        assert result['level'] == expected_level, (value, result)


def test_fetch_vix_flattens_multiindex():
    with mock.patch.object(ta.yf, "download", return_value=_vix_df(18.0, multiindex=True)):
        result = ta.fetch_vix()
    assert result['level'] == "樂觀"


def test_fetch_vix_returns_empty_on_empty_df_or_exception():
    with mock.patch.object(ta.yf, "download", return_value=pd.DataFrame()):
        assert ta.fetch_vix() == {}
    with mock.patch.object(ta.yf, "download", side_effect=Exception("boom")):
        assert ta.fetch_vix() == {}


# ── fetch_buffett_indicator ──────────────────────────────────────────────────

def test_fetch_buffett_indicator_uses_twse_market_cap():
    twse_data = {'data': [['總市值(百萬元)', '2,000,000,000']]}
    with mock.patch.object(ta.requests, "get", return_value=_resp(json_data=twse_data)):
        result = ta.fetch_buffett_indicator()
    assert result['note'] == "TWSE實時"
    assert result['level'] in {"嚴重低估", "合理", "略偏高", "偏高", "高估"}


def test_fetch_buffett_indicator_falls_back_to_twii_estimate():
    with mock.patch.object(ta.requests, "get", return_value=_resp(json_data={'data': []})), \
         mock.patch.object(ta.yf, "download", return_value=pd.DataFrame({'Close': [20000.0]})):
        result = ta.fetch_buffett_indicator()
    assert "估算" in result['note']


def test_fetch_buffett_indicator_returns_empty_when_both_sources_fail():
    with mock.patch.object(ta.requests, "get", side_effect=Exception("boom")), \
         mock.patch.object(ta.yf, "download", return_value=pd.DataFrame()):
        assert ta.fetch_buffett_indicator() == {}


def test_fetch_buffett_indicator_classifies_each_level():
    # ratio (%) = market_cap_bn / 23599 * 100; drive market_cap_bn via the TWII
    # fallback (idx * 3.0) so each bucket boundary is hit precisely.
    cases = [(60, "嚴重低估"), (100, "合理"), (140, "略偏高"), (180, "偏高"), (250, "高估")]
    for ratio_pct, expected_level in cases:
        idx = (ratio_pct / 100 * 23_599) / 3.0
        with mock.patch.object(ta.requests, "get", side_effect=Exception("boom")), \
             mock.patch.object(ta.yf, "download", return_value=pd.DataFrame({'Close': [idx]})):
            result = ta.fetch_buffett_indicator()
        assert result['level'] == expected_level, (ratio_pct, result)


# ── fetch_macro_events ───────────────────────────────────────────────────────

FOMC_HTML = (
    '<a id="1">2026 FOMC Meetings</a></h4></div>'
    '<div class="fomc-meeting__month col-xs-5"><strong>January</strong></div>'
    '<div class="fomc-meeting__date col-xs-4">27-28</div>'
)
BEA_HTML = (
    'Year 2026'
    '<tr class="scheduled-releases-type-press">'
    '<td class="release-date">January 30</td>'
    '<td class="release-title">GDP (Fourth Estimate)</td></tr>'
)


def test_fetch_macro_events_combines_all_sources(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "fake-key")
    today   = date(2026, 1, 20)
    horizon = today.replace(day=31)

    def fake_get(url, **kwargs):
        if "federalreserve.gov" in url:
            return _resp(text=FOMC_HTML)
        if "bea.gov" in url:
            return _resp(text=BEA_HTML)
        if "stlouisfed.org" in url:
            return _resp(json_data={'release_dates': [{'date': '2026-01-25'}]})
        raise AssertionError(f"unexpected URL {url}")

    with mock.patch.object(ta, "date") as mock_date, \
         mock.patch.object(ta.requests, "get", side_effect=fake_get):
        mock_date.today.return_value = today
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        events = ta.fetch_macro_events(days_ahead=11)

    names = {e['name'] for e in events}
    assert 'FOMC利率決策會議' in names
    assert '美國GDP公布' in names
    assert '美國CPI公布' in names or '美國非農就業報告(NFP)' in names


def test_fetch_macro_events_skips_fred_without_api_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with mock.patch.object(ta.requests, "get", return_value=_resp(text="")) as m:
        events = ta.fetch_macro_events(days_ahead=7)
    assert events == []
    # only FOMC + BEA calls, never stlouisfed
    called_urls = [c.args[0] for c in m.call_args_list]
    assert not any("stlouisfed" in u for u in called_urls)


def test_fetch_macro_events_handles_source_exceptions_gracefully(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with mock.patch.object(ta.requests, "get", side_effect=Exception("boom")):
        assert ta.fetch_macro_events(days_ahead=7) == []


# ── fetch_market_regime ──────────────────────────────────────────────────────

def _twii_df_for_ratio(ratio: float, base: float = 10_000.0) -> pd.DataFrame:
    target_close = 199 * ratio * base / (200 - ratio)
    closes = [base] * 399 + [target_close]
    idx = pd.bdate_range("2025-01-01", periods=400)
    return pd.DataFrame({'Close': closes}, index=idx)


def test_fetch_market_regime_classifies_each_bucket():
    cases = [(1.10, '強多頭'), (1.02, '多頭'), (0.97, '中性'), (0.90, '空頭')]
    for ratio, expected in cases:
        with mock.patch.object(ta.yf, "download", return_value=_twii_df_for_ratio(ratio)):
            result = ta.fetch_market_regime()
        assert result['regime'] == expected, (ratio, result)


def test_fetch_market_regime_flattens_multiindex():
    df = _twii_df_for_ratio(1.10)
    df.columns = pd.MultiIndex.from_product([df.columns, ["^TWII"]])
    with mock.patch.object(ta.yf, "download", return_value=df):
        result = ta.fetch_market_regime()
    assert result['regime'] == '強多頭'


def test_fetch_market_regime_neutral_when_empty_short_history_or_exception():
    with mock.patch.object(ta.yf, "download", return_value=pd.DataFrame()):
        assert ta.fetch_market_regime() == {'regime': '中性', 'bonus': 0.0}
    with mock.patch.object(ta.yf, "download",
                            return_value=pd.DataFrame({'Close': [10000.0] * 50})):
        assert ta.fetch_market_regime() == {'regime': '中性', 'bonus': 0.0}
    with mock.patch.object(ta.yf, "download", side_effect=Exception("boom")):
        assert ta.fetch_market_regime() == {'regime': '中性', 'bonus': 0.0}


# ── futures cache + fetch_futures_net_position ───────────────────────────────

def test_futures_cache_roundtrip(tmp_path):
    cache_file = tmp_path / "cache.json"
    with mock.patch.object(ta, "FUTURES_CACHE_FILE", cache_file):
        assert ta._load_futures_cache() == {}
        ta._save_futures_cache({"2026/01/01": {"net": 1.0, "short": 2.0}})
        assert ta._load_futures_cache() == {"2026/01/01": {"net": 1.0, "short": 2.0}}


def test_load_futures_cache_returns_empty_on_corrupt_json(tmp_path):
    cache_file = tmp_path / "cache.json"
    cache_file.write_text("not json")
    with mock.patch.object(ta, "FUTURES_CACHE_FILE", cache_file):
        assert ta._load_futures_cache() == {}


def _taifex_html(nums):
    body = "".join(f"<td>{n}</td>" for n in nums)
    return f"junk table_f junk 外資{body} trailing"


def test_fetch_futures_net_position_parses_html():
    nums = [100, 200, 300, 400, 500, 600, 700, 800, 5678, 900, 1234]
    with mock.patch.object(ta.requests, "post", return_value=_resp(text=_taifex_html(nums), status_code=200)):
        net_pos, short_oi = ta.fetch_futures_net_position("20260101")
    assert net_pos == 1234.0
    assert short_oi == 5678.0


def test_fetch_futures_net_position_handles_rate_limit_and_cloudflare():
    with mock.patch.object(ta.requests, "post", return_value=_resp(text="x", status_code=429)):
        assert ta.fetch_futures_net_position("20260101") == (None, None)
    with mock.patch.object(ta.requests, "post", return_value=_resp(text="Just a moment...", status_code=200)):
        assert ta.fetch_futures_net_position("20260101") == (None, None)


def test_fetch_futures_net_position_handles_missing_marker_short_nums_and_exception():
    with mock.patch.object(ta.requests, "post", return_value=_resp(text="no marker here", status_code=200)):
        assert ta.fetch_futures_net_position("20260101") == (None, None)
    with mock.patch.object(ta.requests, "post", return_value=_resp(text=_taifex_html([1, 2, 3]), status_code=200)):
        assert ta.fetch_futures_net_position("20260101") == (None, None)
    with mock.patch.object(ta.requests, "post", side_effect=real_requests.RequestException("boom")):
        assert ta.fetch_futures_net_position("20260101") == (None, None)


# ── main() ────────────────────────────────────────────────────────────────────

def test_main_prints_usage_and_exits_when_no_symbols(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tech_analysis.py"])
    try:
        ta.main()
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert e.code == 1
    assert "用法" in capsys.readouterr().out


def _fake_ohlcv(n=260, seed=1):
    rng   = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    idx   = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame({
        'Open': close, 'High': close * 1.01, 'Low': close * 0.99,
        'Close': close, 'Volume': rng.integers(1_000, 100_000, n).astype(float),
    }, index=idx)


def _fake_df_inst():
    pattern = [1000, 2000, 1500, -500, 3000, 800, 1200, -300, -900, -1100]
    return pd.DataFrame({
        'date':    [f'2026{(i % 12) + 1:02d}{(i % 27) + 1:02d}' for i in range(len(pattern))],
        'foreign': pattern, 'trust': [v // 2 for v in pattern],
        'dealer':  [0] * len(pattern), 'total': [v + v // 2 for v in pattern],
    })


def test_main_happy_path_single_and_multi_symbol(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tech_analysis.py", "2330", "2454"])
    fake_df = _fake_ohlcv()
    fake_inst = _fake_df_inst()

    with mock.patch.object(ta, "fetch_vix", return_value={}), \
         mock.patch.object(ta, "fetch_buffett_indicator", return_value={}), \
         mock.patch.object(ta, "fetch_market_regime", return_value={'regime': '中性', 'bonus': 0.0}), \
         mock.patch.object(ta, "fetch_futures_sentiment", return_value={'bonus': 0.0, 'short_bonus': 0.0}), \
         mock.patch.object(ta, "fetch_macro_events", return_value=[{'date': date(2026, 1, 1), 'name': 'FOMC利率決策會議'}]), \
         mock.patch.object(ta, "fetch_price_data", return_value=(fake_df.copy(), "2330.TW")), \
         mock.patch.object(ta, "fetch_company_name", return_value="台積電"), \
         mock.patch.object(ta, "fetch_institutional", return_value=fake_inst), \
         mock.patch.object(ta, "fetch_stock_earnings_date", return_value=None):
        ta.main()

    out = capsys.readouterr().out
    assert "重大事件提醒" in out
    assert "綜合比較表" in out  # >1 symbol → summary table printed


def test_main_reports_error_and_continues_on_per_symbol_exception(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tech_analysis.py", "BADSTOCK"])
    with mock.patch.object(ta, "fetch_vix", return_value={}), \
         mock.patch.object(ta, "fetch_buffett_indicator", return_value={}), \
         mock.patch.object(ta, "fetch_market_regime", return_value={'regime': '中性', 'bonus': 0.0}), \
         mock.patch.object(ta, "fetch_futures_sentiment", return_value={'bonus': 0.0, 'short_bonus': 0.0}), \
         mock.patch.object(ta, "fetch_macro_events", return_value=[]), \
         mock.patch.object(ta, "fetch_price_data", side_effect=ValueError("找不到股票")):
        ta.main()  # must not raise
    assert "[錯誤]" in capsys.readouterr().out


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                import inspect
                params = inspect.signature(fn).parameters
                if params:
                    print(f"SKIP {name} (needs pytest fixtures; run via `pytest`)")
                    continue
                fn()
                print(f"OK  {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print("all checks passed" if not failures else f"{failures} test(s) failed")
    sys.exit(1 if failures else 0)
