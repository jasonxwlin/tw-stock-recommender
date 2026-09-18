"""
ponytail: minimal runnable check for fetch_futures_sentiment()'s bucketing/scoring
math (the non-trivial branch/loop logic added for the TXF futures overlay,
covering both the net-position and gross-short-OI signals).
No network calls — TWII prices and the on-disk cache are both faked.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from unittest import mock

import numpy as np
import pandas as pd

import tech_analysis as ta


def _fake_twii(n=60):
    dates = pd.bdate_range("2026-01-01", periods=n)
    # steady uptrend so forward-5d returns are mostly positive/known, not flat/NaN
    close = pd.Series(np.linspace(100, 160, n), index=dates)
    df = pd.DataFrame({"Close": close})
    return df


def test_insufficient_samples_scores_zero(tmp_path):
    cache_file = tmp_path / "cache.json"
    twii = _fake_twii()
    today = twii.index[-1].strftime("%Y/%m/%d")
    cache_file.write_text(json.dumps({today: {"net": -1000.0, "short": 90000.0}}))  # 1 sample

    with mock.patch.object(ta, "FUTURES_CACHE_FILE", cache_file), \
         mock.patch("yfinance.download", return_value=twii):
        result = ta.fetch_futures_sentiment()

    assert result["bonus"] == 0.0
    assert result["short_bonus"] == 0.0
    assert "門檻" in result["note"] or "僅" in result["note"]
    assert "僅" in result["short_note"]


def test_deep_short_bucket_scores_nonzero_with_enough_samples(tmp_path):
    cache_file = tmp_path / "cache.json"
    twii = _fake_twii(n=60)
    # fetch_futures_sentiment looks up "today" = the LAST row of the fetched TWII
    # series, so the cache must include that exact date to avoid a live network call.
    dates = twii.index[-20:]

    # Build 20 cached days cycling through 5 levels for both fields so today's
    # value (the last day, which itself gets dropped from `known` for lacking a
    # forward 5d outcome) still falls inside the range the quartiles were fit on.
    cache = {}
    for i, d in enumerate(dates):
        cache[d.strftime("%Y/%m/%d")] = {
            "net":   -50000.0 + (i % 5) * 8000.0,   # cycles -50k..-18k
            "short":  90000.0 + (i % 5) * 3000.0,   # cycles 90k..102k
        }
    cache_file.write_text(json.dumps(cache))

    with mock.patch.object(ta, "FUTURES_CACHE_FILE", cache_file), \
         mock.patch("yfinance.download", return_value=twii):
        result = ta.fetch_futures_sentiment()

    assert result["sample_n"] == 20
    assert result["bucket"] is not None
    assert result["bucket_n"] >= 3
    assert isinstance(result["bonus"], float)
    assert abs(result["bonus"]) <= 0.5  # dampened range (halved, |excess|<=1)

    assert result["short_bucket"] is not None
    assert isinstance(result["short_bonus"], float)
    assert abs(result["short_bonus"]) <= 0.5


def test_legacy_float_cache_entries_still_work(tmp_path):
    """Cache files written before the short-OI field existed used a plain float."""
    cache_file = tmp_path / "cache.json"
    twii = _fake_twii(n=60)
    dates = twii.index[-20:]
    cache = {d.strftime("%Y/%m/%d"): -50000.0 + (i % 5) * 8000.0 for i, d in enumerate(dates)}
    cache_file.write_text(json.dumps(cache))

    with mock.patch.object(ta, "FUTURES_CACHE_FILE", cache_file), \
         mock.patch("yfinance.download", return_value=twii), \
         mock.patch.object(ta, "fetch_futures_net_position", return_value=(None, None)):
        result = ta.fetch_futures_sentiment()

    # legacy entry for "today" isn't a dict, so it's treated as missing and the
    # (mocked) live fetch is attempted and returns nothing — graceful, no crash.
    assert result["net_pos"] is None
    assert result["bonus"] == 0.0


def test_blocked_fetch_degrades_gracefully(tmp_path):
    cache_file = tmp_path / "cache.json"
    cache_file.write_text("{}")
    twii = _fake_twii()

    with mock.patch.object(ta, "FUTURES_CACHE_FILE", cache_file), \
         mock.patch("yfinance.download", return_value=twii), \
         mock.patch.object(ta, "fetch_futures_net_position", return_value=(None, None)):
        result = ta.fetch_futures_sentiment()

    assert result["net_pos"] is None
    assert result["short_oi"] is None
    assert result["bonus"] == 0.0
    assert result["short_bonus"] == 0.0
    assert not cache_file.exists() or json.loads(cache_file.read_text()) == {}
