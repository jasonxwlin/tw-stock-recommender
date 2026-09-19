# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Load FRED_API_KEY (needed for CPI/NFP in the macro-event banner) — .env is
# gitignored, never commit it. Source it once per shell session before running
# any script below; without it, CPI/NFP are silently skipped (FOMC/PCE/GDP
# don't need a key and always work).
source .env

# Run technical analysis on one or more Taiwan stock codes
python3 tech_analysis.py 2330
python3 tech_analysis.py 2330 2317 0050

# Run fundamental snapshot (current P/E & P/B, revenue trend, profitability trend)
# Reference only — does not score or feed into tech_analysis.py's recommendation
python3 fundamentals.py 2330

# Run news snapshot (recent Chinese + English headlines, for you to read and judge)
# Reference only — no sentiment score, does not feed into tech_analysis.py's recommendation
python3 news.py 2330

# Run 3-year walk-forward backtest for a stock
python3 backtest_annual.py 2330

# Run all self-checks under pytest, with coverage (gated at >=90% via pytest.ini)
# Each file is also directly runnable via `python3 tests/test_x.py` — same
# assert-based tests, no pytest required for that path (network-mocked tests
# that need pytest fixtures print SKIP when run that way; use `pytest` for those)
pytest
```

No build step — `tests/test_*.py` are assert-based self-checks covering both pure logic (parsing, scoring math, condition/backtest engine) and network-facing `fetch_*` functions (mocked at the `requests`/`yfinance` boundary — no real network calls). `tests/test_*_core.py` / `tests/test_*.py` cover pure logic; `tests/test_*_network.py` covers the mocked fetchers, `fmt_report`, and `main()`. Coverage across `tech_analysis.py`/`fundamentals.py`/`news.py` is ~96% and enforced at >=90% by `pytest.ini`. They run under `pytest` (auto-discovered from `tests/`) or standalone via `python3 tests/test_x.py`.

## Architecture

The project is a Taiwan stock technical analysis tool exposed as Claude Code Skills (`/stock-analysis`, `/tech-analysis`, `/fundamental-analysis`, `/news-analysis`). Skills are defined in `.claude/commands/`. `/stock-analysis` is the orchestrator: it runs `/tech-analysis` (which produces the only scored 加碼/減碼/持平 recommendation), `/fundamental-analysis`, and `/news-analysis` (both reference-only, no score) and presents all three.

### `tech_analysis.py` — core engine

`tech_analysis.py` is the orchestrator and entry point (`main()`); the pipeline
stages below are split across sibling modules to stay under pylint's
`too-many-lines`, and `tech_analysis.py` re-imports every public/test-facing
name from them (via `__all__`) so `from tech_analysis import X` / `tech_analysis.X`
keeps working unchanged: `fetchers.py` (price/company/earnings data + indicator
calculation), `conditions.py` (`build_conditions()`), `backtest_engine.py`
(`compute_outcomes()`/`backtest_conditions()`/`baseline_stats()`), `market_indicators.py`
(`fetch_vix()`/`fetch_buffett_indicator()`), `report.py` (`fmt_report()`/`fmt_summary_table()`).
Institutional-investor fetching/scoring, futures positioning, macro events,
`analyze()`, and `main()` stay in `tech_analysis.py` itself.

Data flows in this order:

1. **Fetch** — `fetch_price_data()` pulls 400 days of OHLCV from Yahoo Finance (tries `.TW` then `.TWO`). `fetch_institutional()` fetches institutional investor (三大法人) data from TWSE/TPEx APIs in parallel via `ThreadPoolExecutor`. `fetch_market_regime()` pulls `^TWII` vs its MA200 to classify the market as 強多頭 / 多頭 / 中性 / 空頭.

2. **Indicators** — `calc_indicators()` computes KD, RSI, MACD, MAs (5/20/60/200), volume metrics via `ta` library.

3. **Conditions** — `build_conditions()` defines ~50 named boolean Series (one per technical condition). Each condition maps to a group (`_group_of()`) — only the strongest condition per group is scored, preventing double-counting correlated signals.

4. **Backtest** — `backtest_conditions()` runs a walk-forward backtest over the past ~125 trading days. For each condition it computes the historical up/down rates and "excess margin" vs baseline (= condition's net edge minus the stock's natural drift). This excess margin is the scoring weight.

5. **Score** — `analyze()` aggregates: technical score (sum of excess margins weighted by confidence) + institutional score (fixed rules: foreign/trust consecutive buys/sells) × 0.12 + market regime adjustment (+0.25 in 強多頭, down to -0.15 in 空頭). Thresholds: ≥+0.60 強力加碼, ≥+0.25 加碼, ≤-0.25 減碼 (≤-0.40 in 強多頭), ≤-0.60 強力減碼 (≤-0.80 in 強多頭).

6. **Output** — `fmt_report()` formats the full report including indicator snapshot, institutional breakdown, condition backtest table, and final recommendation. When multiple symbols are passed, `main()` also prints `fmt_summary_table()` at the end — a comparison table pre-sorted 強力加碼 → 加碼 → 持平 → 減碼 → 強力減碼 (ties broken by combined score, descending), via the shared `_REC_RANK`/`_REC_ICON` constants.

`fetch_macro_events()` is a separate, informational-only overlay: scrapes FOMC (federalreserve.gov), PCE/GDP (bea.gov), and — if `FRED_API_KEY` is set — CPI/NFP (FRED API) for events in the next 7 days, printed once as a warning banner before the per-stock loop. `fetch_stock_earnings_date()` does the same per-stock via yfinance's earnings calendar. Neither affects scoring.

### `fundamentals.py` — fundamental snapshot (reference only)

Independent module, imports `fetch_price_data()`/`fetch_company_name()` from `tech_analysis.py` for ticker resolution but does not feed back into it — no score, no backtest, no recommendation.

- **Valuation** — `fetch_valuation_snapshot()` returns current P/E and P/B only, no percentile (a partially-backfilled window would make a percentile misleading). Internally it targets a trailing 12-month window (TWSE: `BWIBBU`, one request per month; TPEx: `pera_result.php`, one whole-market request per sampled day since there's no per-stock-per-month endpoint), but only fetches **one new past month per call** plus the always-fresh current month — see the caching note below for why.
- **Revenue** — `fetch_monthly_revenue()` scrapes MOPS's historical monthly revenue report (`t21sc03_<roc_year>_<month>_0.html`, Big5-encoded, one request per month) for trailing YoY growth. Skips the current calendar month (not yet published — MOPS releases ~10 days after month end).
- **Profitability** — `fetch_quality_trend()` pulls trailing 4 quarters from yfinance (`quarterly_financials`/`quarterly_balance_sheet`) for single-quarter ROE (not annualized) and gross margin.

All three fetches fail independently and silently (same pattern as `tech_analysis.py`'s `fetch_macro_events()`) — a missing section just prints "資料無法取得" rather than aborting the run.

Valuation months are cached to `valuation_cache.json` (same pattern as `taifex_net_pos_cache.json`) — a past month never changes once published so it's fetched once and kept forever; only the current (still-accumulating) month is re-fetched every call. TWSE's `BWIBBU` endpoint has anti-flood protection (HTTP 428) that a cold multi-month, multi-stock fetch trips almost immediately, so rather than bursting through the full 12-month target in one run, only one new past month is fetched per call — the rest of the window backfills gradually over ~12 repeated invocations.

### `news.py` — news snapshot (reference only)

Also independent, also imports ticker resolution from `tech_analysis.py`. Unlike the other two modules, it does no analysis itself — it only fetches and returns headlines; the qualitative summary and 利多/利空/中性 tagging happens in the `/news-analysis` skill (i.e. read by Claude, not computed).

- `fetch_google_news()` — Traditional Chinese headlines from Google News RSS (public feed, queried by company name), trailing 7 days.
- `fetch_yahoo_news()` — English headlines from yfinance's `Ticker.news`, trailing 7 days, mostly US/global-investor angle.

Both parsing functions (`_parse_google_news_xml`, `_parse_yahoo_news_items`) are separated from their network calls so they're unit-testable (see `tests/test_news.py`); both fail independently and silently like the other modules.

### `backtest_annual.py` — strategy simulation

Simulates 3 years of daily position management using the tech score only (no institutional data, which is hard to back-fill). Uses a 5-level position ladder (0/25/50/75/100%). Walk-forward: each day only uses the prior 125 trading days for signal computation. Accounts for trading costs (0.1425% commission + 0.3% securities tax on sells).

### Key constants (top of `tech_analysis.py`)

- `MOVE_THRESH` — price move threshold (%) for classifying up/down outcomes in backtest (default 1.5%, raised to 2% in 強多頭)
- `LOOKBACK` — days of history for per-stock signal backtest (~125 trading days)
- Regime thresholds: TWII/MA200 ≥1.05 → 強多頭, ≥1.00 → 多頭, ≥0.95 → 中性, <0.95 → 空頭
