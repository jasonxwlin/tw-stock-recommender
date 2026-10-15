Run a fundamental data snapshot on one or more Taiwan stock symbols. Reference only — does not score or produce a recommendation.

## Steps

1. Run the analysis script:
   ```
   python3 fundamentals.py $ARGUMENTS
   ```

2. Present the raw output with these sections explained:

   **估值** — Current P/E and P/B, shown as raw numbers (no percentile — the historical window builds up gradually in the background, see Notes).

   **月營收年增率趨勢** — Trailing ~5 months of YoY monthly revenue growth (公開資訊觀測站/MOPS), oldest first. Watch for acceleration, deceleration, or a sign flip.

   **獲利能力趨勢** — Trailing 4 quarters of single-quarter ROE (net income ÷ quarter-end equity, not annualized) and gross margin. A trend indicator across quarters, not a headline annual ROE figure.

3. Write a 2–3 sentence Traditional Chinese summary of what the numbers show — valuation richness/cheapness, revenue momentum direction, profitability trajectory. Do **not** issue a 加碼/減碼/持平-style recommendation here — this module is reference only.

4. If multiple stocks: end with a snapshot table
   `股票 | 公司名 | 本益比 | 股價淨值比 | 最新月營收年增 | 最新單季ROE | 最新單季毛利率`

## Notes

- Stock codes are Taiwan exchange codes (e.g. 2330=台積電)
- Script auto-tries `.TW` (TWSE) then `.TWO` (TPEx), reusing `tech_analysis.py`'s ticker resolution
- No scoring, no backtest, no recommendation — this is a data snapshot only. It never feeds into `/tech-analysis`'s combined score or recommendation.
- Any of the three sections may be missing ("資料無法取得") if its data source is temporarily unavailable — each fetch fails independently and silently, same pattern as the macro event reminder in `/tech-analysis`.
- P/E and P/B are shown as raw current values, not a percentile — TWSE's `BWIBBU` endpoint has anti-flood protection, so instead of bursting through a full year of history in one run, only one new past month is cached per run (plus the current month, always refreshed). A trailing-year percentile isn't meaningful until the cache backfills over ~12 runs; asking for one sooner would just be misleading.
