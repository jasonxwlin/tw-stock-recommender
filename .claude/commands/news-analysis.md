Run a news snapshot on one or more Taiwan stock symbols. Reference only — does not score, does not produce a recommendation. Unlike tech-analysis and fundamental-analysis, the "analysis" step here is you reading the headlines and judging them, not a number the script computes.

## Steps

1. Run the analysis script:
   ```
   python3 news.py $ARGUMENTS
   ```

2. Present the raw headline list (Traditional Chinese via Google News RSS, English via yfinance — both last 7 days). Each headline is followed by its source link in the script output — keep it: render headlines as markdown links `[標題](url)`, and whenever a summary bullet cites a specific headline, attach that link after it.

3. Read the headlines yourself and write a short Traditional Chinese summary per stock. Each stock's block **starts** with its 傾向 label plus a one-sentence summary on the same line, then the linked headlines underneath, e.g.:

   **台積電｜中性偏空** — 今天法說，有分析師提出「2 大翻空警訊」；也有人認為長期合理價 5,000 元，現在是科技股中最便宜的。
   - [標題](url)（來源）

   The summary sentence should name the concrete news (event, number, who said what), not just restate the label. Then:
   - 2–4 bullet points on the main themes (e.g. 法說會/財報、供應鏈動態、政策消息、大戶動向、總經連動)
   - Tag each theme 利多 / 利空 / 中性 based on your own reading — this is a qualitative judgment call, not a formula
   - If Chinese and English coverage diverge (e.g. Taiwan media focuses on local expansion news while English coverage is about a US analyst downgrade), say so explicitly
   - Do **not** issue a 加碼/減碼/持平-style recommendation — this module is reference only, same as fundamental-analysis

4. If multiple stocks: end with a short table
   `股票 | 公司名 | 消息面重點（1行）| 傾向`
   where 傾向 is 利多/利空/中性/多空交雂 — still just a label for the reader, not a score.

## Notes

- Stock codes are Taiwan exchange codes (e.g. 2330=台積電)
- Script auto-tries `.TW` (TWSE) then `.TWO` (TPEx), reusing `tech_analysis.py`'s ticker resolution
- Chinese headlines come from Google News RSS (public feed, queried by company name); English headlines from yfinance's news API. Either can return empty ("無資料") if the source is temporarily unavailable — this is expected, not an error.
- Headline titles alone can be misleading (clickbait, missing context) — read the actual gist before tagging 利多/利空, and note if a headline needs the full article to interpret correctly.
- No scoring, no backtest, no recommendation — this never feeds into `/tech-analysis`'s combined score.
