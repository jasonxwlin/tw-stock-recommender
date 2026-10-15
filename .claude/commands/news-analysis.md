Run a news snapshot on one or more Taiwan stock symbols. Reference only — does not score, does not produce a recommendation. Unlike tech-analysis and fundamental-analysis, the "analysis" step here is you reading the headlines and judging them, not a number the script computes.

## Steps

1. Run the analysis script:
   ```
   python3 news.py $ARGUMENTS
   ```

2. Present the raw headline list (Traditional Chinese via Google News RSS, English via yfinance — both last 7 days). Each headline is followed by its source link in the script output — keep it: render headlines as markdown links `[標題](url)`, and whenever a summary bullet cites a specific headline, attach that link after it.

3. Read the headlines yourself and write a short Traditional Chinese summary per stock:
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

## Disclaimer — always append at the end of every response

After every analysis output (single stock or multi-stock summary), always append the following block verbatim:

---
> **免責聲明**：本分析僅供學術研究與個人參考，不構成任何投資建議或邀約。所有結果均基於歷史技術數據，過去績效不代表未來表現。本工具不考慮個人財務狀況與風險承受能力，投資人應自行判斷並承擔一切投資決策之責任。資料來源為公開 API，不保證即時性與正確性。**請在必要時諮詢合格之證券投資顧問。**
