Comprehensive Taiwan stock analysis. Orchestrates available sub-analyses and combines them into a unified recommendation.

## Available sub-commands

| Sub-command | What it does |
|-------------|-------------|
| `/tech-analysis` | Technical indicators + institutional flow + market regime → position-sizing signal (this is the only module that scores and recommends) |
| `/fundamental-analysis` | Current valuation (P/E, P/B) + monthly revenue trend + profitability trend — reference only, no score |
| `/news-analysis` | Recent Chinese + English headlines, read and tagged 利多/利空/中性 by you — reference only, no score |

## Steps

1. Parse `$ARGUMENTS` to get stock symbol(s). Codes or company names (e.g. 台積電) both work — the scripts resolve names themselves via `resolve_symbol()`.

2. Run `/tech-analysis $ARGUMENTS` and present its full output. The 加碼/減碼/持平 recommendation comes **only** from this module.

3. Run `/fundamental-analysis $ARGUMENTS` and present its full output directly underneath, as reference context.

4. Run `/news-analysis $ARGUMENTS` and present its full output (headlines + your qualitative summary) directly underneath.

   Neither step 3 nor step 4 changes or overrides the tech-analysis recommendation — they have no score to merge in.

5. If multiple stocks: end with a combined table
   `股票 | 公司名 | 收盤價 | 技術分 | 籌碼分 | 市場趨勢 | 合計 | 建議 | 本益比 | 最新月營收年增 | 消息面傾向`
   where the last three columns come from fundamental-analysis and news-analysis and are informational only — they do not factor into 建議.

6. Note at the end:

   > **分析模組**: 技術面 ✅  基本面 ✅（僅供參考，不計分）  消息面 ✅（僅供參考，不計分）

7. **Always** end the whole response with the 免責聲明 block below, verbatim, as the very last thing — every run, single or multi-stock. The sub-commands each define the same block; when orchestrating, print it **once** here at the bottom instead of after each section.

---
> **免責聲明**：本分析僅供學術研究與個人參考，不構成任何投資建議或邀約。所有結果均基於歷史技術數據，過去績效不代表未來表現。本工具不考慮個人財務狀況與風險承受能力，投資人應自行判斷並承擔一切投資決策之責任。資料來源為公開 API，不保證即時性與正確性。**請在必要時諮詢合格之證券投資顧問。**

## Notes
- Stock codes are Taiwan exchange codes (e.g. 2330=台積電, 0050=台灣50 ETF)
- This command is the entry point; it delegates to sub-commands for each analysis type
- Not financial advice
