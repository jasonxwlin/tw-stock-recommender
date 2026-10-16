# AGENTS.md

Shared instructions for any coding agent working in this repo — Claude Code, OpenAI Codex, Gemini CLI, etc.
This file is the **single source of truth** for the stock-analysis workflow; the per-tool command files
(`.claude/commands/`, `.agents/skills/`, `.gemini/commands/`) only point here.

For development details (architecture, tests, coverage gate, pylint), see `CLAUDE.md`.

## Setup

```bash
pip install -r requirements.txt
source .env   # optional: FRED_API_KEY enables CPI/NFP in the macro-event banner; never commit .env
```

## 股票分析流程 (stock analysis workflow)

Triggered by requests like 「分析股票」「分析 台積電 聯發科」 or `/stock-analysis <args>`.

**Arguments**: stock codes or company names, mixed freely (`2330`, `台積電`, `日月光` → 3711).
The scripts resolve names themselves. If the user gives no stocks, reuse the list from earlier in the
conversation; if there is none, ask.

Run the three scripts **one at a time, not in parallel** (TWSE/TPEx rate-limit parallel bursts):

```bash
python3 tech_analysis.py <args>    # 技術面 — the ONLY module that scores and recommends
python3 fundamentals.py <args>     # 基本面 — reference only, no score
python3 news.py <args>             # 消息面 — reference only, no score
```

Read each script's **full** output. Never truncate with `tail` — the 7-day 重大事件 banner
(FOMC/PCE/GDP/CPI/NFP) prints at the very top of `tech_analysis.py`'s output.

### Report layout (Traditional Chinese, in this order)

1. **重大事件** — the macro-event banner, if any; note if prices are intraday (market open 09:00–13:30 Taipei).
2. **市場環境** — TWII vs MA200 regime and its score bonus, VIX, Buffett indicator, 外資期貨 positioning.
3. **技術面（唯一計分）** — per stock: 收盤價, K/RSI, 技術分, 籌碼分, 合計, 建議.
   - Single stock: 3–4 sentence commentary (key ★ signals, institutional flow vs technicals, risk/reward).
   - Multiple stocks: a short 觀察 list of what stands out.
   - Flag data problems honestly: e.g. when 近5日 equals 今日 for most stocks, the TWSE T86 feed
     probably returned only one day — say 籌碼分 is unreliable.
4. **基本面（僅供參考）** — 本益比, 股價淨值比, latest 月營收年增, latest 單季 ROE／毛利率. No recommendation.
5. **消息面（僅供參考）** — per stock, a block that **starts** with the 傾向 label plus one concrete sentence,
   then the cited headlines as markdown links (`news.py` prints each headline's URL under it):

   ```
   **台積電｜中性偏空** — 今天法說，有分析師提出「2 大翻空警訊」；也有人認為長期合理價 5,000 元，現在是科技股中最便宜的。
   - [標題](url)（來源）
   ```

   The sentence must name the concrete news (event, number, who said what), not restate the label.
   傾向 ∈ 利多 / 偏多 / 中性 / 偏空 / 利空 / 多空交雜 — your own reading of the headlines, not a score.
   Read past clickbait titles; say so if Chinese and English coverage diverge.
6. **綜合表** (multiple stocks):
   `股票 | 公司名 | 收盤價 | 技術分 | 籌碼分 | 市場趨勢 | 合計 | 建議 | 本益比 | 最新月營收年增 | 消息面傾向`
   The last three columns are informational only.
7. `> **分析模組**: 技術面 ✅  基本面 ✅（僅供參考，不計分）  消息面 ✅（僅供參考，不計分）`
8. **免責聲明** — the scripts print it as their last line; show it **once**, as the very last thing
   in the report. Don't paraphrase it or add it to instruction files — it lives in `core/report.py` (`DISCLAIMER`).

### Rules

- The 加碼/減碼/持平 recommendation comes **only** from `tech_analysis.py`. Fundamentals and news
  never change or override it — they have no score to merge in.
- When the model's signal conflicts with fundamentals/news (e.g. 強力加碼 purely from oversold
  readings while revenue growth stalls), say so plainly. The backtest window is the past ~6 months;
  call out regimes it hasn't seen (e.g. a macro-driven selloff).
- Not financial advice — no promises about future prices.

### Scoring reference

- **Combined** = tech score + institutional score × 0.12 + regime bonus (+0.25 強多頭 / +0.15 多頭 / 0 中性 / −0.15 空頭) + futures adjustment.
- Thresholds: ≥ +0.60 強力加碼 🚀 · ≥ +0.25 加碼 ↑ · ≤ −0.25 減碼 ↓ (≤ −0.40 in 強多頭) · ≤ −0.60 強力減碼 ⚠ (≤ −0.80 in 強多頭) · otherwise 持平 →.
- Tech score: each active ★ condition contributes its backtested excess edge × sample confidence;
  one winner per indicator group (KD / MACD / RSI / MA5 / MA20 / MA60 / MA200 / Volume / Compound).
  The same pattern can be bullish for one stock and bearish for another.

## Single-module runs

- Technical only: `python3 tech_analysis.py <args>` → sections 1–3 (+ 綜合表 the script prints) + 免責聲明.
- Fundamentals only: `python3 fundamentals.py <args>` → section 4 + 免責聲明.
- News only: `python3 news.py <args>` → section 5 + 免責聲明.
- 3-year strategy backtest: `python3 backtest_annual.py <code-or-name>`.
