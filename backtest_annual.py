#!/usr/bin/env python3
"""
Walk-Forward 3-Year Backtest with Position Sizing
模擬過去 3 年，每天依據 stock-analysis skill 的加碼/持平/減碼建議調整倉位。

倉位模型（5 個等級）：
  強力加碼 → +2 級（上限 100%）
  加碼     → +1 級
  持平     →  0  （不動）
  減碼     → -1 級
  強力減碼 → -2 級（下限 0%）
  起始倉位 → 50%（中性）

交易成本：
  買進手續費 0.1425%
  賣出手續費 0.1425% + 證交稅 0.3%
"""

from __future__ import annotations
import argparse
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# yfinance (and tech_analysis, which imports requests too) transitively
# triggers urllib3's NotOpenSSLWarning at import time; filterwarnings must
# run before importing either to suppress it, which unavoidably breaks
# import-block contiguity for these lines only. tech_analysis.py lives in
# the same directory as this script, which Python already puts on sys.path
# when running `python3 backtest_annual.py` directly — no manual path hack
# needed.
warnings.filterwarnings('ignore')
import yfinance as yf  # pylint: disable=wrong-import-position
from tech_analysis import (  # pylint: disable=wrong-import-position
    calc_indicators,
    build_conditions,
    compute_outcomes,
    backtest_conditions,
    baseline_stats,
)

# ─── 常數 ─────────────────────────────────────────────────────────────────────
LOOKBACK_DAYS = 125          # 每次回測用的歷史窗口（約 6 個月）
MIN_WINDOW    = 40           # 最少需要這麼多天才發出信號

COMM_BUY  = 0.001425
COMM_SELL = 0.001425
TAX_SELL  = 0.003

# 倉位等級：0%/25%/50%/75%/100%
LEVELS = [0.0, 0.25, 0.50, 0.75, 1.00]
START_LEVEL = 2   # 50%

LEVEL_DELTA = {
    '強力加碼': +2,
    '加碼':     +1,
    '持平':      0,
    '減碼':     -1,
    '強力減碼': -2,
}


# ─── 數據抓取 ──────────────────────────────────────────────────────────────────

def fetch_twii_regime(years: float = 3.6, end_date: datetime | None = None) -> pd.Series:
    """
    Returns a daily Series of regime bonus values indexed by date.
      TWII / MA200 ≥ 1.05  → +0.25  (強多頭)
      TWII / MA200 ≥ 1.00  → +0.15  (多頭)
      TWII / MA200 ≥ 0.95  →  0.00  (中性)
      TWII / MA200 <  0.95 → -0.15  (空頭)
    Fetches extra history so MA200 is valid from the start of the sim period.
    """
    end   = end_date or datetime.now()
    start = end - timedelta(days=int(years * 365.25) + 250)
    df = yf.download('^TWII', start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df['Close']
    close.index = pd.to_datetime(close.index).tz_localize(None)
    ma200 = close.rolling(200).mean()
    ratio = close / ma200
    bonus = pd.Series(0.0, index=close.index)
    bonus[ratio >= 1.05] = 0.25
    bonus[(ratio >= 1.00) & (ratio < 1.05)] = 0.15
    bonus[(ratio >= 0.95) & (ratio < 1.00)] = 0.0
    bonus[ratio < 0.95]  = -0.15
    bonus[ma200.isna()]  = 0.0   # not enough history yet
    return bonus


def fetch_extended(symbol: str, years: float = 3.5, end_date: datetime | None = None) -> tuple[pd.DataFrame, str]:
    """抓取 years 年的歷史數據。"""
    end   = end_date or datetime.now()
    start = end - timedelta(days=int(years * 365.25))
    for suffix in ['.TW', '.TWO']:
        ticker = f'{symbol}{suffix}'
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df, ticker
    raise ValueError(f'找不到股票 {symbol}')


# ─── 信號計算 ──────────────────────────────────────────────────────────────────

STRONG_BULL_BONUS = 0.25

# Per-regime thresholds: (add, reduce, strong_reduce, move_thresh, start_level)
# 強多頭-only tuning is the empirically best approach:
# - Tighten sell threshold in bull markets → avoid false exits during trends
# - Bear/neutral markets: keep original thresholds to prevent whipsaw on noisy signals
REGIME_PARAMS: dict[float, dict] = {
     0.25: {"add": 0.25, "reduce": -0.40, "strong": -0.80, "move": 0.020, "start": 3},  # 強多頭 → 75%
     0.15: {"add": 0.25, "reduce": -0.25, "strong": -0.60, "move": 0.015, "start": 2},  # 多頭   → 50%
     0.00: {"add": 0.25, "reduce": -0.25, "strong": -0.60, "move": 0.015, "start": 2},  # 中性   → 50%
    -0.15: {"add": 0.25, "reduce": -0.25, "strong": -0.60, "move": 0.015, "start": 2},  # 空頭   → 50%
}
_DEFAULT_PARAMS = REGIME_PARAMS[0.00]


def _tier_from_score(score: float, p: dict) -> str:
    """Map a combined score to its recommendation tier under regime params `p`."""
    if score >= 0.60:
        return '強力加碼'
    if score >= p['add']:
        return '加碼'
    if score <= p['strong']:
        return '強力減碼'
    if score <= p['reduce']:
        return '減碼'
    return '持平'


def get_signal(window: pd.DataFrame, regime_bonus: float = 0.0) -> tuple[str, float]:
    """完全複製 tech_analysis.py 技術面邏輯（不含籌碼面），含市場多空調整。"""
    if len(window) < MIN_WINDOW:
        return '持平', 0.0
    try:
        p = REGIME_PARAMS.get(regime_bonus, _DEFAULT_PARAMS)

        conditions = build_conditions(window)
        outcomes   = compute_outcomes(window, move_thresh=p['move'])
        base       = baseline_stats(outcomes)
        bt         = backtest_conditions(conditions, outcomes, base)

        today_flags = {name: bool(cond.iloc[-1]) for name, cond in conditions.items()}

        group_best: dict[str, dict] = {}
        for name, stats in bt.items():
            if not today_flags.get(name, False):
                continue
            grp = stats['group']
            if grp not in group_best or abs(stats['weight']) > abs(group_best[grp]['weight']):
                group_best[grp] = stats

        score = sum(s['weight'] for s in group_best.values()) + regime_bonus
        return _tier_from_score(score, p), score
    except Exception:
        return '持平', 0.0


# ─── 倉位管理 ──────────────────────────────────────────────────────────────────

def rebalance(
    cash: float, shares: float, price: float, target_pct: float
) -> tuple[float, float, float]:
    """
    重新平衡到目標倉位比例。
    回傳 (new_cash, new_shares, trade_cost)。
    """
    if price <= 0:
        return cash, shares, 0.0

    portfolio = cash + shares * price
    target_stock = portfolio * target_pct
    current_stock = shares * price
    delta = target_stock - current_stock

    if delta > portfolio * 0.01:          # 買入（差距 > 1% 才執行，避免頻繁小額交易）
        buy_value = min(delta, cash / (1 + COMM_BUY))
        new_shares  = buy_value / price
        cost        = buy_value * COMM_BUY
        cash       -= buy_value + cost
        shares     += new_shares
        return cash, shares, cost

    if delta < -portfolio * 0.01:       # 賣出
        sell_shares = min(shares, -delta / price)
        gross       = sell_shares * price
        fee         = gross * (COMM_SELL + TAX_SELL)
        cash       += gross - fee
        shares     -= sell_shares
        return cash, shares, fee

    return cash, shares, 0.0


# ─── 交易模擬 ──────────────────────────────────────────────────────────────────

def _apply_signal_rebalance(cash: float, shares: float, level: int, price: float, prev_rec: str) -> tuple:
    """Adjust position toward the level implied by yesterday's signal. Returns
    (cash, shares, level, cost, log_entry_or_None)."""
    delta     = LEVEL_DELTA.get(prev_rec, 0)
    new_level = max(0, min(len(LEVELS) - 1, level + delta))
    if new_level == level:
        return cash, shares, level, 0.0, None

    old_pct = LEVELS[level]
    new_pct = LEVELS[new_level]
    cash, shares, cost = rebalance(cash, shares, price, new_pct)
    log_entry = {
        'signal':    prev_rec,
        'old_level': old_pct,
        'new_level': new_pct,
        'price':     price,
        'cost':      cost,
    }
    return cash, shares, new_level, cost, log_entry


def simulate(days: list[dict], start_level: int = START_LEVEL) -> dict:
    """
    倉位分級策略模擬。
    days: [{'date', 'price', 'rec', 'score'}, ...]
    """
    n = len(days)
    # 從指定倉位開始（強多頭用 75%）— bundled in one dict to stay under the
    # local-variable budget for a function this size.
    state = {'cash': 1_000_000.0, 'shares': 0.0, 'level': start_level, 'total_cost': 0.0}

    portfolio_arr = np.zeros(n)
    position_arr  = np.zeros(n)
    level_arr     = np.zeros(n, dtype=int)
    rebalance_log: list[dict] = []

    # 第一天：依起始倉位建立初始部位
    if n > 0:
        p0 = days[0]['price']
        state['cash'], state['shares'], cost = rebalance(
            state['cash'], state['shares'], p0, LEVELS[state['level']]
        )
        state['total_cost'] += cost

    for i, day in enumerate(days):
        price = day['price']

        # 依前一天的信號調整目標倉位等級（第 0 天用起始等級）
        if i > 0:
            state['cash'], state['shares'], state['level'], cost, log_entry = _apply_signal_rebalance(
                state['cash'], state['shares'], state['level'], price, days[i - 1]['rec']
            )
            state['total_cost'] += cost
            if log_entry is not None:
                log_entry['date'] = day['date']
                rebalance_log.append(log_entry)

        portfolio_arr[i] = state['cash'] + state['shares'] * price
        position_arr[i]  = (state['shares'] * price) / portfolio_arr[i] if portfolio_arr[i] > 0 else 0
        level_arr[i]     = state['level']

    return {
        'portfolio':     portfolio_arr,
        'position':      position_arr,
        'level':         level_arr,
        'total_cost':    state['total_cost'],
        'rebalances':    rebalance_log,
    }


# ─── 績效計算 ──────────────────────────────────────────────────────────────────

def _max_drawdown(arr: np.ndarray) -> float:
    peak = np.maximum.accumulate(arr)
    return float(((arr - peak) / peak).min())


def _segment_win_rate(rebs: list[dict]) -> float:
    """勝率：每筆換倉後到下次換倉期間的報酬。"""
    seg_rets: list[float] = []
    if len(rebs) >= 2:
        for j in range(len(rebs) - 1):
            r0 = rebs[j]['price']
            r1 = rebs[j + 1]['price']
            direction = rebs[j]['new_level'] - rebs[j]['old_level']
            if direction > 0:               # 加碼後看漲
                seg_rets.append(r1 / r0 - 1)
            elif direction < 0:             # 減碼後看跌（減碼對空才算贏）
                seg_rets.append(r0 / r1 - 1)

    wins = [r for r in seg_rets if r > 0]
    return len(wins) / len(seg_rets) if seg_rets else 0.0


def calc_metrics(sim: dict, prices: np.ndarray) -> dict:
    """Derive return/Sharpe/drawdown/turnover stats from a simulate() result."""
    port = sim['portfolio']
    pos  = sim['position']

    strat_ret = port[-1] / port[0] - 1
    bah_ret   = prices[-1] / prices[0] - 1
    alpha     = strat_ret - bah_ret
    n_years   = len(port) / 252
    annual    = (1 + strat_ret) ** (1 / n_years) - 1 if n_years > 0 else strat_ret

    bah_port = prices / prices[0] * port[0]

    daily  = np.diff(port) / port[:-1]
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0

    rebs = sim['rebalances']

    return {
        'strat_ret':  strat_ret,
        'bah_ret':    bah_ret,
        'alpha':      alpha,
        'annual':     annual,
        'max_dd':     _max_drawdown(port),
        'bah_max_dd': _max_drawdown(bah_port),
        'sharpe':     sharpe,
        'n_rebal':    len(rebs),
        'total_cost': sim['total_cost'],
        'win_rate':   _segment_win_rate(rebs),
        'avg_pos':    float(pos.mean()),
        'portfolio':  port,
        'bah_port':   bah_port,
        'rebalances': rebs,
    }


# ─── 圖表 & 輸出 ───────────────────────────────────────────────────────────────

def _pct(v: float, d: int = 2) -> str:
    return f"{v*100:>+{d+5}.{d}f}%"

def _sign(v: float) -> str:
    return '▲' if v > 0 else ('▼' if v < 0 else '─')


def _chart_series(port: np.ndarray, bah: np.ndarray, pos: np.ndarray, W: int, step: int) -> tuple:
    s_raw = (port / port[0] * 100)[::step][:W]
    b_raw = (bah  / bah[0]  * 100)[::step][:W]
    p_raw = (pos * 100)[::step][:W]      # 倉位 %
    return s_raw, b_raw, p_raw


def _chart_y_range(s_raw: np.ndarray, b_raw: np.ndarray) -> tuple[float, float]:
    y_lo = min(s_raw.min(), b_raw.min()) * 0.98
    y_hi = max(s_raw.max(), b_raw.max()) * 1.02
    return y_hi, max(y_hi - y_lo, 0.01)


def _build_chart_grid(
    s_raw: np.ndarray, b_raw: np.ndarray, p_raw: np.ndarray, dims: tuple[int, int], y_range: tuple[float, float]
) -> list[list[str]]:
    W, H = dims
    y_hi, rng = y_range

    def row(v):
        return max(0, min(H - 2, int((y_hi - v) / rng * (H - 2))))

    grid = [[' '] * W for _ in range(H)]
    # 倉位條（最下兩行）
    for x in range(min(W, len(p_raw))):
        pv = p_raw[x]
        if pv >= 75:
            grid[H-1][x] = '▓'
        elif pv >= 50:
            grid[H-1][x] = '▒'
        elif pv >= 25:
            grid[H-1][x] = '░'
        else:
            grid[H-1][x] = '·'

    for x in range(min(W, len(s_raw))):
        rb = row(b_raw[x])
        rs = row(s_raw[x])
        if 0 <= rb < H-1:
            grid[rb][x] = '·'
        if 0 <= rs < H-1:
            grid[rs][x] = '█' if s_raw[x] >= b_raw[x] else '░'
    return grid


def _print_chart_grid(grid: list[list[str]], W: int, H: int, y_hi: float, rng: float) -> None:
    print(f"  ┌{'─'*W}┐")
    for ri, r in enumerate(grid):
        if ri == H - 1:
            print(f"  │{''.join(r)}│ 倉%")
        else:
            v = y_hi - ri / (H - 2) * rng
            print(f"  │{''.join(r)}│{v:5.0f}")
    print(f"  └{'─'*W}┘")


def _print_chart_xaxis(dates: list, W: int, n: int) -> None:
    d0 = pd.Timestamp(dates[0]).strftime('%Y/%m')
    dm = pd.Timestamp(dates[n // 2]).strftime('%Y/%m')
    de = pd.Timestamp(dates[-1]).strftime('%Y/%m')
    pad1 = (W - len(d0) - len(dm)) // 2
    pad2 = W - len(d0) - pad1 - len(dm) - len(de)
    print(f"   {d0}{' '*pad1}{dm}{' '*max(0,pad2)}{de}")


def print_chart(port: np.ndarray, bah: np.ndarray, pos: np.ndarray, dates: list):
    """ASCII chart of strategy vs buy-and-hold value, with a position-size color strip."""
    W, H = 62, 14
    n    = len(port)
    step = max(1, n // W)

    s_raw, b_raw, p_raw = _chart_series(port, bah, pos, W, step)
    y_hi, rng = _chart_y_range(s_raw, b_raw)
    grid = _build_chart_grid(s_raw, b_raw, p_raw, (W, H), (y_hi, rng))

    print("\n  策略走勢 vs 買進持有（基準=100）")
    print("  █/░ 策略   · 買進持有   底部色塊=倉位（▓≥75% ▒≥50% ░≥25%）")
    _print_chart_grid(grid, W, H, y_hi, rng)
    _print_chart_xaxis(dates, W, n)


def print_rebal_log(rebs: list[dict], max_show: int = 15):
    """Print the most recent `max_show` position-change events."""
    if not rebs:
        print("  （無換倉紀錄）")
        return
    show = rebs[-max_show:]
    print(f"\n  {'日期':12} {'信號':6} {'舊倉位':>6} {'新倉位':>6} {'執行價':>8} {'手續費':>8}")
    print("  " + "─" * 50)
    for r in show:
        print(
            f"  {str(pd.Timestamp(r['date']))[:10]:12}"
            f" {r['signal']:6}"
            f" {r['old_level']*100:>5.0f}%"
            f" → {r['new_level']*100:>4.0f}%"
            f" {r['price']:>8.1f}"
            f" {r['cost']:>8.1f}"
        )
    if len(rebs) > max_show:
        print(f"  ...（共 {len(rebs)} 次換倉，僅顯示最後 {max_show} 筆）")


def print_monthly(days: list[dict], port: np.ndarray, pos_arr: np.ndarray):
    """Print month-by-month portfolio return and average position size."""
    df = pd.DataFrame(days)
    df['portfolio'] = port
    df['position']  = pos_arr
    df['month']     = pd.to_datetime([d['date'] for d in days]).to_period('M')
    df['year']      = df['month'].apply(lambda m: m.year)

    print(f"\n  {'月份':^8}  {'策略':>8}  {'持有':>8}  {'超額':>8}  {'平均倉%':>7}  {'信號分佈':}")
    print("  " + "─" * 68)

    for month, grp in df.groupby('month'):
        p_ret = grp['portfolio'].iloc[-1] / grp['portfolio'].iloc[0] - 1
        s_ret = grp['price'].iloc[-1]     / grp['price'].iloc[0]     - 1
        alpha = p_ret - s_ret
        avg_p = grp['position'].mean() * 100

        # Signal distribution this month
        rc = grp['rec'].value_counts()
        sig_str = ' '.join(f"{v}:{rc.get(v,0)}" for v in ['強力加碼','加碼','持平','減碼','強力減碼'] if rc.get(v,0)>0)

        star = ' ★' if p_ret > s_ret else '  '
        print(
            f"  {str(month):^8}  "
            f"{_sign(p_ret)}{_pct(p_ret,1):>7}  "
            f"{_sign(s_ret)}{_pct(s_ret,1):>7}  "
            f"{_pct(alpha,1):>8}{star}  "
            f"{avg_p:>5.0f}%  "
            f"{sig_str}"
        )


def print_yearly(days: list[dict], port: np.ndarray):
    """Print year-by-year portfolio return."""
    df = pd.DataFrame(days)
    df['portfolio'] = port
    df['year']      = pd.to_datetime([d['date'] for d in days]).year

    print(f"\n  {'年份':^6}  {'策略年報酬':>10}  {'買進持有':>10}  {'超額':>8}")
    print("  " + "─" * 40)
    for year, grp in df.groupby('year'):
        p_ret = grp['portfolio'].iloc[-1] / grp['portfolio'].iloc[0] - 1
        s_ret = grp['price'].iloc[-1]     / grp['price'].iloc[0]     - 1
        alpha = p_ret - s_ret
        star  = ' ★' if p_ret > s_ret else '  '
        print(
            f"  {year:^6}  "
            f"{_sign(p_ret)}{_pct(p_ret,1):>9}  "
            f"{_sign(s_ret)}{_pct(s_ret,1):>9}  "
            f"{_pct(alpha,1):>8}{star}"
        )


# ─── 主程式 ────────────────────────────────────────────────────────────────────

def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('symbol', nargs='?', default='2330')
    parser.add_argument('--start', default=None, help='回測起始日 YYYY-MM-DD')
    parser.add_argument('--end',   default=None, help='回測結束日 YYYY-MM-DD')
    return parser.parse_args()


def _resolve_sim_window(args: argparse.Namespace) -> tuple:
    symbol       = args.symbol.upper()
    sim_end_dt   = datetime.strptime(args.end,   '%Y-%m-%d') if args.end   else datetime.now()
    sim_start_dt = datetime.strptime(args.start, '%Y-%m-%d') if args.start else None
    span_years   = ((sim_end_dt - (sim_start_dt or (sim_end_dt - timedelta(days=365*3)))).days / 365.25) + 0.6
    return symbol, sim_end_dt, sim_start_dt, span_years


def _print_run_header(symbol: str, args: argparse.Namespace, W: int) -> None:
    period_label = f"{args.start} → {args.end}" if args.start or args.end else "3 年"
    print(f"\n{'='*W}")
    print(f"  {symbol}  Walk-Forward 回測（倉位加碼/減碼策略）  [{period_label}]")
    print("  倉位等級：0% / 25% / 50% / 75% / 100%  起始：依市場趨勢（強多頭→75% 其他→50%）")
    print(f"  交易成本：買 {COMM_BUY*100:.4f}%  賣 {(COMM_SELL+TAX_SELL)*100:.4f}%")
    print("  * 不含三大法人籌碼（歷史籌碼 API 難以批量回溯）")
    print(f"{'='*W}")


def _find_sim_start_index(all_dates: list, target_start: datetime) -> int:
    for i, d in enumerate(all_dates):
        if pd.Timestamp(d) >= pd.Timestamp(target_start):
            return max(i, LOOKBACK_DAYS)
    return LOOKBACK_DAYS


def _regime_bonus_for_date(twii_regime: pd.Series, date_ts: pd.Timestamp) -> float:
    try:
        regime_bonus = float(twii_regime.asof(date_ts)) if not twii_regime.empty else 0.0
        return 0.0 if pd.isna(regime_bonus) else regime_bonus
    except Exception:
        return 0.0


def _compute_daily_signals(
    df_full: pd.DataFrame, all_dates: list, twii_regime: pd.Series, sim_start: int
) -> list[dict]:
    sim_n = len(all_dates) - sim_start
    print(f"正在計算每日信號（共 {sim_n} 天）...", end='', flush=True)
    days: list[dict] = []
    for i in range(sim_start, len(all_dates)):
        window = df_full.iloc[max(0, i - LOOKBACK_DAYS) : i + 1]
        price  = float(df_full['Close'].iloc[i])
        date_ts = pd.Timestamp(all_dates[i]).tz_localize(None)
        regime_bonus = _regime_bonus_for_date(twii_regime, date_ts)

        rec, score = get_signal(window, regime_bonus=regime_bonus)
        days.append({'date': all_dates[i], 'price': price, 'rec': rec, 'score': score, 'regime_bonus': regime_bonus})
        if (i - sim_start + 1) % 50 == 0:
            print('.', end='', flush=True)
    print(' 完成！')
    return days


def _load_backtest_context(symbol: str, span_years: float, sim_end_dt: datetime, sim_start_dt: datetime | None) -> dict:
    """Fetch price/regime history, pick the simulation start index, and walk
    forward the daily signal computation. Returns {'days', 'prices'}."""
    print(f"\n正在下載 {symbol} 歷史數據...", flush=True)
    df_full, _ = fetch_extended(symbol, years=span_years, end_date=sim_end_dt)
    df_full = calc_indicators(df_full)
    all_dates = list(df_full.index)

    print("正在下載 ^TWII 市場趨勢數據...", flush=True)
    twii_regime = fetch_twii_regime(years=span_years, end_date=sim_end_dt)

    target_start = sim_start_dt if sim_start_dt else (sim_end_dt - timedelta(days=365 * 3))
    sim_start = _find_sim_start_index(all_dates, target_start)

    print(f"模擬期間:  {all_dates[sim_start].date()} → {all_dates[-1].date()}")
    print(f"交易日數:  {len(all_dates) - sim_start} 天（回測窗口 {LOOKBACK_DAYS} 天）\n")

    days = _compute_daily_signals(df_full, all_dates, twii_regime, sim_start)
    return {'days': days, 'prices': np.array([d['price'] for d in days])}


def _print_regime_distribution(days: list[dict], sim_n: int, W: int) -> None:
    print(f"\n{'─'*W}")
    regime_map = {0.25: '強多頭', 0.15: '多頭', 0.0: '中性', -0.15: '空頭'}
    regime_cnt: dict[str, int] = {}
    for d in days:
        label = regime_map.get(d.get('regime_bonus', 0.0), '中性')
        regime_cnt[label] = regime_cnt.get(label, 0) + 1
    print(f"  台股趨勢分佈（TWII vs MA200，共 {sim_n} 天）")
    for lbl in ['強多頭', '多頭', '中性', '空頭']:
        cnt = regime_cnt.get(lbl, 0)
        pct = cnt / sim_n * 100
        print(f"  {lbl:4s}  {cnt:4d}天 ({pct:5.1f}%)")


def _print_signal_distribution(days: list[dict], sim_n: int, W: int) -> None:
    print(f"\n{'─'*W}")
    print(f"  信號分佈（共 {sim_n} 天）")
    rec_cnt: dict[str, int] = {}
    for d in days:
        rec_cnt[d['rec']] = rec_cnt.get(d['rec'], 0) + 1
    for lbl in ['強力加碼', '加碼', '持平', '減碼', '強力減碼']:
        cnt = rec_cnt.get(lbl, 0)
        pct = cnt / sim_n * 100
        bar_str = '█' * max(1, int(pct / 2))
        print(f"  {lbl:6s}  {cnt:4d}天 ({pct:5.1f}%) {bar_str}")


def _arrow(v: float) -> str:
    return '🚀' if v > 0.10 else ('📈' if v > 0 else ('📉' if v < -0.10 else '➡️'))


def _print_performance_summary(m: dict, W: int) -> None:
    print(f"\n{'─'*W}")
    print("  績效摘要")
    print(f"{'─'*W}")
    print(f"  策略總報酬:   {_pct(m['strat_ret']):>10}   {_arrow(m['strat_ret'])}")
    print(f"  買進持有報酬: {_pct(m['bah_ret']):>10}")
    print(f"  超額報酬:     {_pct(m['alpha']):>10}   {'（策略勝）' if m['alpha'] > 0 else '（持有勝）'}")
    print(f"  年化報酬率:   {_pct(m['annual']):>10}")
    print(f"  Sharpe Ratio: {m['sharpe']:>10.2f}")
    print()
    print(f"  換倉次數:     {m['n_rebal']:>6} 次    總手續費: NT${m['total_cost']:,.0f}")
    print(f"  平均持倉比:   {m['avg_pos']*100:>6.1f}%")
    print(f"  策略最大回撤: {_pct(m['max_dd']):>10}   買進持有最大回撤: {_pct(m['bah_max_dd'])}")


def _print_backtest_report(days: list[dict], sim: dict, m: dict, W: int) -> None:
    _print_regime_distribution(days, len(days), W)
    _print_signal_distribution(days, len(days), W)
    _print_performance_summary(m, W)

    print_chart(m['portfolio'], m['bah_port'], sim['position'], [d['date'] for d in days])

    print(f"\n{'─'*W}")
    print("  年度報酬比較  ★ = 策略勝出")
    print_yearly(days, m['portfolio'])

    print(f"\n{'─'*W}")
    print("  月份報酬比較  ★ = 策略勝出")
    print_monthly(days, m['portfolio'], sim['position'])

    print(f"\n{'─'*W}")
    print("  換倉紀錄（最後 15 筆）")
    print_rebal_log(m['rebalances'])

    print(f"\n{'='*W}\n")


def main():
    """CLI entry point: run the walk-forward simulation for one symbol and print
    the chart/rebalance-log/monthly/yearly/summary report."""
    args = _parse_cli_args()
    symbol, sim_end_dt, sim_start_dt, span_years = _resolve_sim_window(args)

    W = 68
    _print_run_header(symbol, args, W)

    # ── 1-3. 取得數據、決定起始點、Walk-Forward 計算每日信號 ────────────────────
    ctx = _load_backtest_context(symbol, span_years, sim_end_dt, sim_start_dt)
    days, prices_arr = ctx['days'], ctx['prices']

    # ── 4. 模擬 ──────────────────────────────────────────────────────────────
    # Start level depends on regime of first day
    first_bonus = days[0].get('regime_bonus', 0.0) if days else 0.0
    init_level  = REGIME_PARAMS.get(first_bonus, _DEFAULT_PARAMS)['start']
    sim = simulate(days, start_level=init_level)
    m   = calc_metrics(sim, prices_arr)

    # ── 5. 報告 ──────────────────────────────────────────────────────────────
    _print_backtest_report(days, sim, m, W)


if __name__ == '__main__':
    main()
