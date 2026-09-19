"""
Text report rendering for tech_analysis.py: turns analyze()'s result dict
into the full single-stock report, plus the multi-stock summary table.
"""
from __future__ import annotations

from datetime import date, timedelta

from backtest_engine import MOVE_THRESH

_REC_ICON = {"強力加碼": "🚀", "加碼": "↑", "持平": "→", "減碼": "↓", "強力減碼": "⚠"}
_REC_RANK = {"強力加碼": 5, "加碼": 4, "持平": 3, "減碼": 2, "強力減碼": 1}


def _direction_label(excess: float, n: int) -> str:
    """Label a condition based on its excess edge vs baseline."""
    # Require minimum statistical confidence: penalise small samples
    effective = excess * min(n / 10.0, 1.0)
    if effective >= +0.20:
        return "強多 ↑↑"
    if effective >= +0.10:
        return "多  ↑ "
    if effective <= -0.20:
        return "強空 ↓↓"
    if effective <= -0.10:
        return "空  ↓ "
    return "中性 → "


def _fmt_val(x, fmt: str = '.2f') -> str:
    return f"{x:{fmt}}" if x is not None else "N/A"


def _fmt_header(r: dict, w: int) -> list[str]:
    lines = ["=" * w]
    name_str = f"  {r['company_name']}" if r.get('company_name') else ""
    lines.append(f"  股票代號: {r['symbol']}{name_str}")
    chg = f"  ({r['price_chg']:+.2f}, {r['price_pct']:+.2f}%)" if r['price_chg'] else ""
    lines.append(f"  最新收盤: {_fmt_val(r['price'])}{chg}")
    ed = r.get('earnings_date')
    if ed and date.today() <= ed <= date.today() + timedelta(days=7):
        lines.append(f"  ⚠ 財報將於 {ed.strftime('%-m/%-d')} 公布，財報前後波動放大，建議觀望")
    return lines


def _fmt_indicators(r: dict, w: int) -> list[str]:
    lines = ["-" * w, "  技術指標快照:"]
    lines.append(f"    KD:   K={_fmt_val(r['K'],'.1f')}  D={_fmt_val(r['D'],'.1f')}")
    lines.append(f"    RSI:  {_fmt_val(r['RSI'],'.1f')}")
    lines.append(f"    MACD: {_fmt_val(r['MACD'],'.3f')}  Signal: {_fmt_val(r['MACD_Signal'],'.3f')}")
    ma_parts = []
    for key, lbl in [('MA5','5MA'),('MA20','20MA'),('MA60','60MA'),('MA200','200MA')]:
        if r[key] is not None:
            ma_parts.append(f"{lbl}={_fmt_val(r[key])}")
    lines.append(f"    均線: {'  '.join(ma_parts)}")
    bias_parts = []
    for key, lbl in [('BIAS5','5日'),('BIAS20','20日'),('BIAS60','60日')]:
        v = r.get(key)
        if v is not None:
            bias_parts.append(f"{lbl}={v:+.1f}%")
    if bias_parts:
        lines.append(f"    乖離率: {'  '.join(bias_parts)}")
    return lines


def _fmt_futures_lines(futures: dict) -> list[str]:
    net = futures.get('net_pos')
    net_str = f"{net:+,.0f}口" if net is not None else "N/A"
    bonus_str = f"  分數調整 {futures['bonus']:+.2f}" if futures.get('bonus') else ""
    lines = [f"    外資期貨淨部位(TXF): {net_str}{bonus_str}  ({futures.get('note', '')})"]
    short = futures.get('short_oi')
    if short is not None:
        short_str = f"{short:+,.0f}口"
        short_bonus_str = f"  分數調整 {futures['short_bonus']:+.2f}" if futures.get('short_bonus') else ""
        lines.append(
            f"    外資期貨空方口數(TXF): {short_str}{short_bonus_str}  ({futures.get('short_note', '')})"
        )
    return lines


def _fmt_market(r: dict, w: int) -> list[str]:
    vix     = r.get('vix',     {})
    buffett = r.get('buffett', {})
    regime  = r.get('regime',  {})
    futures = r.get('futures', {})
    if not (vix or buffett or regime or futures):
        return []

    lines = ["-" * w, "  總體市場指標:"]
    if regime:
        twii_str = f"TWII {regime['twii']:.0f} / MA200 {regime['ma200']:.0f}" if regime.get('twii') else ""
        bonus_str = f"  分數調整 {regime['bonus']:+.2f}" if regime.get('bonus') is not None else ""
        lines.append(f"    台股趨勢: {twii_str}  [{regime['regime']}]{bonus_str}")
    if vix:
        lines.append(
            f"    恐慌指數(VIX): {vix['value']:.1f}  "
            f"[{vix['level']}]  {vix['desc']}"
        )
    if buffett:
        cap_t  = buffett['market_cap_bn'] / 1_000
        gdp_t  = buffett['gdp_bn']        / 1_000
        lines.append(
            f"    巴菲特指標:  {buffett['ratio']:.1f}%"
            f"  (市值{cap_t:.1f}兆 / GDP{gdp_t:.1f}兆 {buffett['gdp_year']})"
            f"  [{buffett['level']}]"
        )
        note = buffett.get('note', '')
        suffix = f"  ({note})" if note else ""
        lines.append(f"    {buffett['desc']}{suffix}")
    if futures:
        lines.extend(_fmt_futures_lines(futures))
    return lines


def _fmt_institutional(r: dict, w: int) -> list[str]:
    inst = r.get('inst_summary', {})
    if not inst:
        return []

    def streak(b, s):
        return f"連買{b}日" if b else (f"連賣{s}日" if s else "持平")

    return [
        "-" * w,
        "  三大法人 (近5日累積，萬股):",
        f"    外資: 今日{inst['f_today']:+,}萬股  近5日{inst['f5']:+,}萬股"
        f"  [{streak(inst['f_buy'],inst['f_sell'])}]",
        f"    投信: 今日{inst['t_today']:+,}萬股  近5日{inst['t5']:+,}萬股"
        f"  [{streak(inst['t_buy'],inst['t_sell'])}]",
        f"    三大合計: 今日{inst['tot_today']:+,}萬股  近5日{inst['tot5']:+,}萬股",
    ]


def _fmt_backtest_row(name: str, stats: dict, active_names: set, scoring_names: set, col_w: list[int]) -> str:
    is_active  = name in active_names
    is_scoring = name in scoring_names
    if is_active and is_scoring:
        marker = " ★"   # active + scored
    elif is_active:
        marker = " ✓"   # active but group already covered by a better condition
    else:
        marker = "  "
    up_pct    = stats['up_rate']   * 100
    down_pct  = stats['down_rate'] * 100
    excess    = stats['excess_edge']
    direction = _direction_label(excess, stats['count'])
    return (
        f"  {name:<{col_w[0]}} {stats['count']:>{col_w[1]}} {marker:>{col_w[2]}}"
        f" {up_pct:>{col_w[3]}.0f}% {down_pct:>{col_w[4]}.0f}%"
        f" {excess:>+{col_w[5]}.3f} {direction:<{col_w[6]}}"
    )


def _fmt_backtest_table(r: dict, w: int) -> list[str]:
    base = r['base']
    lines = [
        "-" * w,
        f"  歷史回測 ({base['count']} 個交易日 | 目標: 5日後漲跌>{MOVE_THRESH*100:.0f}%"
        f" | 基準: 上漲{base['up_rate']*100:.0f}% 下跌{base['down_rate']*100:.0f}%)",
        "",
    ]

    # Determine which conditions actually contributed to the score (one per group)
    scoring_names = {s['name'] for s in r['group_best'].values()}
    active_names  = {s['name'] for s in r['active_bt']}

    col_w = [27, 5, 5, 7, 7, 8, 10]
    lines.append(
        f"  {'條件':<{col_w[0]}} {'次數':>{col_w[1]}} {'觸發':>{col_w[2]}}"
        f" {'上漲率':>{col_w[3]}} {'下跌率':>{col_w[4]}} {'超額邊際':>{col_w[5]}} {'結論':<{col_w[6]}}"
    )
    lines.append("  " + "─" * (w - 2))

    def sort_key(item):
        name, stats = item
        # Active conditions first (sorted by |excess_edge|), then inactive
        return (0 if name in active_names else 1, -abs(stats['excess_edge']))

    for name, stats in sorted(r['all_bt'].items(), key=sort_key):
        lines.append(_fmt_backtest_row(name, stats, active_names, scoring_names, col_w))

    lines.append(f"  (★=計入評分 ✓=觸發但同組已有更強信號  超額邊際=條件邊際−基準{base['edge']:+.2f})")
    return lines


def _fmt_inst_signals(r: dict) -> list[str]:
    if not r['inst_signals']:
        return []
    lines = ["", "  [籌碼面]"]
    for tag, desc in r['inst_signals']:
        lines.append(f"    [{tag}] {desc}")
    return lines


def _fmt_verdict(r: dict, w: int) -> list[str]:
    reg_bonus = r.get('regime_bonus', 0.0)
    reg_name  = r.get('regime', {}).get('regime', '')
    reg_str   = f"  市場{reg_name}({reg_bonus:+.2f})" if reg_bonus != 0 else ""
    fut_bonus = r.get('futures_bonus', 0.0)
    fut_str   = f"  期貨籌碼({fut_bonus:+.2f})" if fut_bonus != 0 else ""
    rec  = r['recommendation']
    icon = _REC_ICON.get(rec, "")
    return [
        "-" * w,
        f"  技術分數: {r['tech_score']:+.3f}  "
        f"籌碼分數: {r['inst_score']:+d}(×0.12={r['inst_score']*0.12:+.2f})"
        f"{reg_str}{fut_str}  合計: {r['combined']:+.3f}",
        f"  ▶ 未來一週建議: 【{rec}】 {icon}",
    ]


def fmt_report(r: dict) -> str:
    """Render analyze()'s result dict into the full single-stock text report."""
    w = 70
    lines: list[str] = []
    lines.extend(_fmt_header(r, w))
    lines.extend(_fmt_indicators(r, w))
    lines.extend(_fmt_market(r, w))
    lines.extend(_fmt_institutional(r, w))
    lines.extend(_fmt_backtest_table(r, w))
    lines.extend(_fmt_inst_signals(r))
    lines.extend(_fmt_verdict(r, w))
    lines.append("=" * w)
    return "\n".join(lines)


def fmt_summary_table(results: list[dict]) -> str:
    """Multi-stock comparison table, sorted 強力加碼 → 加碼 → 持平 → 減碼 → 強力減碼
    (ties broken by combined score, descending)."""
    w = 70
    lines: list[str] = []
    lines.append("=" * w)
    lines.append("  綜合比較表（依建議排序）")
    lines.append("-" * w)
    col_w = [6, 8, 9, 8, 8, 8]
    lines.append(
        f"  {'股票':<{col_w[0]}}{'公司名':<{col_w[1]}}{'收盤價':>{col_w[2]}}"
        f"{'技術分':>{col_w[3]}}{'籌碼分':>{col_w[4]}}{'合計':>{col_w[5]}}  建議"
    )
    ordered = sorted(results, key=lambda r: (-_REC_RANK.get(r['recommendation'], 0), -r['combined']))
    for r in ordered:
        rec = r['recommendation']
        icon = _REC_ICON.get(rec, "")
        lines.append(
            f"  {r['symbol']:<{col_w[0]}}{(r.get('company_name') or ''):<{col_w[1]}}"
            f"{r['price']:>{col_w[2]}.2f}{r['tech_score']:>+{col_w[3]}.3f}"
            f"{r['inst_score']*0.12:>+{col_w[4]}.2f}{r['combined']:>+{col_w[5]}.3f}"
            f"  【{rec}】{icon}"
        )
    lines.append("=" * w)
    return "\n".join(lines)
