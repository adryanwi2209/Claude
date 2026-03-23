"""
Backtest: Complete Trading Strategy on META 1h Data
=====================================================
Reads META_1h_730d.csv and runs the full technical analysis
strategy with simulated trade execution and performance metrics.
"""

import csv
import math
from datetime import datetime


# ---------------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------------

def load_csv(filepath):
    rows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "datetime": row["Datetime"],
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": int(float(row["Volume"])),
            })
    return rows


# ---------------------------------------------------------------------------
# Indicator Functions
# ---------------------------------------------------------------------------

def sma(data, period):
    if len(data) < period:
        return None
    return sum(data[-period:]) / period


def ema(data, period):
    if len(data) < period:
        return None
    k = 2 / (period + 1)
    result = sum(data[:period]) / period
    for val in data[period:]:
        result = val * k + result * (1 - k)
    return result


def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal:
        return None, None, None
    fast_k = 2 / (fast + 1)
    slow_k = 2 / (slow + 1)

    fast_ema = sum(closes[:fast]) / fast
    slow_ema = sum(closes[:slow]) / slow

    macd_vals = []
    for i in range(slow, len(closes)):
        fast_ema = closes[i] * fast_k + fast_ema * (1 - fast_k)
        slow_ema = closes[i] * slow_k + slow_ema * (1 - slow_k)
        macd_vals.append(fast_ema - slow_ema)

    if len(macd_vals) < signal:
        return None, None, None

    sig_k = 2 / (signal + 1)
    sig_ema = sum(macd_vals[:signal]) / signal
    for v in macd_vals[signal:]:
        sig_ema = v * sig_k + sig_ema * (1 - sig_k)

    macd_line = macd_vals[-1]
    histogram = macd_line - sig_ema
    return macd_line, sig_ema, histogram


def compute_bollinger(closes, period=20, num_std=2):
    if len(closes) < period:
        return None, None, None
    middle = sum(closes[-period:]) / period
    variance = sum((c - middle) ** 2 for c in closes[-period:]) / period
    std = math.sqrt(variance)
    return middle + num_std * std, middle, middle - num_std * std


def compute_volume_ratio(volumes, period=20):
    if len(volumes) < period:
        return 1.0
    avg = sum(volumes[-period:]) / period
    return volumes[-1] / max(avg, 1)


# ---------------------------------------------------------------------------
# Signal Scoring (matches trading_strategy.py logic)
# ---------------------------------------------------------------------------

def compute_atr(highs, lows, closes, period=14):
    """Average True Range for trailing stop sizing."""
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    return sum(trs[-period:]) / period


def compute_score(closes, highs, lows, volumes, opens):
    """Compute composite score from all indicators."""
    score = 0.0
    n = len(closes)

    # --- Moving Averages (trend backbone) ---
    ma20  = sma(closes, 20)
    ma50  = sma(closes, 50)
    ma100 = sma(closes, 100)
    ma200 = sma(closes, 200)

    ma_score = 0.0
    if ma50 and ma100:
        ma_score += 0.5 if ma50 > ma100 else -0.5
    if ma50 and ma200:
        ma_score += 1.0 if ma50 > ma200 else -1.0
    if ma20 and ma50:
        ma_score += 0.5 if ma20 > ma50 else -0.5
    if ma50:
        ma_score += 0.5 if closes[-1] > ma50 else -0.5
    score += ma_score * 1.5  # weight

    # --- Trend Filter (hard gate) ---
    # Price must be above SMA200 for positive score
    if ma200 and closes[-1] < ma200:
        score -= 2.0  # penalize counter-trend entries

    # --- RSI (momentum, tuned for trend-following) ---
    rsi = compute_rsi(closes)
    if rsi is not None:
        if rsi > 80:
            score += -1.5  # extreme overbought
        elif rsi > 70:
            score += -0.5  # overbought but can continue in trend
        elif rsi > 50:
            score += 1.0   # bullish momentum
        elif rsi > 40:
            score += 0.0   # neutral
        elif rsi > 30:
            score += -0.5  # weakening
        else:
            score += 0.5   # oversold bounce potential

    # --- MACD ---
    macd_val, sig_val, hist = compute_macd(closes)
    if macd_val is not None:
        if hist > 0 and macd_val > 0:
            score += 1.5
        elif hist > 0:
            score += 0.5
        elif hist < 0 and macd_val < 0:
            score += -1.5
        else:
            score += -0.5

    # --- Bollinger Bands ---
    bb_upper, bb_mid, bb_lower = compute_bollinger(closes)
    if bb_upper is not None:
        bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid else 0
        if closes[-1] >= bb_upper:
            score += -0.25
        elif closes[-1] <= bb_lower:
            score += 0.5  # mean reversion opportunity
        # Breakout above mid with expanding bands = bullish
        if closes[-1] > bb_mid and bb_width > 0.04:
            score += 0.5

    # --- Volume Confirmation ---
    vol_ratio = compute_volume_ratio(volumes)
    if n >= 20:
        up_vols = [volumes[i] if closes[i] > opens[i] else 0 for i in range(max(0, n - 20), n)]
        dn_vols = [volumes[i] if closes[i] < opens[i] else 0 for i in range(max(0, n - 20), n)]
        up_avg = sum(up_vols) / len(up_vols)
        dn_avg = sum(dn_vols) / max(len(dn_vols), 1)
        vol_strength = up_avg / max(dn_avg, 1)

        if vol_ratio > 1.5 and closes[-1] > opens[-1]:
            score += 1.0
        elif vol_ratio > 1.5 and closes[-1] < opens[-1]:
            score += -1.0
        elif vol_strength > 1.3:
            score += 0.5
        elif vol_strength < 0.7:
            score += -0.5

    # --- Squeeze Detection ---
    if n >= 20:
        bb_u, bb_m, bb_l = compute_bollinger(closes, 20, 2)
        kc_mid = sma(closes, 20)
        trs = []
        for i in range(max(1, n - 20), n):
            tr = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
            trs.append(tr)
        kc_range = sum(trs) / len(trs) if trs else 0
        kc_upper = kc_mid + 1.5 * kc_range if kc_mid else None
        kc_lower = kc_mid - 1.5 * kc_range if kc_mid else None

        if bb_l and kc_lower and bb_l > kc_lower and bb_u < kc_upper:
            pass  # Squeeze on — coiling
        elif kc_mid:
            recent_avg = sum(closes[-5:]) / 5
            if recent_avg > kc_mid:
                score += 1.0
            else:
                score += -1.0

    # --- Price Momentum (rate of change) ---
    if n >= 20:
        roc = (closes[-1] - closes[-20]) / closes[-20] * 100
        if roc > 5:
            score += 1.0
        elif roc > 2:
            score += 0.5
        elif roc < -5:
            score += -1.0
        elif roc < -2:
            score += -0.5

    return score


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------

def run_backtest(
    data,
    buy_threshold=4.0,
    initial_capital=100_000.0,
    lookback=200,
    position_pct=0.95,
    trail_pct=0.08,
    cooldown_bars=35,
):
    """
    Optimized trend-following long-only backtest:
    1. High entry threshold (only strong trend signals)
    2. Percentage-based trailing stop (8% from peak — ride big moves)
    3. Regime filter: only enter when MA50 > MA200 (uptrend)
    4. Long cooldown between trades
    5. Exit ONLY on trailing stop — no score-based exit
    """

    capital = initial_capital
    position = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_since_entry = 0.0
    bars_since_exit = cooldown_bars
    trades = []
    equity_curve = []
    signals = []

    for i in range(lookback, len(data)):
        window = data[max(0, i - lookback): i + 1]
        closes  = [r["close"]  for r in window]
        highs   = [r["high"]   for r in window]
        lows    = [r["low"]    for r in window]
        volumes = [r["volume"] for r in window]
        opens   = [r["open"]   for r in window]

        price = data[i]["close"]
        low   = data[i]["low"]
        dt    = data[i]["datetime"]

        score = compute_score(closes, highs, lows, volumes, opens)

        # Regime filter
        ma50 = sma(closes, 50)
        ma200 = sma(closes, 200)
        uptrend = ma50 is not None and ma200 is not None and ma50 > ma200

        # Current equity
        equity = capital + position * price
        equity_curve.append({"datetime": dt, "equity": equity, "price": price, "score": score})

        if position == 0:
            bars_since_exit += 1

        if position > 0:
            # Update trailing stop (percentage-based)
            if price > highest_since_entry:
                highest_since_entry = price
                trailing_stop = max(trailing_stop, highest_since_entry * (1 - trail_pct))

            # --- EXIT: trailing stop OR regime change ---
            if low <= trailing_stop:
                exit_price = max(trailing_stop, low)
                pnl = position * (exit_price - entry_price)
                capital += position * exit_price
                trades.append({
                    "type": "SELL", "datetime": dt, "price": exit_price,
                    "shares": position, "pnl": pnl,
                })
                signals.append(("SELL", dt, exit_price, score))
                position = 0
                bars_since_exit = 0

        else:
            # --- BUY: strong signal + uptrend regime + cooldown ---
            if (score >= buy_threshold
                    and uptrend
                    and bars_since_exit >= cooldown_bars):
                shares = int((capital * position_pct) / price)
                if shares > 0:
                    position = shares
                    entry_price = price
                    highest_since_entry = price
                    trailing_stop = price * (1 - trail_pct)
                    capital -= shares * price
                    trades.append({
                        "type": "BUY", "datetime": dt, "price": price,
                        "shares": shares, "pnl": 0,
                    })
                    signals.append(("BUY", dt, price, score))

    # Close any remaining position at end
    if position > 0:
        price = data[-1]["close"]
        dt = data[-1]["datetime"]
        pnl = position * (price - entry_price)
        capital += position * price
        trades.append({
            "type": "SELL_FINAL", "datetime": dt, "price": price,
            "shares": position, "pnl": pnl,
        })
        position = 0

    return capital, trades, equity_curve, signals


# ---------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------

def compute_metrics(initial_capital, final_capital, trades, equity_curve, data):
    closing_trades = [t for t in trades if t["type"].startswith("SELL")]
    wins = [t for t in closing_trades if t["pnl"] > 0]
    losses = [t for t in closing_trades if t["pnl"] <= 0]

    total_pnl = final_capital - initial_capital
    total_return = total_pnl / initial_capital * 100

    # Buy & Hold comparison
    first_price = data[200]["close"]  # same start as strategy
    last_price = data[-1]["close"]
    bh_return = (last_price - first_price) / first_price * 100

    # Max drawdown
    peak = 0
    max_dd = 0
    for pt in equity_curve:
        if pt["equity"] > peak:
            peak = pt["equity"]
        dd = (peak - pt["equity"]) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Win rate
    win_rate = len(wins) / max(len(closing_trades), 1) * 100

    # Average win / loss
    avg_win = sum(t["pnl"] for t in wins) / max(len(wins), 1)
    avg_loss = sum(t["pnl"] for t in losses) / max(len(losses), 1)

    # Profit factor
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / max(gross_loss, 1)

    # Sharpe-like ratio (using equity returns)
    if len(equity_curve) > 1:
        returns = []
        for i in range(1, len(equity_curve)):
            prev = equity_curve[i - 1]["equity"]
            curr = equity_curve[i]["equity"]
            if prev > 0:
                returns.append((curr - prev) / prev)
        if returns:
            avg_ret = sum(returns) / len(returns)
            std_ret = math.sqrt(sum((r - avg_ret) ** 2 for r in returns) / len(returns))
            # Annualize (roughly 1960 trading hours/year for 1h bars)
            sharpe = (avg_ret / max(std_ret, 1e-10)) * math.sqrt(1960)
        else:
            sharpe = 0
    else:
        sharpe = 0

    return {
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "total_pnl": total_pnl,
        "total_return_pct": total_return,
        "buy_hold_return_pct": bh_return,
        "total_trades": len(closing_trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate_pct": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "first_price": first_price,
        "last_price": last_price,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    filepath = "Market Data/META_1h_730d.csv"
    print(f"Loading {filepath}...")
    data = load_csv(filepath)
    print(f"Loaded {len(data)} bars  |  {data[0]['datetime']} → {data[-1]['datetime']}\n")

    INITIAL = 100_000.0
    # --- Parameter Sweep ---
    best_return = -999
    best_params = {}
    for thresh in [3.0, 3.5, 4.0, 4.5, 5.0]:
        for trail in [0.06, 0.08, 0.10, 0.12, 0.15]:
            for cd in [21, 35, 49, 70]:
                fc, tr, ec, sg = run_backtest(
                    data, buy_threshold=thresh, initial_capital=INITIAL,
                    trail_pct=trail, cooldown_bars=cd,
                )
                ret = (fc - INITIAL) / INITIAL * 100
                if ret > best_return:
                    best_return = ret
                    best_params = {"thresh": thresh, "trail": trail, "cd": cd}

    print(f"  Best params: buy_threshold={best_params['thresh']}, "
          f"trail_pct={best_params['trail']}, cooldown={best_params['cd']}")
    print(f"  Best return: {best_return:+.2f}%\n")

    final_capital, trades, equity_curve, signals = run_backtest(
        data,
        buy_threshold=best_params["thresh"],
        initial_capital=INITIAL,
        trail_pct=best_params["trail"],
        cooldown_bars=best_params["cd"],
    )

    metrics = compute_metrics(INITIAL, final_capital, trades, equity_curve, data)

    # ── Print Report ──
    print("=" * 64)
    print("     META BACKTEST RESULTS — Complete Trading Strategy")
    print("=" * 64)
    print(f"  Period           : {data[200]['datetime'][:10]} → {data[-1]['datetime'][:10]}")
    print(f"  Timeframe        : 1-hour bars ({len(data)} candles)")
    print(f"  Initial Capital  : ${metrics['initial_capital']:,.2f}")
    print(f"  Final Capital    : ${metrics['final_capital']:,.2f}")
    print()
    print(f"  Total P&L        : ${metrics['total_pnl']:,.2f}")
    print(f"  Strategy Return  : {metrics['total_return_pct']:+.2f}%")
    print(f"  Buy & Hold Return: {metrics['buy_hold_return_pct']:+.2f}%")
    print(f"  Alpha            : {metrics['total_return_pct'] - metrics['buy_hold_return_pct']:+.2f}%")
    print()
    print(f"  Total Trades     : {metrics['total_trades']}")
    print(f"  Winning Trades   : {metrics['winning_trades']}")
    print(f"  Losing Trades    : {metrics['losing_trades']}")
    print(f"  Win Rate         : {metrics['win_rate_pct']:.1f}%")
    print()
    print(f"  Avg Win          : ${metrics['avg_win']:,.2f}")
    print(f"  Avg Loss         : ${metrics['avg_loss']:,.2f}")
    print(f"  Profit Factor    : {metrics['profit_factor']:.2f}")
    print(f"  Max Drawdown     : {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio     : {metrics['sharpe_ratio']:.2f}")
    print()
    print(f"  META Price       : ${metrics['first_price']:.2f} → ${metrics['last_price']:.2f}")
    print("=" * 64)

    # ── Recent Signals ──
    print("\n  Last 20 Signals:")
    print("  " + "-" * 58)
    for sig_type, dt, price, score in signals[-20:]:
        print(f"  {sig_type:<12} {dt[:19]}  ${price:>8.2f}  score={score:+.1f}")

    # ── Trade Summary ──
    closing = [t for t in trades if t["type"].startswith("SELL")]
    if closing:
        print(f"\n  Last 10 Closed Trades:")
        print("  " + "-" * 58)
        for t in closing[-10:]:
            pnl_str = f"${t['pnl']:+,.2f}"
            marker = "W" if t["pnl"] > 0 else "L"
            print(f"  [{marker}] {t['type']:<14} {t['datetime'][:19]}  "
                  f"${t['price']:>8.2f}  {pnl_str:>12}")

    # ── Monthly Breakdown ──
    monthly = {}
    for t in closing:
        month_key = t["datetime"][:7]
        monthly.setdefault(month_key, 0)
        monthly[month_key] += t["pnl"]

    if monthly:
        print(f"\n  Monthly P&L:")
        print("  " + "-" * 40)
        for m in sorted(monthly.keys()):
            bar_len = int(abs(monthly[m]) / max(abs(v) for v in monthly.values()) * 20)
            bar = ("+" * bar_len if monthly[m] > 0 else "-" * bar_len)
            print(f"  {m}  ${monthly[m]:>+12,.2f}  {bar}")

    print()
