"""
Optimized Cross-Ticker Backtest v2
====================================
Adaptive trailing stop, fast re-entry, profit lock-in, regime strength gate.
Runs parameter sweep across META, MSFT, GOOGL simultaneously.
"""

import csv
import math


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


def sma(data, period):
    if len(data) < period:
        return None
    return sum(data[-period:]) / period


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
    return macd_vals[-1], sig_ema, macd_vals[-1] - sig_ema


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


def compute_atr(highs, lows, closes, period=20):
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    return sum(trs[-period:]) / period


# ---------------------------------------------------------------------------
# Signal Scoring (same as before)
# ---------------------------------------------------------------------------

def compute_score(closes, highs, lows, volumes, opens):
    score = 0.0
    n = len(closes)

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
    score += ma_score * 1.5

    if ma200 and closes[-1] < ma200:
        score -= 2.0

    rsi = compute_rsi(closes)
    if rsi is not None:
        if rsi > 80:
            score += -1.5
        elif rsi > 70:
            score += -0.5
        elif rsi > 50:
            score += 1.0
        elif rsi > 40:
            score += 0.0
        elif rsi > 30:
            score += -0.5
        else:
            score += 0.5

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

    bb_upper, bb_mid, bb_lower = compute_bollinger(closes)
    if bb_upper is not None:
        bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid else 0
        if closes[-1] >= bb_upper:
            score += -0.25
        elif closes[-1] <= bb_lower:
            score += 0.5
        if closes[-1] > bb_mid and bb_width > 0.04:
            score += 0.5

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
            pass
        elif kc_mid:
            recent_avg = sum(closes[-5:]) / 5
            if recent_avg > kc_mid:
                score += 1.0
            else:
                score += -1.0

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
# Optimized Backtest Engine v2
# ---------------------------------------------------------------------------

def run_backtest_v2(
    data,
    buy_threshold=4.0,
    initial_capital=100_000.0,
    lookback=200,
    position_pct=0.95,
    base_trail_pct=0.10,
    cooldown_bars=35,
    fast_reentry_score=6.0,
    fast_reentry_cd=14,
    profit_lock_trigger=0.15,
    profit_lock_trail=0.07,
):
    """
    v2 Optimized Backtest:
    1. Adaptive trailing stop — base trail widens by ATR volatility ratio
    2. Fast re-entry — skip normal cooldown if score >= fast_reentry_score
    3. Regime strength gate — price must be above MA50 AND MA50 > MA200
    4. Profit lock-in — once trade is up profit_lock_trigger%, tighten trail
    5. Trend confirmation — require MA20 > MA50 for entry (short-term alignment)
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
        atr = compute_atr(highs, lows, closes, 20)

        # Regime checks
        ma20  = sma(closes, 20)
        ma50  = sma(closes, 50)
        ma200 = sma(closes, 200)
        uptrend = (ma50 is not None and ma200 is not None and ma50 > ma200)
        price_above_ma50 = ma50 is not None and price > ma50
        ma20_above_ma50 = ma20 is not None and ma50 is not None and ma20 > ma50

        # Adaptive trail: widen in high-volatility, tighten in low-vol
        if atr and price > 0:
            atr_pct = atr / price
            # Normalize: if ATR% > 2%, widen trail; if < 1%, tighten
            vol_multiplier = max(0.8, min(1.5, atr_pct / 0.015))
            adaptive_trail = base_trail_pct * vol_multiplier
        else:
            adaptive_trail = base_trail_pct

        equity = capital + position * price
        equity_curve.append({"datetime": dt, "equity": equity, "price": price, "score": score})

        if position == 0:
            bars_since_exit += 1

        if position > 0:
            # Update highest price
            if price > highest_since_entry:
                highest_since_entry = price

            # Profit lock-in: once up enough, tighten trail
            gain_pct = (highest_since_entry - entry_price) / entry_price
            if gain_pct >= profit_lock_trigger:
                current_trail = profit_lock_trail
            else:
                current_trail = adaptive_trail

            # Update trailing stop
            new_stop = highest_since_entry * (1 - current_trail)
            trailing_stop = max(trailing_stop, new_stop)

            # EXIT: trailing stop hit
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
            # Cooldown check: normal or fast re-entry
            cd_met = bars_since_exit >= cooldown_bars
            fast_reentry = (score >= fast_reentry_score
                            and bars_since_exit >= fast_reentry_cd)
            cooldown_ok = cd_met or fast_reentry

            # BUY: strong signal + uptrend + price above MA50 + MA20 > MA50
            if (score >= buy_threshold
                    and uptrend
                    and price_above_ma50
                    and ma20_above_ma50
                    and cooldown_ok):
                shares = int((capital * position_pct) / price)
                if shares > 0:
                    position = shares
                    entry_price = price
                    highest_since_entry = price
                    trailing_stop = price * (1 - adaptive_trail)
                    capital -= shares * price
                    trades.append({
                        "type": "BUY", "datetime": dt, "price": price,
                        "shares": shares, "pnl": 0,
                    })
                    signals.append(("BUY", dt, price, score))

    # Close remaining position
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

    first_price = data[200]["close"]
    last_price = data[-1]["close"]
    bh_return = (last_price - first_price) / first_price * 100

    peak = 0
    max_dd = 0
    for pt in equity_curve:
        if pt["equity"] > peak:
            peak = pt["equity"]
        dd = (peak - pt["equity"]) / peak * 100
        if dd > max_dd:
            max_dd = dd

    win_rate = len(wins) / max(len(closing_trades), 1) * 100
    avg_win = sum(t["pnl"] for t in wins) / max(len(wins), 1)
    avg_loss = sum(t["pnl"] for t in losses) / max(len(losses), 1)

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / max(gross_loss, 1)

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
        "alpha": total_return - bh_return,
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
# Main: Cross-Ticker Optimization
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TICKERS = {
        "META":  "Market Data/META_1h_730d.csv",
        "MSFT":  "Market Data/MSFT_1h_730d.csv",
        "GOOGL": "Market Data/GOOGL_1h_730d.csv",
    }

    INITIAL = 100_000.0

    # Load all data
    all_data = {}
    for ticker, path in TICKERS.items():
        all_data[ticker] = load_csv(path)
        d = all_data[ticker]
        print(f"  {ticker}: {len(d)} bars  |  {d[0]['datetime'][:10]} → {d[-1]['datetime'][:10]}")

    print("\n  Running PER-TICKER parameter sweep...\n")

    best_per_ticker = {}

    for ticker, data in all_data.items():
        best_return = -999
        best_p = {}
        for thresh in [3.0, 3.5, 4.0, 4.5]:
            for trail in [0.10, 0.12, 0.15, 0.18]:
                for cd in [21, 35, 49, 70]:
                    for fast_score in [5.5, 6.5]:
                        for lock_trig in [0.12, 0.18]:
                            for lock_trail in [0.06, 0.08]:
                                fc, _, _, _ = run_backtest_v2(
                                    data,
                                    buy_threshold=thresh,
                                    initial_capital=INITIAL,
                                    base_trail_pct=trail,
                                    cooldown_bars=cd,
                                    fast_reentry_score=fast_score,
                                    profit_lock_trigger=lock_trig,
                                    profit_lock_trail=lock_trail,
                                )
                                ret = (fc - INITIAL) / INITIAL * 100
                                bh = (data[-1]["close"] - data[200]["close"]) / data[200]["close"] * 100
                                alpha = ret - bh
                                if alpha > best_return:
                                    best_return = alpha
                                    best_p = {
                                        "thresh": thresh, "trail": trail, "cd": cd,
                                        "fast_score": fast_score,
                                        "lock_trig": lock_trig, "lock_trail": lock_trail,
                                    }
        best_per_ticker[ticker] = best_p
        print(f"  {ticker}: threshold={best_p['thresh']}, trail={best_p['trail']*100:.0f}%, "
              f"cd={best_p['cd']}, fast_re={best_p['fast_score']}, "
              f"lock={best_p['lock_trig']*100:.0f}%→{best_p['lock_trail']*100:.0f}%  "
              f"alpha={best_return:+.2f}%")

    print()

    # Run final backtest on each ticker with its own best params
    for ticker, data in all_data.items():
        best_params = best_per_ticker[ticker]
        fc, trades, ec, signals = run_backtest_v2(
            data,
            buy_threshold=best_params["thresh"],
            initial_capital=INITIAL,
            base_trail_pct=best_params["trail"],
            cooldown_bars=best_params["cd"],
            fast_reentry_score=best_params["fast_score"],
            profit_lock_trigger=best_params["lock_trig"],
            profit_lock_trail=best_params["lock_trail"],
        )

        m = compute_metrics(INITIAL, fc, trades, ec, data)

        print("=" * 64)
        print(f"     {ticker} BACKTEST RESULTS — Optimized v2 Strategy")
        print("=" * 64)
        print(f"  Period           : {data[200]['datetime'][:10]} → {data[-1]['datetime'][:10]}")
        print(f"  Initial Capital  : ${m['initial_capital']:,.2f}")
        print(f"  Final Capital    : ${m['final_capital']:,.2f}")
        print()
        print(f"  Total P&L        : ${m['total_pnl']:,.2f}")
        print(f"  Strategy Return  : {m['total_return_pct']:+.2f}%")
        print(f"  Buy & Hold Return: {m['buy_hold_return_pct']:+.2f}%")
        print(f"  Alpha            : {m['alpha']:+.2f}%")
        print()
        print(f"  Total Trades     : {m['total_trades']}")
        print(f"  Win / Loss       : {m['winning_trades']}W / {m['losing_trades']}L")
        print(f"  Win Rate         : {m['win_rate_pct']:.1f}%")
        print(f"  Avg Win          : ${m['avg_win']:,.2f}")
        print(f"  Avg Loss         : ${m['avg_loss']:,.2f}")
        print(f"  Profit Factor    : {m['profit_factor']:.2f}")
        print(f"  Max Drawdown     : {m['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio     : {m['sharpe_ratio']:.2f}")
        print(f"  {ticker} Price    : ${m['first_price']:.2f} → ${m['last_price']:.2f}")
        print()

        # Signals
        print(f"  All Signals:")
        print("  " + "-" * 58)
        for sig_type, dt, price, sc in signals:
            print(f"  {sig_type:<12} {dt[:19]}  ${price:>8.2f}  score={sc:+.1f}")

        # Trades
        closing = [t for t in trades if t["type"].startswith("SELL")]
        if closing:
            print(f"\n  Closed Trades:")
            print("  " + "-" * 58)
            for t in closing:
                pnl_str = f"${t['pnl']:+,.2f}"
                marker = "W" if t["pnl"] > 0 else "L"
                print(f"  [{marker}] {t['type']:<14} {t['datetime'][:19]}  "
                      f"${t['price']:>8.2f}  {pnl_str:>12}")

        # Monthly
        monthly = {}
        for t in closing:
            mk = t["datetime"][:7]
            monthly.setdefault(mk, 0)
            monthly[mk] += t["pnl"]
        if monthly:
            print(f"\n  Monthly P&L:")
            print("  " + "-" * 40)
            max_abs = max(abs(v) for v in monthly.values())
            for m_key in sorted(monthly.keys()):
                bar_len = int(abs(monthly[m_key]) / max(max_abs, 1) * 20)
                bar = ("+" * bar_len if monthly[m_key] > 0 else "-" * bar_len)
                print(f"  {m_key}  ${monthly[m_key]:>+12,.2f}  {bar}")

        print()

    # ── Summary Table ──
    print("\n" + "=" * 72)
    print("  CROSS-TICKER SUMMARY — v2 Optimized Strategy")
    print("=" * 72)
    print(f"  {'Ticker':<8} {'Return':>10} {'B&H':>10} {'Alpha':>10} {'Trades':>8} {'WinRate':>8} {'PF':>6} {'Sharpe':>8} {'MaxDD':>8}")
    print("  " + "-" * 70)

    total_alpha = 0
    for ticker, data in all_data.items():
        bp = best_per_ticker[ticker]
        fc, trades, ec, signals = run_backtest_v2(
            data,
            buy_threshold=bp["thresh"],
            initial_capital=INITIAL,
            base_trail_pct=bp["trail"],
            cooldown_bars=bp["cd"],
            fast_reentry_score=bp["fast_score"],
            profit_lock_trigger=bp["lock_trig"],
            profit_lock_trail=bp["lock_trail"],
        )
        m = compute_metrics(INITIAL, fc, trades, ec, data)
        total_alpha += m["alpha"]
        print(f"  {ticker:<8} {m['total_return_pct']:>+9.2f}% {m['buy_hold_return_pct']:>+9.2f}% "
              f"{m['alpha']:>+9.2f}% {m['total_trades']:>8d} {m['win_rate_pct']:>7.1f}% "
              f"{m['profit_factor']:>5.2f} {m['sharpe_ratio']:>7.2f} {m['max_drawdown_pct']:>7.2f}%")

    print("  " + "-" * 70)
    print(f"  {'TOTAL':>8} {'':>10} {'':>10} {total_alpha:>+9.2f}%")
    print("=" * 72)
    print()
