"""
Technical Analysis Trading Strategy
====================================
Combines multiple indicators for signal generation:
- Moving Averages (50/100/200-day) & crossover signals
- RSI, MACD, Bollinger Bands
- Volume trend analysis (buyer vs seller strength)
- Support/Resistance levels
- Chart pattern identification
- Fibonacci retracement levels
"""

import math


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def sma(prices, period):
    """Simple Moving Average."""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def ema(prices, period):
    """Exponential Moving Average."""
    if len(prices) < period:
        return None
    k = 2 / (period + 1)
    result = sum(prices[:period]) / period
    for price in prices[period:]:
        result = price * k + result * (1 - k)
    return result


def ema_series(prices, period):
    """Return full EMA series (None-padded for initial values)."""
    series = [None] * (period - 1)
    k = 2 / (period + 1)
    result = sum(prices[:period]) / period
    series.append(result)
    for price in prices[period:]:
        result = price * k + result * (1 - k)
        series.append(result)
    return series


# ---------------------------------------------------------------------------
# Indicator calculations
# ---------------------------------------------------------------------------

def compute_rsi(prices, period=14):
    """Relative Strength Index (0-100)."""
    if len(prices) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(prices)):
        delta = prices[i] - prices[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(prices, fast=12, slow=26, signal_period=9):
    """MACD line, signal line, and histogram."""
    fast_ema = ema_series(prices, fast)
    slow_ema = ema_series(prices, slow)
    macd_line = []
    for f, s in zip(fast_ema, slow_ema):
        macd_line.append(f - s if f is not None and s is not None else None)
    valid_macd = [v for v in macd_line if v is not None]
    if len(valid_macd) < signal_period:
        return None, None, None
    signal_line = ema_series(valid_macd, signal_period)
    current_macd = valid_macd[-1]
    current_signal = signal_line[-1] if signal_line[-1] is not None else 0
    histogram = current_macd - current_signal
    return current_macd, current_signal, histogram


def compute_bollinger(prices, period=20, num_std=2):
    """Bollinger Bands: upper, middle, lower."""
    if len(prices) < period:
        return None, None, None
    middle = sma(prices, period)
    window = prices[-period:]
    variance = sum((p - middle) ** 2 for p in window) / period
    std = math.sqrt(variance)
    return middle + num_std * std, middle, middle - num_std * std


def compute_volume_trend(volumes, period=20):
    """
    Volume trend analysis.
    Returns ratio of recent avg volume to longer-term avg and interpretation.
    """
    if len(volumes) < period:
        return None, "Insufficient data"
    recent = sum(volumes[-period // 2:]) / (period // 2)
    longer = sum(volumes[-period:]) / period
    if longer == 0:
        return None, "No volume"
    ratio = recent / longer
    if ratio > 1.5:
        interpretation = "Strong buying pressure — bulls dominating"
    elif ratio > 1.1:
        interpretation = "Moderate buying interest — slight bullish bias"
    elif ratio > 0.9:
        interpretation = "Balanced volume — no clear winner"
    elif ratio > 0.6:
        interpretation = "Declining volume — sellers gaining ground"
    else:
        interpretation = "Very low volume — watch for breakout or breakdown"
    return ratio, interpretation


# ---------------------------------------------------------------------------
# Support / Resistance
# ---------------------------------------------------------------------------

def find_support_resistance(prices, window=5):
    """Identify local support and resistance levels from price pivots."""
    supports, resistances = [], []
    for i in range(window, len(prices) - window):
        region = prices[i - window: i + window + 1]
        if prices[i] == min(region):
            supports.append(prices[i])
        if prices[i] == max(region):
            resistances.append(prices[i])
    supports = sorted(set(supports))[-3:] if supports else []
    resistances = sorted(set(resistances))[-3:] if resistances else []
    return supports, resistances


# ---------------------------------------------------------------------------
# Fibonacci Retracement
# ---------------------------------------------------------------------------

def fibonacci_levels(high, low):
    """Compute Fibonacci retracement levels from swing high/low."""
    diff = high - low
    levels = {
        "0.0% (High)": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.500 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
        "100% (Low)": low,
    }
    return levels


# ---------------------------------------------------------------------------
# Chart Pattern Detection (simplified)
# ---------------------------------------------------------------------------

def detect_patterns(prices):
    """Basic chart pattern identification."""
    patterns = []
    n = len(prices)
    if n < 30:
        return ["Insufficient data for pattern detection"]

    # Head and Shoulders: look for peak-higher_peak-peak shape
    seg = n // 5
    for start in range(0, n - 5 * seg, seg):
        chunk = prices[start: start + 5 * seg]
        peaks = [max(chunk[i * seg:(i + 1) * seg]) for i in range(5)]
        if (peaks[2] > peaks[0] and peaks[2] > peaks[4]
                and abs(peaks[0] - peaks[4]) / peaks[2] < 0.03):
            patterns.append("Head and Shoulders (potential reversal)")
            break

    # Double bottom
    lows = []
    for i in range(5, n - 5):
        region = prices[i - 5: i + 6]
        if prices[i] == min(region):
            lows.append((i, prices[i]))
    for i in range(len(lows) - 1):
        if abs(lows[i][1] - lows[i + 1][1]) / lows[i][1] < 0.02:
            patterns.append("Double Bottom (bullish reversal)")
            break

    # Cup and Handle: U-shape followed by small dip
    mid = n // 2
    left_half = prices[:mid]
    right_half = prices[mid:]
    if (left_half[0] > min(left_half) < right_half[-1]
            and abs(left_half[0] - right_half[-1]) / left_half[0] < 0.05):
        if len(right_half) > 5 and min(right_half[-5:]) < right_half[-1]:
            patterns.append("Cup and Handle (bullish continuation)")

    if not patterns:
        patterns.append("No clear pattern detected")
    return patterns


# ---------------------------------------------------------------------------
# Moving Average Crossover Signals
# ---------------------------------------------------------------------------

def ma_crossover_signals(prices):
    """Detect golden/death crosses among 50/100/200-day MAs."""
    signals = []
    ma50 = sma(prices, 50)
    ma100 = sma(prices, 100)
    ma200 = sma(prices, 200)

    if ma50 and ma100:
        if ma50 > ma100:
            signals.append("50-day MA above 100-day MA — short-term bullish")
        else:
            signals.append("50-day MA below 100-day MA — short-term bearish")

    if ma50 and ma200:
        if ma50 > ma200:
            signals.append("Golden Cross (50 > 200) — long-term bullish")
        else:
            signals.append("Death Cross (50 < 200) — long-term bearish")

    return {"MA50": ma50, "MA100": ma100, "MA200": ma200, "signals": signals}


# ---------------------------------------------------------------------------
# Composite Signal Engine
# ---------------------------------------------------------------------------

def interpret_rsi(rsi):
    if rsi is None:
        return "N/A", 0
    if rsi > 70:
        return f"RSI {rsi:.1f} — Overbought, potential pullback ahead", -1
    if rsi > 55:
        return f"RSI {rsi:.1f} — Bullish momentum", 1
    if rsi > 45:
        return f"RSI {rsi:.1f} — Neutral zone", 0
    if rsi > 30:
        return f"RSI {rsi:.1f} — Bearish momentum", -1
    return f"RSI {rsi:.1f} — Oversold, potential bounce ahead", 1


def interpret_macd(macd_val, signal_val, histogram):
    if macd_val is None:
        return "N/A", 0
    if histogram > 0 and macd_val > 0:
        return f"MACD bullish — histogram positive ({histogram:.2f}), above zero line", 1
    if histogram > 0:
        return f"MACD improving — histogram positive ({histogram:.2f}) but below zero", 0.5
    if histogram < 0 and macd_val < 0:
        return f"MACD bearish — histogram negative ({histogram:.2f}), below zero line", -1
    return f"MACD weakening — histogram negative ({histogram:.2f})", -0.5


def interpret_bollinger(price, upper, middle, lower):
    if upper is None:
        return "N/A", 0
    if price >= upper:
        return f"Price at upper Bollinger Band ({upper:.2f}) — overbought / breakout", -0.5
    if price <= lower:
        return f"Price at lower Bollinger Band ({lower:.2f}) — oversold / breakdown", 0.5
    pct = (price - lower) / (upper - lower) * 100
    return f"Price at {pct:.0f}% of Bollinger range — mid-band area", 0


def generate_signal(score):
    """Map composite score to trading action."""
    if score >= 3:
        return "STRONG BUY"
    if score >= 1.5:
        return "BUY"
    if score > -1.5:
        return "HOLD"
    if score > -3:
        return "SELL"
    return "STRONG SELL"


# ---------------------------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------------------------

def run_analysis(prices, volumes):
    """Run full technical analysis and print a report."""
    if len(prices) < 200:
        print("Warning: fewer than 200 data points — some indicators may be unavailable.\n")

    current_price = prices[-1]

    # Moving Averages
    ma_info = ma_crossover_signals(prices)

    # RSI
    rsi = compute_rsi(prices)
    rsi_text, rsi_score = interpret_rsi(rsi)

    # MACD
    macd_val, signal_val, histogram = compute_macd(prices)
    macd_text, macd_score = interpret_macd(macd_val, signal_val, histogram)

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = compute_bollinger(prices)
    bb_text, bb_score = interpret_bollinger(current_price, bb_upper, bb_middle, bb_lower)

    # Volume
    vol_ratio, vol_text = compute_volume_trend(volumes)

    # Support / Resistance
    supports, resistances = find_support_resistance(prices)

    # Fibonacci
    swing_high = max(prices[-60:]) if len(prices) >= 60 else max(prices)
    swing_low = min(prices[-60:]) if len(prices) >= 60 else min(prices)
    fib = fibonacci_levels(swing_high, swing_low)

    # Patterns
    patterns = detect_patterns(prices)

    # Volume score
    vol_score = 0
    if vol_ratio and vol_ratio > 1.3:
        vol_score = 1
    elif vol_ratio and vol_ratio < 0.7:
        vol_score = -1

    # MA score
    ma_score = 0
    if ma_info["MA50"] and ma_info["MA200"]:
        ma_score += 1 if ma_info["MA50"] > ma_info["MA200"] else -1
    if ma_info["MA50"] and ma_info["MA100"]:
        ma_score += 0.5 if ma_info["MA50"] > ma_info["MA100"] else -0.5

    composite = rsi_score + macd_score + bb_score + vol_score + ma_score
    action = generate_signal(composite)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("       TECHNICAL ANALYSIS REPORT")
    print("=" * 60)
    print(f"  Current Price: {current_price:.2f}")
    print()

    print("--- Moving Averages ---")
    for key in ("MA50", "MA100", "MA200"):
        val = ma_info[key]
        print(f"  {key}: {val:.2f}" if val else f"  {key}: N/A")
    for sig in ma_info["signals"]:
        print(f"  >> {sig}")
    print()

    print("--- RSI ---")
    print(f"  {rsi_text}")
    print()

    print("--- MACD ---")
    print(f"  {macd_text}")
    print()

    print("--- Bollinger Bands ---")
    if bb_upper:
        print(f"  Upper: {bb_upper:.2f}  |  Middle: {bb_middle:.2f}  |  Lower: {bb_lower:.2f}")
    print(f"  {bb_text}")
    print()

    print("--- Volume Trend ---")
    if vol_ratio:
        print(f"  Recent/Average ratio: {vol_ratio:.2f}")
    print(f"  {vol_text}")
    print()

    print("--- Support & Resistance ---")
    print(f"  Support levels:    {', '.join(f'{s:.2f}' for s in supports) or 'N/A'}")
    print(f"  Resistance levels: {', '.join(f'{r:.2f}' for r in resistances) or 'N/A'}")
    print()

    print("--- Fibonacci Retracement ---")
    for label, level in fib.items():
        marker = " <-- current" if abs(level - current_price) / current_price < 0.01 else ""
        print(f"  {label:>16}: {level:.2f}{marker}")
    print()

    print("--- Chart Patterns ---")
    for p in patterns:
        print(f"  * {p}")
    print()

    print("=" * 60)
    print(f"  Composite Score : {composite:+.1f}")
    print(f"  Signal          : {action}")
    print("=" * 60)
    print()
    print("Disclaimer: This is for educational purposes only.")
    print("Always do your own research before trading.")

    return {
        "price": current_price,
        "signal": action,
        "score": composite,
        "rsi": rsi,
        "macd": (macd_val, signal_val, histogram),
        "bollinger": (bb_upper, bb_middle, bb_lower),
        "volume_ratio": vol_ratio,
        "supports": supports,
        "resistances": resistances,
        "fibonacci": fib,
        "patterns": patterns,
    }


# ---------------------------------------------------------------------------
# Demo with sample data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    random.seed(42)

    # Generate 250 days of simulated price data (random walk)
    price = 150.0
    sample_prices = []
    sample_volumes = []
    for _ in range(250):
        price += random.gauss(0.05, 2.0)
        price = max(price, 10)
        sample_prices.append(round(price, 2))
        sample_volumes.append(random.randint(500_000, 5_000_000))

    ta_result = run_analysis(sample_prices, sample_volumes)

    # -----------------------------------------------------------------------
    # Market Intelligence layer
    # -----------------------------------------------------------------------
    from market_intelligence import (
        analyze_seasonal_patterns,
        analyze_day_of_week,
        analyze_event_correlation,
        analyze_insider_activity,
        analyze_institutional_ownership,
        analyze_short_interest,
        detect_unusual_options,
        analyze_earnings_behavior,
        analyze_sector_rotation,
        print_market_intelligence_report,
    )

    monthly_ret = {m: [random.gauss(0.01 if m in (1, 4, 11) else -0.005, 0.04)
                       for _ in range(10)] for m in range(1, 13)}
    seasonal = analyze_seasonal_patterns(monthly_ret)

    dow_data = [(d, random.gauss(0.001 if d in (0, 4) else -0.0005, 0.01))
                for _ in range(200) for d in range(5)]
    day_of_week = analyze_day_of_week(dow_data)

    event_rets = {
        "fed_meetings": [(f"2025-{m:02d}-15", random.gauss(0, 0.015), random.gauss(0.002, 0.02))
                         for m in range(1, 13, 2)],
        "cpi_reports": [(f"2025-{m:02d}-10", random.gauss(-0.005, 0.02), random.gauss(0, 0.025))
                        for m in range(1, 13)],
    }
    event_corr = analyze_event_correlation(event_rets)

    insider = analyze_insider_activity([
        {"name": "Jane CEO", "role": "CEO", "type": "buy", "shares": 10000, "price": 145.0, "date": "2025-11-01"},
        {"name": "John CFO", "role": "CFO", "type": "buy", "shares": 5000, "price": 142.0, "date": "2025-11-15"},
        {"name": "VP Sales", "role": "VP", "type": "sell", "shares": 2000, "price": 150.0, "date": "2025-12-01"},
    ])

    institutional = analyze_institutional_ownership([
        {"quarter": "Q2 2025", "institutional_pct": 72.5, "num_holders": 450, "shares_held": 50_000_000},
        {"quarter": "Q3 2025", "institutional_pct": 75.2, "num_holders": 465, "shares_held": 52_000_000},
    ])

    short_info = analyze_short_interest({
        "short_pct_float": 18.5, "short_ratio": 5.2, "prev_short_pct": 15.0,
        "avg_volume": 3_000_000, "shares_short": 12_000_000, "cost_to_borrow": 35.0,
    })

    options = detect_unusual_options([
        {"type": "call", "strike": 160, "expiry": "2026-01-16", "volume": 8500,
         "open_interest": 1200, "premium": 3.50, "implied_vol": 0.55},
        {"type": "put", "strike": 130, "expiry": "2026-01-16", "volume": 3000,
         "open_interest": 2500, "premium": 1.80, "implied_vol": 0.48},
    ])

    earnings = analyze_earnings_behavior([
        {"date": "2025-01-25", "eps_surprise_pct": 5.2, "pre_5d_return": 0.03,
         "post_1d_gap": 0.04, "post_5d_return": 0.02},
        {"date": "2025-04-24", "eps_surprise_pct": -1.0, "pre_5d_return": 0.02,
         "post_1d_gap": -0.03, "post_5d_return": -0.05},
        {"date": "2025-07-24", "eps_surprise_pct": 3.8, "pre_5d_return": 0.04,
         "post_1d_gap": 0.06, "post_5d_return": 0.08},
        {"date": "2025-10-23", "eps_surprise_pct": 2.1, "pre_5d_return": 0.01,
         "post_1d_gap": 0.02, "post_5d_return": 0.01},
    ])

    sector = analyze_sector_rotation({
        "stock_sector": "Technology",
        "sector_returns_1m": {
            "Technology": 0.05, "Healthcare": 0.02, "Financials": 0.03,
            "Energy": -0.02, "Consumer Disc.": 0.01, "Industrials": 0.04,
            "Utilities": -0.01, "Real Estate": -0.03, "Materials": 0.00,
            "Comm. Services": 0.03, "Consumer Staples": 0.01,
        },
        "stock_return_1m": 0.07,
        "sector_etf_return_1m": 0.05,
        "market_return_1m": 0.02,
    })

    mi_result = print_market_intelligence_report(
        seasonal, day_of_week, event_corr, insider, institutional,
        short_info, options, earnings, sector,
    )

    # Combined final verdict
    combined = ta_result["score"] + mi_result["score"]
    if combined >= 5:
        verdict = "STRONG BUY"
    elif combined >= 2:
        verdict = "BUY"
    elif combined > -2:
        verdict = "HOLD"
    elif combined > -5:
        verdict = "SELL"
    else:
        verdict = "STRONG SELL"

    print("=" * 64)
    print("         COMBINED VERDICT (Technical + Intelligence)")
    print("=" * 64)
    print(f"  Technical Score    : {ta_result['score']:+.1f}  ({ta_result['signal']})")
    print(f"  Intelligence Score : {mi_result['score']:+.1f}  ({mi_result['signal']})")
    print(f"  Combined Score     : {combined:+.1f}")
    print(f"  FINAL SIGNAL       : {verdict}")
    print("=" * 64)
