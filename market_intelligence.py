"""
Market Intelligence & Sentiment Module
========================================
Advanced analytics layer that complements technical analysis:
- Seasonal & day-of-week patterns
- Market event correlation (Fed, CPI, earnings)
- Insider trading & institutional ownership trends
- Short interest & squeeze potential
- Unusual options activity detection
- Earnings price behavior (pre-run, post-gap)
- Sector rotation signals
"""

from collections import defaultdict
import math


# ---------------------------------------------------------------------------
# Seasonal & Calendar Analysis
# ---------------------------------------------------------------------------

def analyze_seasonal_patterns(monthly_returns):
    """
    Identify best/worst months historically.

    Args:
        monthly_returns: dict of {month_number: [list of historical returns]}
                         e.g. {1: [0.02, -0.01, 0.05], 2: [...], ...}
    """
    month_names = [
        "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    stats = {}
    for month, returns in monthly_returns.items():
        if not returns:
            continue
        avg = sum(returns) / len(returns)
        win_rate = sum(1 for r in returns if r > 0) / len(returns) * 100
        stats[month] = {"avg_return": avg, "win_rate": win_rate, "samples": len(returns)}

    ranked = sorted(stats.items(), key=lambda x: x[1]["avg_return"], reverse=True)
    best = ranked[:3] if len(ranked) >= 3 else ranked
    worst = ranked[-3:] if len(ranked) >= 3 else ranked

    return {
        "best_months": [(month_names[m], s) for m, s in best],
        "worst_months": [(month_names[m], s) for m, s in worst],
        "all_months": {month_names[m]: s for m, s in stats.items()},
    }


def analyze_day_of_week(daily_data):
    """
    Detect day-of-week performance bias.

    Args:
        daily_data: list of (day_of_week, return_pct)
                    day_of_week: 0=Mon, 1=Tue, ..., 4=Fri
    """
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    buckets = defaultdict(list)
    for dow, ret in daily_data:
        buckets[dow].append(ret)

    results = {}
    for dow in range(5):
        returns = buckets[dow]
        if not returns:
            continue
        avg = sum(returns) / len(returns)
        win_rate = sum(1 for r in returns if r > 0) / len(returns) * 100
        results[day_names[dow]] = {
            "avg_return": avg,
            "win_rate": win_rate,
            "samples": len(returns),
        }

    # Identify significant patterns (> 1 std dev from mean)
    all_avgs = [v["avg_return"] for v in results.values()]
    if len(all_avgs) >= 3:
        mean_avg = sum(all_avgs) / len(all_avgs)
        std_avg = math.sqrt(sum((a - mean_avg) ** 2 for a in all_avgs) / len(all_avgs))
        patterns = []
        for day, info in results.items():
            if std_avg > 0 and abs(info["avg_return"] - mean_avg) > std_avg:
                bias = "bullish" if info["avg_return"] > mean_avg else "bearish"
                patterns.append(f"{day} shows {bias} bias ({info['avg_return']:+.2%})")
        results["_patterns"] = patterns if patterns else ["No significant day-of-week bias"]
    return results


# ---------------------------------------------------------------------------
# Market Event Correlation
# ---------------------------------------------------------------------------

def analyze_event_correlation(event_returns):
    """
    Measure stock behavior around major market events.

    Args:
        event_returns: dict of {
            "fed_meetings": [(date_str, return_1d, return_5d), ...],
            "cpi_reports":  [(date_str, return_1d, return_5d), ...],
            "fomc_minutes": [(date_str, return_1d, return_5d), ...],
            "jobs_reports": [(date_str, return_1d, return_5d), ...],
        }
    """
    analysis = {}
    for event_type, entries in event_returns.items():
        if not entries:
            continue
        returns_1d = [e[1] for e in entries]
        returns_5d = [e[2] for e in entries]
        avg_1d = sum(returns_1d) / len(returns_1d)
        avg_5d = sum(returns_5d) / len(returns_5d)
        volatility = math.sqrt(sum((r - avg_1d) ** 2 for r in returns_1d) / len(returns_1d))

        if volatility > 0.02:
            sensitivity = "HIGH — stock reacts strongly to this event"
        elif volatility > 0.01:
            sensitivity = "MODERATE — noticeable but manageable impact"
        else:
            sensitivity = "LOW — minimal direct impact"

        analysis[event_type] = {
            "avg_1d_return": avg_1d,
            "avg_5d_return": avg_5d,
            "volatility": volatility,
            "sensitivity": sensitivity,
            "occurrences": len(entries),
        }
    return analysis


# ---------------------------------------------------------------------------
# Insider Trading Analysis
# ---------------------------------------------------------------------------

def analyze_insider_activity(transactions):
    """
    Analyze insider buying/selling patterns.

    Args:
        transactions: list of dicts:
            {"name": str, "role": str, "type": "buy"|"sell",
             "shares": int, "price": float, "date": str}
    """
    if not transactions:
        return {"signal": "NEUTRAL", "interpretation": "No recent insider activity"}

    total_buy_value = 0
    total_sell_value = 0
    buy_count = 0
    sell_count = 0
    notable = []

    for tx in transactions:
        value = tx["shares"] * tx["price"]
        if tx["type"] == "buy":
            total_buy_value += value
            buy_count += 1
            if value > 500_000:
                notable.append(
                    f"{tx['name']} ({tx['role']}) bought ${value:,.0f} on {tx['date']}"
                )
        else:
            total_sell_value += value
            sell_count += 1
            if value > 1_000_000:
                notable.append(
                    f"{tx['name']} ({tx['role']}) sold ${value:,.0f} on {tx['date']}"
                )

    ratio = total_buy_value / max(total_sell_value, 1)

    if ratio > 3 and buy_count >= 3:
        signal = "STRONG BULLISH"
        interpretation = "Heavy insider buying — insiders are confident"
    elif ratio > 1.5:
        signal = "BULLISH"
        interpretation = "Insiders buying more than selling — positive signal"
    elif ratio > 0.5:
        signal = "NEUTRAL"
        interpretation = "Mixed insider activity — no clear direction"
    elif ratio > 0.2:
        signal = "BEARISH"
        interpretation = "Insiders selling more than buying — caution warranted"
    else:
        signal = "STRONG BEARISH"
        interpretation = "Heavy insider selling — potential red flag"

    return {
        "signal": signal,
        "interpretation": interpretation,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "total_buy_value": total_buy_value,
        "total_sell_value": total_sell_value,
        "buy_sell_ratio": ratio,
        "notable_transactions": notable,
    }


# ---------------------------------------------------------------------------
# Institutional Ownership
# ---------------------------------------------------------------------------

def analyze_institutional_ownership(quarterly_data):
    """
    Track whether big funds are accumulating or distributing.

    Args:
        quarterly_data: list of dicts (chronological order):
            {"quarter": str, "institutional_pct": float,
             "num_holders": int, "shares_held": int}
    """
    if len(quarterly_data) < 2:
        return {"trend": "INSUFFICIENT DATA", "interpretation": "Need at least 2 quarters"}

    latest = quarterly_data[-1]
    previous = quarterly_data[-2]

    pct_change = latest["institutional_pct"] - previous["institutional_pct"]
    holder_change = latest["num_holders"] - previous["num_holders"]
    share_change = latest["shares_held"] - previous["shares_held"]

    if pct_change > 2 and holder_change > 0:
        trend = "ACCUMULATING"
        interpretation = "Institutions increasing positions — bullish signal"
    elif pct_change > 0:
        trend = "SLIGHTLY ACCUMULATING"
        interpretation = "Modest institutional buying"
    elif pct_change > -2:
        trend = "STABLE"
        interpretation = "Institutional ownership largely unchanged"
    elif pct_change > -5:
        trend = "DISTRIBUTING"
        interpretation = "Institutions reducing exposure — bearish signal"
    else:
        trend = "HEAVY DISTRIBUTION"
        interpretation = "Significant institutional selling — major caution"

    return {
        "trend": trend,
        "interpretation": interpretation,
        "current_pct": latest["institutional_pct"],
        "pct_change": pct_change,
        "holder_change": holder_change,
        "share_change": share_change,
    }


# ---------------------------------------------------------------------------
# Short Interest & Squeeze Potential
# ---------------------------------------------------------------------------

def analyze_short_interest(short_data):
    """
    Evaluate short squeeze potential.

    Args:
        short_data: dict with:
            "short_pct_float": float (e.g. 15.0 for 15%),
            "short_ratio": float (days to cover),
            "prev_short_pct": float,
            "avg_volume": int,
            "shares_short": int,
            "cost_to_borrow": float (annual % rate),
    """
    pct = short_data.get("short_pct_float", 0)
    ratio = short_data.get("short_ratio", 0)
    prev_pct = short_data.get("prev_short_pct", pct)
    ctb = short_data.get("cost_to_borrow", 0)

    squeeze_score = 0

    # High short interest
    if pct > 20:
        squeeze_score += 3
    elif pct > 10:
        squeeze_score += 2
    elif pct > 5:
        squeeze_score += 1

    # Days to cover
    if ratio > 7:
        squeeze_score += 3
    elif ratio > 4:
        squeeze_score += 2
    elif ratio > 2:
        squeeze_score += 1

    # Increasing short interest (shorts piling in)
    if pct > prev_pct * 1.1:
        squeeze_score += 1

    # High cost to borrow
    if ctb > 50:
        squeeze_score += 2
    elif ctb > 20:
        squeeze_score += 1

    if squeeze_score >= 7:
        potential = "VERY HIGH"
        interpretation = "Extremely crowded short — high squeeze probability"
    elif squeeze_score >= 5:
        potential = "HIGH"
        interpretation = "Elevated short interest with squeeze catalysts present"
    elif squeeze_score >= 3:
        potential = "MODERATE"
        interpretation = "Some short interest but squeeze not imminent"
    else:
        potential = "LOW"
        interpretation = "Manageable short interest — squeeze unlikely"

    return {
        "squeeze_potential": potential,
        "squeeze_score": squeeze_score,
        "interpretation": interpretation,
        "short_pct_float": pct,
        "days_to_cover": ratio,
        "cost_to_borrow": ctb,
        "short_trend": "INCREASING" if pct > prev_pct else "DECREASING",
    }


# ---------------------------------------------------------------------------
# Unusual Options Activity
# ---------------------------------------------------------------------------

def detect_unusual_options(options_flow):
    """
    Flag unusual options activity signals.

    Args:
        options_flow: list of dicts:
            {"type": "call"|"put", "strike": float, "expiry": str,
             "volume": int, "open_interest": int, "premium": float,
             "implied_vol": float}
    """
    signals = []
    total_call_premium = 0
    total_put_premium = 0

    for opt in options_flow:
        vol_oi_ratio = opt["volume"] / max(opt["open_interest"], 1)
        total_premium = opt["premium"] * opt["volume"] * 100

        if opt["type"] == "call":
            total_call_premium += total_premium
        else:
            total_put_premium += total_premium

        # Unusual volume: volume >> open interest
        if vol_oi_ratio > 5 and opt["volume"] > 1000:
            signals.append({
                "alert": "UNUSUAL VOLUME",
                "detail": (
                    f"{opt['type'].upper()} ${opt['strike']} {opt['expiry']} — "
                    f"vol {opt['volume']:,} vs OI {opt['open_interest']:,} "
                    f"(ratio {vol_oi_ratio:.1f}x)"
                ),
                "premium": total_premium,
            })

        # Large single-leg premium
        if total_premium > 500_000:
            signals.append({
                "alert": "LARGE PREMIUM",
                "detail": (
                    f"{opt['type'].upper()} ${opt['strike']} {opt['expiry']} — "
                    f"${total_premium:,.0f} in premium"
                ),
                "premium": total_premium,
            })

        # High implied volatility spike
        if opt["implied_vol"] > 0.8:
            signals.append({
                "alert": "HIGH IV",
                "detail": (
                    f"{opt['type'].upper()} ${opt['strike']} {opt['expiry']} — "
                    f"IV at {opt['implied_vol']:.0%}"
                ),
                "premium": total_premium,
            })

    # Put/Call premium ratio
    pc_ratio = total_put_premium / max(total_call_premium, 1)
    if pc_ratio > 2:
        sentiment = "Heavily bearish options flow"
    elif pc_ratio > 1.2:
        sentiment = "Bearish-leaning options flow"
    elif pc_ratio > 0.8:
        sentiment = "Balanced options flow"
    elif pc_ratio > 0.5:
        sentiment = "Bullish-leaning options flow"
    else:
        sentiment = "Heavily bullish options flow"

    return {
        "signals": signals,
        "put_call_premium_ratio": pc_ratio,
        "sentiment": sentiment,
        "total_call_premium": total_call_premium,
        "total_put_premium": total_put_premium,
    }


# ---------------------------------------------------------------------------
# Earnings Price Behavior
# ---------------------------------------------------------------------------

def analyze_earnings_behavior(earnings_history):
    """
    Analyze pre-run and post-gap patterns around earnings.

    Args:
        earnings_history: list of dicts:
            {"date": str, "eps_surprise_pct": float,
             "pre_5d_return": float, "post_1d_gap": float,
             "post_5d_return": float}
    """
    if not earnings_history:
        return {"pattern": "NO DATA"}

    pre_runs = [e["pre_5d_return"] for e in earnings_history]
    post_gaps = [e["post_1d_gap"] for e in earnings_history]
    post_5d = [e["post_5d_return"] for e in earnings_history]
    surprises = [e["eps_surprise_pct"] for e in earnings_history]

    avg_pre_run = sum(pre_runs) / len(pre_runs)
    avg_post_gap = sum(post_gaps) / len(post_gaps)
    avg_post_5d = sum(post_5d) / len(post_5d)

    # Pre-earnings drift
    pre_pattern = []
    if avg_pre_run > 0.02:
        pre_pattern.append("Stock tends to run UP into earnings (+{:.1%} avg 5-day)".format(avg_pre_run))
    elif avg_pre_run < -0.02:
        pre_pattern.append("Stock tends to drift DOWN into earnings ({:.1%} avg 5-day)".format(avg_pre_run))
    else:
        pre_pattern.append("No consistent pre-earnings drift")

    # Post-earnings gap
    gap_up_rate = sum(1 for g in post_gaps if g > 0) / len(post_gaps) * 100
    if gap_up_rate > 65:
        post_pattern = "Tends to gap UP after earnings ({:.0f}% of the time)".format(gap_up_rate)
    elif gap_up_rate < 35:
        post_pattern = "Tends to gap DOWN after earnings ({:.0f}% gap-down rate)".format(100 - gap_up_rate)
    else:
        post_pattern = "Mixed post-earnings gaps (no strong directional bias)"

    # Gap fade or continuation
    fade_count = sum(
        1 for g, p in zip(post_gaps, post_5d)
        if (g > 0 and p < g) or (g < 0 and p > g)
    )
    fade_rate = fade_count / len(post_gaps) * 100

    return {
        "avg_pre_run": avg_pre_run,
        "avg_post_gap": avg_post_gap,
        "avg_post_5d_return": avg_post_5d,
        "gap_up_rate": gap_up_rate,
        "gap_fade_rate": fade_rate,
        "pre_pattern": pre_pattern,
        "post_pattern": post_pattern,
        "gap_behavior": (
            "Gaps tend to FADE (mean-revert)" if fade_rate > 60
            else "Gaps tend to CONTINUE (momentum)" if fade_rate < 40
            else "Mixed gap follow-through"
        ),
        "quarters_analyzed": len(earnings_history),
    }


# ---------------------------------------------------------------------------
# Sector Rotation
# ---------------------------------------------------------------------------

def analyze_sector_rotation(sector_data):
    """
    Determine sector rotation impact on the stock.

    Args:
        sector_data: dict with:
            "stock_sector": str,
            "sector_returns_1m": dict of {sector_name: return_pct},
            "stock_return_1m": float,
            "sector_etf_return_1m": float,
            "market_return_1m": float,
    """
    stock_sector = sector_data["stock_sector"]
    sector_ret = sector_data["sector_etf_return_1m"]
    market_ret = sector_data["market_return_1m"]
    stock_ret = sector_data["stock_return_1m"]
    all_sectors = sector_data["sector_returns_1m"]

    # Rank sectors
    ranked = sorted(all_sectors.items(), key=lambda x: x[1], reverse=True)
    sector_rank = next(
        (i + 1 for i, (name, _) in enumerate(ranked) if name == stock_sector),
        None,
    )

    # Relative strength
    vs_sector = stock_ret - sector_ret
    vs_market = stock_ret - market_ret

    # Rotation phase
    if sector_ret > market_ret and sector_ret > 0:
        phase = "LEADING — money rotating INTO this sector"
    elif sector_ret > 0 and sector_ret <= market_ret:
        phase = "WEAKENING — sector up but lagging the market"
    elif sector_ret <= 0 and sector_ret > market_ret:
        phase = "IMPROVING — sector down less than market"
    else:
        phase = "LAGGING — money rotating OUT of this sector"

    signals = []
    if vs_sector > 0.03:
        signals.append("Stock outperforming its sector — relative strength leader")
    elif vs_sector < -0.03:
        signals.append("Stock underperforming its sector — relative weakness")

    if phase.startswith("LEADING"):
        signals.append("Sector tailwind — favorable rotation for this stock")
    elif phase.startswith("LAGGING"):
        signals.append("Sector headwind — rotation unfavorable")

    top_sectors = [(name, ret) for name, ret in ranked[:3]]
    bottom_sectors = [(name, ret) for name, ret in ranked[-3:]]

    return {
        "sector": stock_sector,
        "rotation_phase": phase,
        "sector_rank": f"{sector_rank}/{len(ranked)}" if sector_rank else "N/A",
        "stock_vs_sector": vs_sector,
        "stock_vs_market": vs_market,
        "signals": signals,
        "top_sectors": top_sectors,
        "bottom_sectors": bottom_sectors,
    }


# ---------------------------------------------------------------------------
# Composite Market Intelligence Score
# ---------------------------------------------------------------------------

def compute_intelligence_score(insider, institutional, short_info, options, earnings, sector):
    """Aggregate all market intelligence into a single directional score."""
    score = 0

    # Insider signal
    insider_map = {"STRONG BULLISH": 2, "BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1, "STRONG BEARISH": -2}
    score += insider_map.get(insider.get("signal", "NEUTRAL"), 0)

    # Institutional trend
    inst_map = {"ACCUMULATING": 1.5, "SLIGHTLY ACCUMULATING": 0.5, "STABLE": 0,
                "DISTRIBUTING": -1, "HEAVY DISTRIBUTION": -2}
    score += inst_map.get(institutional.get("trend", "STABLE"), 0)

    # Short squeeze adds bullish pressure if high
    squeeze_map = {"VERY HIGH": 2, "HIGH": 1, "MODERATE": 0.5, "LOW": 0}
    score += squeeze_map.get(short_info.get("squeeze_potential", "LOW"), 0)

    # Options sentiment
    pc = options.get("put_call_premium_ratio", 1)
    if pc < 0.5:
        score += 1.5
    elif pc < 0.8:
        score += 0.5
    elif pc > 2:
        score -= 1.5
    elif pc > 1.2:
        score -= 0.5

    # Earnings momentum
    gap_up = earnings.get("gap_up_rate", 50)
    if gap_up > 65:
        score += 1
    elif gap_up < 35:
        score -= 1

    # Sector rotation
    if sector.get("rotation_phase", "").startswith("LEADING"):
        score += 1
    elif sector.get("rotation_phase", "").startswith("LAGGING"):
        score -= 1

    if score >= 4:
        signal = "STRONG BUY"
    elif score >= 2:
        signal = "BUY"
    elif score > -2:
        signal = "HOLD"
    elif score > -4:
        signal = "SELL"
    else:
        signal = "STRONG SELL"

    return {"score": score, "signal": signal}


# ---------------------------------------------------------------------------
# Full Report Printer
# ---------------------------------------------------------------------------

def print_market_intelligence_report(
    seasonal, day_of_week, event_corr, insider, institutional,
    short_info, options, earnings, sector,
):
    """Print a formatted market intelligence report."""

    print("=" * 64)
    print("         MARKET INTELLIGENCE & SENTIMENT REPORT")
    print("=" * 64)

    # Seasonal
    print("\n--- Seasonal Patterns ---")
    if seasonal:
        print("  Best months:")
        for name, s in seasonal.get("best_months", []):
            print(f"    {name}: avg {s['avg_return']:+.2%} (win rate {s['win_rate']:.0f}%)")
        print("  Worst months:")
        for name, s in seasonal.get("worst_months", []):
            print(f"    {name}: avg {s['avg_return']:+.2%} (win rate {s['win_rate']:.0f}%)")

    # Day of week
    print("\n--- Day-of-Week Patterns ---")
    if day_of_week:
        patterns = day_of_week.get("_patterns", [])
        for p in patterns:
            print(f"  >> {p}")
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            info = day_of_week.get(day)
            if info:
                print(f"  {day:>10}: {info['avg_return']:+.3%}  (win {info['win_rate']:.0f}%)")

    # Event correlation
    print("\n--- Market Event Correlation ---")
    if event_corr:
        for event, info in event_corr.items():
            print(f"  {event}:")
            print(f"    1-day avg: {info['avg_1d_return']:+.2%}  |  "
                  f"5-day avg: {info['avg_5d_return']:+.2%}")
            print(f"    Sensitivity: {info['sensitivity']}")

    # Insider
    print("\n--- Insider Activity ---")
    print(f"  Signal: {insider.get('signal', 'N/A')}")
    print(f"  {insider.get('interpretation', '')}")
    if insider.get("notable_transactions"):
        for tx in insider["notable_transactions"]:
            print(f"    * {tx}")
    bc = insider.get("buy_count", 0)
    sc = insider.get("sell_count", 0)
    if bc or sc:
        print(f"  Buys: {bc} (${insider.get('total_buy_value', 0):,.0f})  |  "
              f"Sells: {sc} (${insider.get('total_sell_value', 0):,.0f})")

    # Institutional
    print("\n--- Institutional Ownership ---")
    print(f"  Trend: {institutional.get('trend', 'N/A')}")
    print(f"  {institutional.get('interpretation', '')}")
    if institutional.get("current_pct") is not None:
        print(f"  Ownership: {institutional['current_pct']:.1f}% "
              f"({institutional.get('pct_change', 0):+.1f}% QoQ)")

    # Short interest
    print("\n--- Short Interest & Squeeze ---")
    print(f"  Squeeze potential: {short_info.get('squeeze_potential', 'N/A')}")
    print(f"  {short_info.get('interpretation', '')}")
    print(f"  Short % of float: {short_info.get('short_pct_float', 0):.1f}%  |  "
          f"Days to cover: {short_info.get('days_to_cover', 0):.1f}")
    print(f"  Cost to borrow: {short_info.get('cost_to_borrow', 0):.1f}%  |  "
          f"Trend: {short_info.get('short_trend', 'N/A')}")

    # Options
    print("\n--- Unusual Options Activity ---")
    print(f"  Sentiment: {options.get('sentiment', 'N/A')}")
    print(f"  Put/Call premium ratio: {options.get('put_call_premium_ratio', 0):.2f}")
    for sig in options.get("signals", []):
        print(f"    ! [{sig['alert']}] {sig['detail']}")

    # Earnings
    print("\n--- Earnings Price Behavior ---")
    for p in earnings.get("pre_pattern", []):
        print(f"  {p}")
    print(f"  {earnings.get('post_pattern', 'N/A')}")
    print(f"  Gap behavior: {earnings.get('gap_behavior', 'N/A')}")
    print(f"  Quarters analyzed: {earnings.get('quarters_analyzed', 0)}")

    # Sector rotation
    print("\n--- Sector Rotation ---")
    print(f"  Sector: {sector.get('sector', 'N/A')}  |  "
          f"Rank: {sector.get('sector_rank', 'N/A')}")
    print(f"  Phase: {sector.get('rotation_phase', 'N/A')}")
    for s in sector.get("signals", []):
        print(f"  >> {s}")
    if sector.get("top_sectors"):
        print(f"  Hot sectors: {', '.join(f'{n} ({r:+.1%})' for n, r in sector['top_sectors'])}")

    # Composite
    composite = compute_intelligence_score(
        insider, institutional, short_info, options, earnings, sector,
    )
    print()
    print("=" * 64)
    print(f"  Intelligence Score : {composite['score']:+.1f}")
    print(f"  Signal             : {composite['signal']}")
    print("=" * 64)
    print()

    return composite


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Seasonal ---
    import random
    random.seed(99)
    monthly_ret = {m: [random.gauss(0.01 if m in (1, 4, 11) else -0.005, 0.04)
                       for _ in range(10)] for m in range(1, 13)}
    seasonal = analyze_seasonal_patterns(monthly_ret)

    # --- Day of week ---
    dow_data = [(d, random.gauss(0.001 if d in (0, 4) else -0.0005, 0.01))
                for _ in range(200) for d in range(5)]
    day_of_week = analyze_day_of_week(dow_data)

    # --- Events ---
    event_rets = {
        "fed_meetings": [(f"2025-{m:02d}-15", random.gauss(0, 0.015), random.gauss(0.002, 0.02))
                         for m in range(1, 13, 2)],
        "cpi_reports": [(f"2025-{m:02d}-10", random.gauss(-0.005, 0.02), random.gauss(0, 0.025))
                        for m in range(1, 13)],
    }
    event_corr = analyze_event_correlation(event_rets)

    # --- Insider ---
    insider = analyze_insider_activity([
        {"name": "Jane CEO", "role": "CEO", "type": "buy", "shares": 10000, "price": 145.0, "date": "2025-11-01"},
        {"name": "John CFO", "role": "CFO", "type": "buy", "shares": 5000, "price": 142.0, "date": "2025-11-15"},
        {"name": "VP Sales", "role": "VP", "type": "sell", "shares": 2000, "price": 150.0, "date": "2025-12-01"},
    ])

    # --- Institutional ---
    institutional = analyze_institutional_ownership([
        {"quarter": "Q2 2025", "institutional_pct": 72.5, "num_holders": 450, "shares_held": 50_000_000},
        {"quarter": "Q3 2025", "institutional_pct": 75.2, "num_holders": 465, "shares_held": 52_000_000},
    ])

    # --- Short interest ---
    short_info = analyze_short_interest({
        "short_pct_float": 18.5,
        "short_ratio": 5.2,
        "prev_short_pct": 15.0,
        "avg_volume": 3_000_000,
        "shares_short": 12_000_000,
        "cost_to_borrow": 35.0,
    })

    # --- Options ---
    options = detect_unusual_options([
        {"type": "call", "strike": 160, "expiry": "2026-01-16", "volume": 8500,
         "open_interest": 1200, "premium": 3.50, "implied_vol": 0.55},
        {"type": "call", "strike": 170, "expiry": "2026-02-20", "volume": 5000,
         "open_interest": 800, "premium": 2.10, "implied_vol": 0.62},
        {"type": "put", "strike": 130, "expiry": "2026-01-16", "volume": 3000,
         "open_interest": 2500, "premium": 1.80, "implied_vol": 0.48},
    ])

    # --- Earnings ---
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

    # --- Sector rotation ---
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

    print_market_intelligence_report(
        seasonal, day_of_week, event_corr, insider, institutional,
        short_info, options, earnings, sector,
    )
