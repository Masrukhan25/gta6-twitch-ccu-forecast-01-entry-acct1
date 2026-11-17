"""gta6-twitch-ccu-forecast-01 entry for @arminalyun (account 1): a weighted-quantile
forecaster over past big game launches.

For each hour h = 2..24 of the release day, the forecast is five quantiles of that
hour's Twitch viewers, built from the release-day shapes of past launches
("comparables"). Each comparable contributes its log-ratio l_h = ln(viewers_h /
viewers_1); the forecast is p1 * exp(weighted quantile of l_h), where p1 is the
target's hour-1 print. A comparable's weight is a similarity kernel over:
  - platform match (console-only vs multi-platform; GTA VI is console-only),
  - recency (exponential decay, 3-year half-life),
  - log hour-1 viewers distance (Gaussian, bandwidth 0.7),
  - pre-release buzz: the comparable's mean hourly viewers in the 24 h before its
    release day, relative to its hour-1 print, vs the target's (Gaussian on the log
    ratio, bandwidth 2.0),
  - tier (comparable 1.0, early-access-contaminated 0.5, weak 0.3).
The target's own category history (the Grand Theft Auto VI comparable, which has no
release day) never enters the shape pool; it is only the target's pre-release buzz
via history_hourly.

The kernel and the choice to use no per-hour bias correction and no spread scaling
come from leave-one-out cross-validation over the comparables: the bias correction
and spread scaling both raised the CV loss, so they are not applied. CV (mean
pinball loss): all comparables 0.41, relevant subset (console-only or hour-1
>= 10k) 0.28, practice-like targets (big console-only, clean tier) 0.17, against
the house persistence baseline's 0.93 / 0.36 / ~0.36 on the same sets.

Fallback: if no comparable has a usable release day, forecast persistence with the
house spread (0.15 * sqrt(h - 1) in log space).

Standard library only, deterministic, no network. Reads $MAKEFUN_INPUT_DIR (else
/input) for *.in files, writes $MAKEFUN_OUTPUT_DIR (else /output) <id>.out.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
FALLBACK_SIGMA = 0.15

# Similarity-kernel parameters (tuned by leave-one-out CV over the comparables).
PLAT_MISMATCH = 0.4      # weight when console-only status differs
HALFLIFE_DAYS = 3 * 365  # recency exponential decay half-life
H1_BW = 0.7              # Gaussian bandwidth on log hour-1 viewers distance
BUZZ_BW = 2.0            # Gaussian bandwidth on log pre-release buzz ratio distance
TIER_WEIGHT = {"comparable": 1.0, "early-access": 0.5, "weak": 0.3}


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def parse_ts(ts):
    """Parse an ISO 8601 UTC timestamp like 2026-09-15T00:00:00Z."""
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")


def weighted_quantile(pairs, tau):
    """Weighted tau-quantile of (value, weight) pairs, by cumulative weight."""
    ordered = sorted(pairs, key=lambda p: p[0])
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    cum = 0.0
    for value, w in ordered:
        cum += w
        if cum >= tau * total:
            return value
    return ordered[-1][0]


def tier_of(comp):
    """The research tier recorded in a comparable's unlock note."""
    note = comp.get("unlock", "") or ""
    if "target-history" in note:
        return "target-history"
    if "early-access-contaminated" in note:
        return "early-access"
    if "Tier: weak" in note:
        return "weak"
    return "comparable"


def console_only(comp):
    """True when the launch had no PC release (GTA VI is console-only)."""
    return "PC" not in str(comp.get("platforms", ""))


def release_day_hours(comp):
    """Map hour (1..24) -> viewers for a comparable's release day (UTC day)."""
    try:
        day = parse_ts(comp["day_start"])
    except (KeyError, ValueError):
        return {}
    out = {}
    for point in comp.get("hourly", []) or []:
        try:
            t = parse_ts(point["from"])
            viewers = point["viewers"]
        except (KeyError, ValueError, TypeError):
            continue
        if day <= t < day + timedelta(hours=24):
            hour = int((t - day).total_seconds() // 3600) + 1
            if 1 <= hour <= 24 and isinstance(viewers, (int, float)):
                out[hour] = max(1, viewers)
    return out


def pre24_ratio(hourly, day_start, hour1):
    """Mean viewers in the 24 h before the release day, relative to hour 1.

    `hourly` is a list of {from, viewers} points; `day_start` the release day's
    first instant; `hour1` the hour-1 print. Returns None when there is no usable
    pre-release point or no hour-1 print.
    """
    if not hour1 or hour1 < 1:
        return None
    try:
        day = parse_ts(day_start)
    except (KeyError, ValueError, TypeError):
        return None
    vals = []
    for point in hourly or []:
        try:
            t = parse_ts(point["from"])
            viewers = point["viewers"]
        except (KeyError, ValueError, TypeError):
            continue
        if day - timedelta(hours=24) <= t < day and isinstance(viewers, (int, float)):
            vals.append(max(1, viewers))
    if not vals:
        return None
    return (sum(vals) / len(vals)) / hour1


def extract_comparable(comp):
    """Reduce one comparable to the features the model needs, or None to skip it.

    Skipped: the target's own category history (no release day), any launch with
    no valid hour-1 print, and any launch with fewer than 12 of hours 2..24.
    """
    if tier_of(comp) == "target-history":
        return None
    day_hours = release_day_hours(comp)
    hour1 = day_hours.get(1)
    if not hour1 or hour1 < 1:
        return None
    if len([h for h in range(2, 25) if h in day_hours]) < 12:
        return None
    try:
        day = parse_ts(comp["day_start"])
    except (KeyError, ValueError):
        return None
    return {
        "name": comp.get("name", "?"),
        "day": day,
        "tier": tier_of(comp),
        "console_only": console_only(comp),
        "hour1": hour1,
        "hours": day_hours,
        "pre24": pre24_ratio(comp.get("hourly", []), comp["day_start"], hour1),
    }


def similarity(target, pool_entry):
    """Similarity weight of one pool comparable for the target."""
    w = 1.0
    if target["console_only"] != pool_entry["console_only"]:
        w *= PLAT_MISMATCH
    age_days = abs((target["day"] - pool_entry["day"]).days)
    w *= math.exp(-math.log(2) * age_days / HALFLIFE_DAYS)
    d_h1 = math.log(target["hour1"]) - math.log(pool_entry["hour1"])
    w *= math.exp(-0.5 * (d_h1 / H1_BW) ** 2)
    t_buzz = target.get("pre24")
    p_buzz = pool_entry.get("pre24")
    if t_buzz and p_buzz and t_buzz > 0 and p_buzz > 0:
        d_buzz = math.log(t_buzz) - math.log(p_buzz)
        w *= math.exp(-0.5 * (d_buzz / BUZZ_BW) ** 2)
    w *= TIER_WEIGHT.get(pool_entry["tier"], 1.0)
    return w


def fallback_quantiles(p1, hour):
    """House persistence forecast: every hour at the hour-1 print, widening spread."""
    return [round2(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(hour - 1))) for z in Z]


def forecast_hours(target, pool, hours):
    """Five quantiles per forecast hour from the weighted comparable shapes."""
    p1 = max(1, target["hour1"])
    out = {}
    for hour in hours:
        pairs = []
        for entry in pool:
            if hour in entry["hours"]:
                log_ratio = math.log(entry["hours"][hour] / entry["hour1"])
                pairs.append((log_ratio, similarity(target, entry)))
        if len(pairs) < 3:
            # Too few comparables report this hour: fall back to persistence
            # rather than trusting a quantile from one or two shapes.
            out[hour] = fallback_quantiles(p1, hour)
            continue
        quantiles = []
        for tau in LEVELS:
            log_q = weighted_quantile(pairs, tau)
            quantiles.append(p1 * math.exp(log_q))
        for i in range(1, 5):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        out[hour] = [round2(min(1e9, max(1.0, q))) for q in quantiles]
    return out


def forecast_case(case, name):
    """Forecast one case: 23 rows (hours 2..24) of five quantiles each."""
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    hours = [row["hour"] for row in case.get("forecast", [])]
    if sorted(hours) != list(range(2, 25)):
        raise ValueError(f"{name} does not ask for hours 2..24")
    target = {
        "day": parse_ts(case["day_start"]),
        "console_only": True,  # GTA VI is PS5 / Xbox Series X|S only, single-player
        "hour1": p1,
        "pre24": pre24_ratio(case.get("history_hourly", []), case["day_start"], p1),
    }
    pool = []
    for comp in case.get("comparables", []) or []:
        entry = extract_comparable(comp)
        if entry is not None:
            pool.append(entry)
    quantiles = forecast_hours(target, pool, hours)
    return {"hours": [{"hour": h, "q": quantiles[h]} for h in hours]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
