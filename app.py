"""
TITAN Multi-Flip Recovery Monitor
=================================
Dedicated Day-1 → Day-5 episode manager for BOTH-sides (multi-flip) days.

System B (Natural Method) rules only:
- FE direction retained (fade Asia first extreme)
- Cap flips: 3 normal / 2 congested
- Nearer-first level snaps
- EOD lock (flat)
- Congestion filter reduces aggression on chop
- No claim of universal recovery; goal = reduce exposure / break-even or small positive

Data: Twelve Data primary, optional Biquote fallback
Auto-refresh · Live prices · Levels rebuilt ~22:00 Spain each day

Decision-support only. No broker orders.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SPAIN = ZoneInfo("Europe/Madrid")
UTC = timezone.utc

TWELVE_BASE = "https://api.twelvedata.com"
BIQUOTE = "https://biquote.io/api"
NTFY_URL = os.environ.get("NTFY_URL", "https://ntfy.sh/TYf7NDB9moLR3kTq")

ENTRY_UTC = dtime(6, 0)
LOCK_UTC = dtime(14, 0)  # System B hard lock
LEVELS_BUILD_SPAIN = dtime(22, 0)  # 10pm Spain — rebuild levels for next session

PIP = 0.0001
FLIP_EUR = 20.0
FLIP_GBP = 25.0
MAX_FLIP_NORMAL = 3
MAX_FLIP_CONGESTED = 2
ROUND_STEP = 0.005
SWING_W = 4
NEAR_BAND_PIPS = 12.0  # snap band around grid trigger
TIMING_GATE_BARS = 8

REFRESH_SECONDS = 60
STATE_PATH = ROOT / "multiflip_episode_state.json"
LEVELS_CACHE = ROOT / "multiflip_levels_cache.json"

PAIRS = {
    "EURUSD": {
        "symbol": "EUR/USD",
        "pip": PIP,
        "flip": FLIP_EUR,
        "side_cost": 0.40,
        "median_atr": 70.0,  # daily-ish proxy in pips for congestion
    },
    "GBPUSD": {
        "symbol": "GBP/USD",
        "pip": PIP,
        "flip": FLIP_GBP,
        "side_cost": 0.50,
        "median_atr": 90.0,
    },
}

st.set_page_config(
    page_title="TITAN Multi-Flip Recovery",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# TIME HELPERS
# ---------------------------------------------------------------------------
def now_utc() -> datetime:
    return datetime.now(tz=UTC)


def now_spain() -> datetime:
    return datetime.now(tz=SPAIN)


def spain_hour() -> int:
    return now_spain().hour


def strategy_date_utc() -> date:
    """Trading day label in UTC (00:00–23:59 UTC)."""
    return now_utc().date()


# ---------------------------------------------------------------------------
# DATA FETCH
# ---------------------------------------------------------------------------
def get_api_key() -> str:
    key = st.session_state.get("td_key") or os.environ.get("TWELVEDATA_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("TWELVEDATA_API_KEY", "")  # type: ignore
        except Exception:
            key = ""
    return key or ""


def fetch_twelve_h1(symbol: str, outputsize: int = 200) -> pd.DataFrame:
    key = get_api_key()
    if not key:
        return pd.DataFrame()
    try:
        r = requests.get(
            f"{TWELVE_BASE}/time_series",
            params={
                "symbol": symbol,
                "interval": "1h",
                "outputsize": outputsize,
                "apikey": key,
                "timezone": "UTC",
            },
            timeout=20,
        )
        js = r.json()
        if "values" not in js:
            return pd.DataFrame()
        df = pd.DataFrame(js["values"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for c in ("open", "high", "low", "close", "volume"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("datetime")
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def fetch_twelve_price(symbol: str) -> Optional[float]:
    key = get_api_key()
    if not key:
        return None
    try:
        r = requests.get(
            f"{TWELVE_BASE}/price",
            params={"symbol": symbol, "apikey": key},
            timeout=12,
        )
        js = r.json()
        if "price" in js:
            return float(js["price"])
    except Exception:
        pass
    return None


def fetch_biquote_h1(pair: str, bars: int = 200) -> pd.DataFrame:
    """Optional fallback — path may vary by deployment."""
    try:
        # Common pattern used in prior TITAN apps
        url = f"{BIQUOTE}/ohlc"
        r = requests.get(
            url,
            params={"pair": pair.lower(), "tf": "H1", "limit": bars},
            timeout=15,
        )
        if r.status_code != 200:
            return pd.DataFrame()
        js = r.json()
        rows = js if isinstance(js, list) else js.get("data") or js.get("candles") or []
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # flexible column map
        colmap = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ("datetime", "time", "timestamp", "date"):
                colmap[c] = "datetime"
            elif cl in ("open", "o"):
                colmap[c] = "open"
            elif cl in ("high", "h"):
                colmap[c] = "high"
            elif cl in ("low", "l"):
                colmap[c] = "low"
            elif cl in ("close", "c"):
                colmap[c] = "close"
        df = df.rename(columns=colmap)
        if "datetime" not in df.columns:
            return pd.DataFrame()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        for c in ("open", "high", "low", "close"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).sort_values(
            "datetime"
        )
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def market_h1(pair: str) -> tuple[pd.DataFrame, str]:
    cfg = PAIRS[pair]
    df = fetch_twelve_h1(cfg["symbol"], 250)
    if not df.empty:
        return df, "Twelve Data"
    df = fetch_biquote_h1(pair, 250)
    if not df.empty:
        return df, "Biquote"
    return pd.DataFrame(), "none"


def live_price(pair: str) -> Optional[float]:
    cfg = PAIRS[pair]
    p = fetch_twelve_price(cfg["symbol"])
    if p is not None:
        return p
    df, _ = market_h1(pair)
    if not df.empty:
        return float(df.iloc[-1]["close"])
    return None


# ---------------------------------------------------------------------------
# STRUCTURE: ASIA FE, ATR, LEVELS
# ---------------------------------------------------------------------------
def asia_slice(df: pd.DataFrame, d: date) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df["datetime"].dt.date == d) & (df["datetime"].dt.hour < 6)
    return df.loc[mask].copy()


def post_asia_slice(df: pd.DataFrame, d: date) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df["datetime"].dt.date == d) & (df["datetime"].dt.hour >= 6)
    return df.loc[mask].copy()


def compute_asia_fe(df: pd.DataFrame, d: date) -> dict[str, Any]:
    """Causal Asia FE at/after 06:00 — no full-day high/low."""
    asia = asia_slice(df, d)
    out = {
        "ok": False,
        "ah": None,
        "al": None,
        "fe": None,  # "HIGH" or "LOW"
        "side": 0,  # -1 SELL, +1 BUY (fade)
        "ah_time": None,
        "al_time": None,
    }
    if asia.empty or len(asia) < 2:
        return out
    ah_idx = asia["high"].idxmax()
    al_idx = asia["low"].idxmin()
    ah = float(asia.loc[ah_idx, "high"])
    al = float(asia.loc[al_idx, "low"])
    ah_t = asia.loc[ah_idx, "datetime"]
    al_t = asia.loc[al_idx, "datetime"]
    if ah_t <= al_t:
        fe = "HIGH"
        side = -1  # fade high → sell
    else:
        fe = "LOW"
        side = +1  # fade low → buy
    out.update(
        ok=True, ah=ah, al=al, fe=fe, side=side, ah_time=ah_t, al_time=al_t
    )
    return out


def both_sides_broken(df: pd.DataFrame, d: date, ah: float, al: float) -> dict[str, Any]:
    post = post_asia_slice(df, d)
    out = {
        "both": False,
        "broke_ah": False,
        "broke_al": False,
        "first_break": None,  # "AH" or "AL"
    }
    if post.empty or ah is None or al is None:
        return out
    broke_ah = False
    broke_al = False
    first = None
    for _, row in post.iterrows():
        if not broke_ah and float(row["high"]) > ah:
            broke_ah = True
            if first is None:
                first = "AH"
        if not broke_al and float(row["low"]) < al:
            broke_al = True
            if first is None:
                first = "AL"
        if broke_ah and broke_al:
            break
    out["broke_ah"] = broke_ah
    out["broke_al"] = broke_al
    out["both"] = broke_ah and broke_al
    out["first_break"] = first
    return out


def atr_pips_h1(df: pd.DataFrame, n: int = 14) -> Optional[float]:
    if df is None or len(df) < n + 2:
        return None
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.tail(n).mean()
    if pd.isna(atr):
        return None
    return float(atr / PIP)


def daily_range_pips(df: pd.DataFrame, d: date) -> Optional[float]:
    day = df[df["datetime"].dt.date == d]
    if day.empty:
        return None
    return float((day["high"].max() - day["low"].min()) / PIP)


# ---------------------------------------------------------------------------
# RESEARCH LEVEL ENGINE (exact formulas from Confluence / Complete S&R reports)
# Original (non-Gann) confluence preferred — Gann did not raise high-conf success.
# Refined Level = Σ(Pi × Wi) / Σ Wi
# Cluster if Pi+1 − Pi ≤ 0.0013 (13 pips); keep clusters with Score = Σ Wi ≥ 5.5
# High-confluence operational threshold ≥ 8.0
# Nearer-first is applied ONLY to these refined levels (as tested).
# ---------------------------------------------------------------------------
CLUSTER_DIST = 0.0013  # 13 pips
MIN_CLUSTER_SCORE = 5.5
HIGH_CONF_SCORE = 8.0
EQUAL_TOL = 0.0008  # 8 pips
OB_BODY_PIPS = 14.0
VOL_WINDOW = 40
VOL_TOP_N = 6

# Weights from research (Original / Institutional blend — non-Gann core)
W_SWING = 3.0
W_ROUND = 2.0
W_VOLUME = 1.3
W_EQUAL = 3.5
W_ORDERBLOCK = 2.8


def causal_swings_indexed(
    df: pd.DataFrame, w: int = SWING_W
) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
    """Causal swings: level confirmed only at j+w. Returns (price, confirm_index)."""
    highs, lows = [], []
    if df is None or len(df) < 2 * w + 3:
        return highs, lows
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    n = len(df)
    for j in range(w, n - w):
        if h[j] == h[j - w : j + w + 1].max():
            highs.append((float(h[j]), j + w))
        if l[j] == l[j - w : j + w + 1].min():
            lows.append((float(l[j]), j + w))
    return highs[-12:], lows[-12:]


def model_swings(df: pd.DataFrame) -> list[tuple[float, float]]:
    """(price, weight) from latest swing highs/lows."""
    sh, sl = causal_swings_indexed(df, SWING_W)
    out: list[tuple[float, float]] = []
    for p, _ in sh[-4:]:
        out.append((p, W_SWING))
    for p, _ in sl[-4:]:
        out.append((p, W_SWING))
    return out


def model_rounds(price: float) -> list[tuple[float, float]]:
    """Rk = round(P/0.005)*0.005 + k*0.005, k=-5..+5."""
    base = round(price / ROUND_STEP) * ROUND_STEP
    return [(round(base + k * ROUND_STEP, 5), W_ROUND) for k in range(-5, 6)]


def model_volume_proxy(df: pd.DataFrame) -> list[tuple[float, float]]:
    """Highs+Lows of top-N volume bars in last 25–40 bars."""
    if df is None or len(df) < 10:
        return []
    tail = df.tail(VOL_WINDOW).copy()
    if "volume" not in tail.columns or tail["volume"].isna().all():
        # Fallback: use range as activity proxy when volume missing
        tail = tail.copy()
        tail["_act"] = (tail["high"].astype(float) - tail["low"].astype(float)).abs()
        top = tail.nlargest(VOL_TOP_N, "_act")
    else:
        tail["volume"] = pd.to_numeric(tail["volume"], errors="coerce").fillna(0)
        top = tail.nlargest(VOL_TOP_N, "volume")
    out: list[tuple[float, float]] = []
    for _, row in top.iterrows():
        out.append((float(row["high"]), W_VOLUME))
        out.append((float(row["low"]), W_VOLUME))
    return out


def model_equal_pools(df: pd.DataFrame) -> list[tuple[float, float]]:
    """Equal highs/lows: cluster swings within 8 pips; level = mean; min 2 members."""
    sh, sl = causal_swings_indexed(df, SWING_W)
    out: list[tuple[float, float]] = []

    def pools(levels: list[float]) -> list[float]:
        if len(levels) < 2:
            return []
        s = sorted(levels)
        groups: list[list[float]] = [[s[0]]]
        for x in s[1:]:
            if x - groups[-1][-1] <= EQUAL_TOL:
                groups[-1].append(x)
            else:
                groups.append([x])
        return [sum(g) / len(g) for g in groups if len(g) >= 2]

    out.extend((p, W_EQUAL) for p in pools([x for x, _ in sh]))
    out.extend((p, W_EQUAL) for p in pools([x for x, _ in sl]))
    return out


def model_order_blocks(df: pd.DataFrame) -> list[tuple[float, float]]:
    """
    Bullish OB: last down-close before impulsive up (body > 14 pips).
    Bearish OB: last up-close before impulsive down.
    Level = (H+L)/2 of OB candle.
    """
    if df is None or len(df) < 8:
        return []
    o = df["open"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    out: list[tuple[float, float]] = []
    n = len(df)
    body_min = OB_BODY_PIPS * PIP
    # scan recent history only
    start = max(2, n - 80)
    for i in range(start, n - 1):
        body = abs(c[i] - o[i])
        if body < body_min:
            continue
        # impulsive up after down-close
        if c[i] < o[i] and c[i + 1] > o[i + 1] and (c[i + 1] - o[i + 1]) >= body_min * 0.8:
            mid = (h[i] + l[i]) / 2.0
            out.append((float(mid), W_ORDERBLOCK))
        # impulsive down after up-close
        if c[i] > o[i] and c[i + 1] < o[i + 1] and (o[i + 1] - c[i + 1]) >= body_min * 0.8:
            mid = (h[i] + l[i]) / 2.0
            out.append((float(mid), W_ORDERBLOCK))
    return out[-8:]


def cluster_weighted(
    candidates: list[tuple[float, float]],
    dist: float = CLUSTER_DIST,
    min_score: float = MIN_CLUSTER_SCORE,
) -> list[dict[str, float]]:
    """
    Sort by price. Group while Pi+1 - Pi ≤ dist.
    Refined = Σ(Pi×Wi)/ΣWi ; Score = ΣWi ; keep Score ≥ min_score.
    """
    if not candidates:
        return []
    # merge duplicate prices by summing weights
    bucket: dict[float, float] = {}
    for p, w in candidates:
        key = round(float(p), 5)
        bucket[key] = bucket.get(key, 0.0) + float(w)
    items = sorted(bucket.items(), key=lambda x: x[0])  # (price, weight)

    clusters: list[list[tuple[float, float]]] = [[items[0]]]
    for p, w in items[1:]:
        if p - clusters[-1][-1][0] <= dist:
            clusters[-1].append((p, w))
        else:
            clusters.append([(p, w)])

    refined: list[dict[str, float]] = []
    for cl in clusters:
        sw = sum(w for _, w in cl)
        if sw < min_score:
            continue
        centre = sum(p * w for p, w in cl) / sw
        refined.append(
            {
                "level": float(centre),
                "score": float(sw),
                "width": float(cl[-1][0] - cl[0][0]),
                "n": float(len(cl)),
            }
        )
    return refined


def build_levels(df: pd.DataFrame, price: float) -> dict[str, Any]:
    """
    Research confluence levels (Original non-Gann stack) + nearer-first.
    This is the same mathematical path the sequence tests used:
      candidates → 13-pip clusters → weighted centre → R/S → nearer.
    """
    price = float(price)
    candidates: list[tuple[float, float]] = []
    candidates.extend(model_swings(df))
    candidates.extend(model_rounds(price))
    candidates.extend(model_volume_proxy(df))
    candidates.extend(model_equal_pools(df))
    candidates.extend(model_order_blocks(df))

    all_clusters = cluster_weighted(candidates, CLUSTER_DIST, MIN_CLUSTER_SCORE)
    high_conf = [c for c in all_clusters if c["score"] >= HIGH_CONF_SCORE]

    buffer = 3 * PIP
    above = [c for c in all_clusters if c["level"] > price + buffer]
    below = [c for c in all_clusters if c["level"] < price - buffer]
    # Prefer higher score when picking nearest; primary key remains distance
    above.sort(key=lambda c: (c["level"] - price, -c["score"]))
    below.sort(key=lambda c: (price - c["level"], -c["score"]))

    R = above[0]["level"] if above else None
    R_score = above[0]["score"] if above else None
    S = below[0]["level"] if below else None
    S_score = below[0]["score"] if below else None

    nearer = nearer_side = None
    nearer_score = None
    if R is not None and S is not None:
        if abs(R - price) <= abs(price - S):
            nearer, nearer_side, nearer_score = R, "R", R_score
        else:
            nearer, nearer_side, nearer_score = S, "S", S_score
    elif R is not None:
        nearer, nearer_side, nearer_score = R, "R", R_score
    elif S is not None:
        nearer, nearer_side, nearer_score = S, "S", S_score

    return {
        "price": price,
        "R": R,
        "S": S,
        "R_score": R_score,
        "S_score": S_score,
        "nearer": nearer,
        "nearer_side": nearer_side,
        "nearer_score": nearer_score,
        "invalidation_above": R,
        "invalidation_below": S,
        "clusters": all_clusters,
        "high_conf_clusters": high_conf,
        "n_candidates": len(candidates),
        "method": "research_confluence_v1",
        "formula": "Refined=Σ(Pi·Wi)/ΣWi | cluster≤13pips | min_score≥5.5 | high_conf≥8",
        "weights": {
            "swing": W_SWING,
            "round": W_ROUND,
            "volume": W_VOLUME,
            "equal": W_EQUAL,
            "order_block": W_ORDERBLOCK,
        },
        "built_at": now_utc().isoformat(),
    }


def congestion_flag(day_range_pips: Optional[float], atr_p: Optional[float], pair: str) -> bool:
    """True = congested / choppy → tighter flip cap."""
    med = PAIRS[pair]["median_atr"]
    if day_range_pips is not None and day_range_pips < 0.55 * med:
        return True
    if atr_p is not None and atr_p < 0.70 * (med / 5):  # rough H1 scale
        return True
    return False


# ---------------------------------------------------------------------------
# MON / TUE AGREEMENT
# ---------------------------------------------------------------------------
def fe_for_date(df: pd.DataFrame, d: date) -> Optional[str]:
    fe = compute_asia_fe(df, d)
    return fe["fe"] if fe["ok"] else None


def mon_tue_lean(df: pd.DataFrame, ref: date) -> dict[str, Any]:
    """Rest-of-week lean from Mon+Tue FE agreement (System B context)."""
    # Find Monday of current ISO week
    monday = ref - timedelta(days=ref.weekday())
    tuesday = monday + timedelta(days=1)
    mon_fe = fe_for_date(df, monday) if ref >= monday else None
    tue_fe = fe_for_date(df, tuesday) if ref >= tuesday else None
    agree = mon_fe is not None and tue_fe is not None and mon_fe == tue_fe
    disagree = mon_fe is not None and tue_fe is not None and mon_fe != tue_fe
    lean = None
    if agree and mon_fe == "HIGH":
        lean = "SELL lean rest of week (Mon+Tue FE HIGH agree)"
    elif agree and mon_fe == "LOW":
        lean = "BUY lean rest of week (Mon+Tue FE LOW agree)"
    elif disagree:
        lean = "No lean — Mon/Tue disagree (coin-flip posture)"
    else:
        lean = "Incomplete — waiting Mon/Tue FE"
    return {
        "monday": str(monday),
        "tuesday": str(tuesday),
        "mon_fe": mon_fe,
        "tue_fe": tue_fe,
        "agree": agree,
        "disagree": disagree,
        "lean": lean,
    }


# ---------------------------------------------------------------------------
# EPISODE STATE
# ---------------------------------------------------------------------------
@dataclass
class PairEpisode:
    pair: str
    active: bool = False
    episode_day: int = 0  # 0=idle, 1..5
    start_date: Optional[str] = None
    fe_side: int = 0  # +1 buy / -1 sell fade
    fe_label: str = ""
    entry: Optional[float] = None
    flips: int = 0
    max_flips: int = MAX_FLIP_NORMAL
    basket_pips: float = 0.0
    locked_today: float = 0.0
    last_action: str = "IDLE"
    notes: str = ""
    ah: Optional[float] = None
    al: Optional[float] = None
    congested: bool = False


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"EURUSD": asdict(PairEpisode("EURUSD")), "GBPUSD": asdict(PairEpisode("GBPUSD"))}


def save_state(state: dict[str, Any]) -> None:
    try:
        STATE_PATH.write_text(json.dumps(state, indent=2, default=str))
    except Exception:
        pass


def load_levels_cache() -> dict[str, Any]:
    if LEVELS_CACHE.exists():
        try:
            return json.loads(LEVELS_CACHE.read_text())
        except Exception:
            pass
    return {}


def save_levels_cache(cache: dict[str, Any]) -> None:
    try:
        LEVELS_CACHE.write_text(json.dumps(cache, indent=2, default=str))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# OPERATOR GUIDANCE (DAY 1–5)
# ---------------------------------------------------------------------------
def guidance_for_day(
    day_n: int,
    pair: str,
    fe_label: str,
    side: int,
    levels: dict,
    both: dict,
    congested: bool,
    basket: float,
    flips: int,
    max_flips: int,
    atr_p: Optional[float],
    lean: dict,
    price: Optional[float],
) -> dict[str, Any]:
    """What the operator should do on this episode day."""
    dir_txt = "SELL (fade Asia HIGH)" if side < 0 else "BUY (fade Asia LOW)" if side > 0 else "—"
    nearer = levels.get("nearer")
    R, S = levels.get("R"), levels.get("S")
    flip_dist = PAIRS[pair]["flip"]

    traffic = "YELLOW"  # careful default on multi
    if day_n == 0:
        traffic = "GREY"
    elif basket >= 0 and day_n >= 1:
        traffic = "GREEN"
    elif basket < -40:
        traffic = "RED"
    elif congested:
        traffic = "YELLOW"

    actions = []
    if day_n == 0:
        actions = [
            "Not in multi-flip episode.",
            "Monitor Asia FE and whether BOTH AH and AL break after 06:00 UTC.",
            "Only open this panel’s episode when BOTH sides are broken (true multi-flip day).",
        ]
    elif day_n == 1:
        actions = [
            f"DAY 1 — Multi-flip confirmed. Direction stays {dir_txt}.",
            f"Flip distance {flip_dist:.0f} pips. Cap = {max_flips} ({'CONGESTED' if congested else 'normal'}).",
            "Use nearer-first: prefer snap to nearest R/S inside the adverse band — do not invent new direction.",
            "Reduce size vs clean FE days (0.5× suggested).",
            f"Lock any positive leg by {LOCK_UTC.strftime('%H:%M')} UTC. Flat — protect pips.",
            "If still negative after lock, carry bag conceptually to Day 2 (manual) — no auto recovery assumed.",
        ]
    elif day_n == 2:
        actions = [
            "DAY 2 — Continue reduction only if Day 1 locked residual still negative.",
            "Re-read TODAY’s Asia FE (new day). Prefer alignment with episode FE if possible; do not force.",
            "Same nearer-first + tight cap. Do not increase size.",
            "Lock positives at 14:00 UTC. Goal: shrink residual, not hero recovery.",
        ]
    elif day_n == 3:
        actions = [
            "DAY 3 — Only if residual still open and structure still multi/choppy.",
            "If today is one-sided FE_HOLD quality, do NOT force multi rules on a clean day — step aside or tiny size.",
            "Hard discipline: max flips still capped. Prefer break-even close of episode if residual small.",
        ]
    elif day_n == 4:
        actions = [
            "DAY 4 — Fatigue zone. Prefer neutralise residual rather than expand risk.",
            "If residual > hard comfort stop (e.g. −80/−100 p episode), accept residual and END episode.",
            "Nearer levels still guide triggers; FE is still number one for direction on any NEW trade.",
        ]
    else:  # 5+
        actions = [
            "DAY 5 — Final episode day under Natural Method research track.",
            "Close residual at lock. Do not extend past Day 5 as a system rule.",
            "Record outcome. Next multi day starts a NEW episode at Day 1.",
        ]

    if both.get("both"):
        status_line = "BOTH Asia sides broken — multi-flip path ACTIVE"
    elif both.get("broke_ah") or both.get("broke_al"):
        status_line = "One Asia side broken — not yet multi (stay FE_HOLD on main system)"
    else:
        status_line = "Asia extremes still intact — not a multi-flip day"

    return {
        "traffic": traffic,
        "direction": dir_txt,
        "status_line": status_line,
        "actions": actions,
        "nearer": nearer,
        "R": R,
        "S": S,
        "invalidation_above": levels.get("invalidation_above"),
        "invalidation_below": levels.get("invalidation_below"),
        "flip_dist": flip_dist,
        "max_flips": max_flips,
        "flips": flips,
        "basket": basket,
        "congested": congested,
        "atr": atr_p,
        "lean": lean.get("lean"),
        "price": price,
        "fe_label": fe_label,
    }


# ---------------------------------------------------------------------------
# UI HELPERS
# ---------------------------------------------------------------------------
def traffic_html(color: str, label: str) -> str:
    colors = {
        "GREEN": "#00d68f",
        "RED": "#ff4d6d",
        "YELLOW": "#ffb703",
        "GREY": "#64748b",
        "BLUE": "#4cc9f0",
    }
    c = colors.get(color, "#64748b")
    return f"""
    <div style="text-align:center;padding:18px;border-radius:12px;border:2px solid {c};
    background:linear-gradient(180deg,rgba(0,0,0,0.25),transparent);">
      <div style="width:28px;height:28px;border-radius:50%;background:{c};margin:0 auto 8px;
      box-shadow:0 0 18px {c};"></div>
      <div style="font-weight:700;letter-spacing:0.5px;color:{c};">{label}</div>
    </div>
    """


def metric_row(cols, items):
    for c, (label, value) in zip(cols, items):
        c.metric(label, value)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    st.title("🛡️ TITAN Multi-Flip Recovery")
    st.caption(
        "Dedicated Day 1–5 episode desk · System B Natural Method · "
        "Use only when BOTH Asia extremes are broken · Decision-support only"
    )

    with st.sidebar:
        st.header("Settings")
        td_key = st.text_input(
            "Twelve Data API key",
            value=get_api_key(),
            type="password",
            help="Required for live H1 + price",
        )
        if td_key:
            st.session_state["td_key"] = td_key
        auto = st.checkbox("Auto-refresh", value=True)
        st.caption(f"Refresh every {REFRESH_SECONDS}s")
        st.divider()
        st.markdown("**Lock time (UTC)**")
        st.code(str(LOCK_UTC))
        st.markdown("**Levels rebuild (Spain)**")
        st.code("22:00")
        st.divider()
        if st.button("Rebuild levels now"):
            st.session_state["force_levels"] = True
        if st.button("Reset all episodes"):
            save_state(
                {
                    "EURUSD": asdict(PairEpisode("EURUSD")),
                    "GBPUSD": asdict(PairEpisode("GBPUSD")),
                }
            )
            st.success("Episodes cleared")
            st.rerun()

    state = load_state()
    levels_cache = load_levels_cache()
    today = strategy_date_utc()
    sp = now_spain()

    st.info(
        f"UTC {now_utc().strftime('%Y-%m-%d %H:%M')} · Spain {sp.strftime('%Y-%m-%d %H:%M')} · "
        f"Strategy date {today.isoformat()}"
    )

    # ------------------------------------------------------------------
    # PER PAIR ANALYSIS
    # ------------------------------------------------------------------
    tabs = st.tabs(["EURUSD", "GBPUSD", "Combined desk", "Rules (System B)"])

    analyses = {}

    for pair, tab in zip(("EURUSD", "GBPUSD"), tabs[:2]):
        with tab:
            df, src = market_h1(pair)
            px = live_price(pair)
            if df.empty:
                st.error(f"No H1 data for {pair} (source={src}). Check API key.")
                continue

            fe = compute_asia_fe(df, today)
            both = (
                both_sides_broken(df, today, fe["ah"], fe["al"])
                if fe["ok"]
                else {"both": False, "broke_ah": False, "broke_al": False, "first_break": None}
            )
            atr_p = atr_pips_h1(df, 14)
            day_rng = daily_range_pips(df, today)
            congested = congestion_flag(day_rng, atr_p, pair)
            lean = mon_tue_lean(df, today)

            # Levels: rebuild if forced, missing, or after 22:00 Spain on new stamp
            need_levels = (
                st.session_state.get("force_levels")
                or pair not in levels_cache
                or levels_cache.get(pair, {}).get("for_date") != str(today)
            )
            # Also rebuild after 22:00 Spain for next day prep (stamp with tomorrow if past 22:00)
            if sp.time() >= LEVELS_BUILD_SPAIN and levels_cache.get(pair, {}).get("evening_stamp") != str(today):
                need_levels = True

            if need_levels and px is not None:
                lv = build_levels(df, px)
                lv["for_date"] = str(today)
                lv["evening_stamp"] = str(today)
                levels_cache[pair] = lv
                save_levels_cache(levels_cache)
            levels = levels_cache.get(pair) or build_levels(df, px or float(df.iloc[-1]["close"]))

            ep = state.get(pair) or asdict(PairEpisode(pair))
            # Auto-suggest start episode if BOTH and not active
            if both.get("both") and not ep.get("active"):
                st.warning(
                    "Multi-flip condition detected (BOTH AH & AL broken). "
                    "Operator can START episode Day 1 below."
                )

            max_flips = MAX_FLIP_CONGESTED if congested or ep.get("congested") else MAX_FLIP_NORMAL
            day_n = int(ep.get("episode_day") or 0)
            side = int(ep.get("fe_side") or fe.get("side") or 0)
            fe_label = ep.get("fe_label") or fe.get("fe") or "—"

            guide = guidance_for_day(
                day_n=day_n if ep.get("active") else (1 if both.get("both") else 0),
                pair=pair,
                fe_label=fe_label,
                side=side if ep.get("active") else int(fe.get("side") or 0),
                levels=levels,
                both=both,
                congested=congested,
                basket=float(ep.get("basket_pips") or 0),
                flips=int(ep.get("flips") or 0),
                max_flips=max_flips,
                atr_p=atr_p,
                lean=lean,
                price=px,
            )

            analyses[pair] = {
                "fe": fe,
                "both": both,
                "atr": atr_p,
                "day_rng": day_rng,
                "congested": congested,
                "lean": lean,
                "levels": levels,
                "guide": guide,
                "src": src,
                "px": px,
                "ep": ep,
            }

            # Header metrics
            c1, c2, c3, c4, c5 = st.columns(5)
            metric_row(
                [c1, c2, c3, c4, c5],
                [
                    ("Price", f"{px:.5f}" if px else "—"),
                    ("Asia FE", fe.get("fe") or "—"),
                    ("BOTH broken", "YES" if both.get("both") else "NO"),
                    ("ATR H1 (p)", f"{atr_p:.1f}" if atr_p else "—"),
                    ("Source", src),
                ],
            )

            # Traffic + status
            left, right = st.columns([1, 2])
            with left:
                label = {
                    "GREEN": "PROTECT / OK",
                    "RED": "DAMAGE CONTROL",
                    "YELLOW": "CAREFUL MULTI",
                    "GREY": "IDLE",
                }.get(guide["traffic"], guide["traffic"])
                st.markdown(traffic_html(guide["traffic"], label), unsafe_allow_html=True)
                st.caption(f"Episode Day {day_n if ep.get('active') else 0} · Basket {float(ep.get('basket_pips') or 0):+.1f} p")
            with right:
                st.subheader("Operator status")
                st.write(guide["status_line"])
                st.write(f"**Direction:** {guide['direction']}")
                st.write(f"**Mon/Tue lean:** {guide['lean']}")
                st.write(
                    f"**Congestion:** {'YES — cap '+str(MAX_FLIP_CONGESTED) if congested else 'No — cap '+str(MAX_FLIP_NORMAL)}"
                )

            st.subheader("Levels (combined · nearer-first)")
            lc1, lc2, lc3, lc4 = st.columns(4)
            lc1.metric("Nearest R", f"{levels.get('R'):.5f}" if levels.get("R") else "—")
            lc2.metric("Nearest S", f"{levels.get('S'):.5f}" if levels.get("S") else "—")
            lc3.metric(
                "Nearer first",
                f"{levels.get('nearer'):.5f} ({levels.get('nearer_side')})"
                if levels.get("nearer")
                else "—",
            )
            inv = levels.get("invalidation_above") or levels.get("invalidation_below")
            lc4.metric(
                "Invalidate thru",
                f"{levels.get('invalidation_above'):.5f} / {levels.get('invalidation_below'):.5f}"
                if levels.get("invalidation_above") and levels.get("invalidation_below")
                else "—",
            )
            st.caption(
                "Research confluence: swings·rounds·volume proxy·equal pools·order blocks → "
                "cluster ≤13 pips → Refined=Σ(Pi·Wi)/ΣWi (min score 5.5). "
                "Nearer-first applied to these refined levels only (≈61–64% as tested). "
                "Does not change FE direction."
            )
            if levels.get("R_score") is not None or levels.get("S_score") is not None:
                st.caption(
                    f"Scores — R: {levels.get('R_score')} · S: {levels.get('S_score')} · "
                    f"Nearer: {levels.get('nearer_score')} · "
                    f"High-conf clusters (≥8): {len(levels.get('high_conf_clusters') or [])} · "
                    f"Candidates: {levels.get('n_candidates')}"
                )

            st.subheader(f"Day {max(day_n, 1 if both.get('both') else 0)} — What to do")
            for a in guide["actions"]:
                st.markdown(f"- {a}")

            # Asia structure
            with st.expander("Asia structure detail", expanded=False):
                st.write(
                    {
                        "AH": fe.get("ah"),
                        "AL": fe.get("al"),
                        "FE": fe.get("fe"),
                        "broke_AH": both.get("broke_ah"),
                        "broke_AL": both.get("broke_al"),
                        "first_break": both.get("first_break"),
                        "day_range_pips": day_rng,
                    }
                )

            # Episode controls
            st.subheader("Episode control")
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                if st.button(f"START Day 1 — {pair}", disabled=ep.get("active")):
                    if not fe.get("ok"):
                        st.error("Need valid Asia FE")
                    else:
                        ep = asdict(
                            PairEpisode(
                                pair=pair,
                                active=True,
                                episode_day=1,
                                start_date=str(today),
                                fe_side=int(fe["side"]),
                                fe_label=str(fe["fe"]),
                                entry=px,
                                flips=0,
                                max_flips=max_flips,
                                basket_pips=0.0,
                                locked_today=0.0,
                                last_action="START_D1",
                                notes="Multi-flip episode opened",
                                ah=fe.get("ah"),
                                al=fe.get("al"),
                                congested=congested,
                            )
                        )
                        state[pair] = ep
                        save_state(state)
                        st.success("Episode Day 1 started")
                        st.rerun()
            with e2:
                if st.button(f"Advance day — {pair}", disabled=not ep.get("active")):
                    dcur = int(ep.get("episode_day") or 1)
                    if dcur >= 5:
                        ep["active"] = False
                        ep["last_action"] = "END_D5"
                        ep["notes"] = "Episode closed at Day 5"
                    else:
                        ep["episode_day"] = dcur + 1
                        ep["flips"] = 0
                        ep["locked_today"] = 0.0
                        ep["last_action"] = f"ADV_D{dcur + 1}"
                    state[pair] = ep
                    save_state(state)
                    st.rerun()
            with e3:
                lock_p = st.number_input(
                    f"Lock pips today ({pair})",
                    value=float(ep.get("locked_today") or 0.0),
                    step=1.0,
                    key=f"lock_{pair}",
                )
                if st.button(f"Apply lock — {pair}", disabled=not ep.get("active")):
                    ep["locked_today"] = float(lock_p)
                    ep["basket_pips"] = float(ep.get("basket_pips") or 0) + float(lock_p)
                    ep["last_action"] = "LOCK"
                    # If basket non-negative → end episode
                    if ep["basket_pips"] >= 0:
                        ep["active"] = False
                        ep["episode_day"] = 0
                        ep["notes"] = "Basket ≥ 0 — episode closed (Natural Method)"
                    state[pair] = ep
                    save_state(state)
                    st.success(f"Basket now {ep['basket_pips']:+.1f} p")
                    st.rerun()
            with e4:
                if st.button(f"END episode — {pair}", disabled=not ep.get("active")):
                    ep["active"] = False
                    ep["episode_day"] = 0
                    ep["last_action"] = "MANUAL_END"
                    state[pair] = ep
                    save_state(state)
                    st.rerun()

            st.write(
                f"Active={ep.get('active')} · Day={ep.get('episode_day')} · "
                f"FE={ep.get('fe_label')} · Flips={ep.get('flips')}/{ep.get('max_flips')} · "
                f"Basket={float(ep.get('basket_pips') or 0):+.1f} p · Last={ep.get('last_action')}"
            )

    st.session_state["force_levels"] = False

    # ------------------------------------------------------------------
    # COMBINED DESK
    # ------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Combined recovery desk")
        st.markdown(
            "Use this page only when at least one pair is in a multi-flip episode. "
            "Main FE website remains for clean one-sided days."
        )
        for pair in ("EURUSD", "GBPUSD"):
            a = analyses.get(pair)
            if not a:
                continue
            g = a["guide"]
            ep = a["ep"]
            st.markdown(f"### {pair}")
            cols = st.columns(4)
            cols[0].markdown(
                traffic_html(g["traffic"], g["traffic"]), unsafe_allow_html=True
            )
            cols[1].write(f"**FE** {a['fe'].get('fe')} · BOTH `{a['both'].get('both')}`")
            cols[2].write(f"**Nearer** {g.get('nearer')}")
            cols[3].write(
                f"**Episode** Day {ep.get('episode_day', 0)} · Basket {float(ep.get('basket_pips') or 0):+.1f}p"
            )
            st.caption(g["status_line"])

    # ------------------------------------------------------------------
    # RULES
    # ------------------------------------------------------------------
    with tabs[3]:
        st.subheader("System B — Multi-flip recovery rules (this site)")
        st.markdown(
            """
**When to use this website**  
Only when **both** Asia high and Asia low are broken after 06:00 UTC (true multi-flip / BOTH day).  
Clean one-sided FE days stay on the main monitor — do not force recovery rules there.

**Priority**  
1. Asia FE direction (fade) — never inverted by levels  
2. Confirm BOTH break  
3. Congestion → max **2** flips; else max **3**  
4. Nearer-first R/S inside adverse band  
5. Lock at **14:00 UTC** — flat; bank positive; residual may continue Day 2–5 manually  
6. Day 5 hard end of episode  

**Levels**  
Causal swing confirm W=4 + 0.005 round numbers → nearest R above, S below.  
Nearer of R/S is preferred first interaction (~60–65% historical).  
Invalidation: decisive trade through R or S against the working leg.

**What this site does *not* claim**  
- Automatic recovery every time  
- All weeks/months/years positive  
- Look-ahead full-day FE  

**Goal**  
Reduce damage on multi-flip days and exit episodes at break-even or small positive when structure allows.
"""
        )
        st.markdown("**Flip distances**")
        st.write({"EURUSD": f"{FLIP_EUR} pips", "GBPUSD": f"{FLIP_GBP} pips"})
        st.markdown("**Mon+Tue lean** is context only — does not rewrite FE on the multi day.")

    # ------------------------------------------------------------------
    # AUTO REFRESH
    # ------------------------------------------------------------------
    if auto:
        time.sleep(REFRESH_SECONDS)
        st.rerun()


if __name__ == "__main__":
    main()
