"""
Gold (XAUUSD) Macro Intelligence Dashboard  -  v4  (spot-only)
----------------------------------------------------------------
Run with:  streamlit run app_v4.py --theme.base dark

Requires: streamlit>=1.37, yfinance, pandas, numpy, plyer   (requests comes with yfinance)

What's new in v4:
  * Gold price is XAU/USD SPOT from gold-api.com. Gold futures (GC=F) are gone.
  * gold-api.com only returns the latest price, so a background thread polls it
    and builds 5-minute OHLC bars itself (saved to spot_bars_5m.csv so history
    survives restarts). Bars keep building even if the browser tab is closed.
  * Optional one-time history pre-fill from Yahoo's XAUUSD=X (spot) so EMAs/ATR
    work immediately instead of after several hours of warm-up.
  * Macro drivers (DXY, US10Y, VIX, S&P futures) still come from Yahoo Finance.

Educational tool only - not financial advice.
"""

import json
import platform
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
ET = ZoneInfo("America/New_York")
UTC = timezone.utc

GOLD_API_URL = "https://api.gold-api.com/price/XAU"
GOLD_LABEL = "Gold Spot (XAU/USD)"
SRC_NAME = "gold-api.com"
YAHOO_SPOT_SEED_SYMBOL = "XAUUSD=X"   # spot pair, used ONLY to pre-fill history
SEED_IF_BARS_BELOW = 300              # skip pre-fill if we already have ~25h of bars
KEEP_DAYS = 7

# key -> (symbol, display name, decimals)
ASSETS = {
    "DXY": ("DX-Y.NYB", "US Dollar Index", 3),
    "TNX": ("^TNX", "10Y Treasury Yield", 3),
    "VIX": ("^VIX", "VIX Volatility", 2),
    "ES": ("ES=F", "S&P 500 Futures", 2),
}

TREND_OFF = "Off"
TREND_PRICE = "Price vs fast EMA"
TREND_FULL = "Full stack (price > fast > slow)"
VIX_LEVEL = "Level (>hi risk-off, <lo complacent)"
VIX_DIR = "Direction (rising = +1)"

DEFAULTS = {
    "preset": "Balanced",
    # data & timing
    "refresh_s": 300, "poll_s": 15, "yahoo_seed": True, "stale_m": 30, "gold_age": 10,
    # macro scoring
    "lookback_min": 15, "deadzone": 0.0,
    "w_dxy": 1.0, "w_tnx": 1.0, "w_vix": 1.0, "w_es": 1.0,
    "vix_mode": VIX_LEVEL, "vix_hi": 20.0, "vix_lo": 15.0,
    "min_conv": 50, "min_live": 3,
    # gold filters
    "trend_mode": TREND_PRICE, "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "htf_filter": False, "min_atr": 0.0,
    # signal rules & risk
    "direction": "Both", "confirm_n": 2, "cooldown_m": 30, "max_open": 1,
    "atr_mult": 1.5, "rr": 2.0, "expiry_h": 4,
    # session & news
    "session_filter": True, "sess_start": 3, "sess_end": 16,
    "blackout_before": 30, "blackout_after": 60, "events_text": "",
    # alerts
    "notify_on": True,
}

OPTIONS = {
    "preset": ["Custom", "Conservative", "Balanced", "Aggressive"],
    "lookback_min": [5, 10, 15, 30, 60],
    "vix_mode": [VIX_LEVEL, VIX_DIR],
    "trend_mode": [TREND_OFF, TREND_PRICE, TREND_FULL],
    "direction": ["Both", "Long only", "Short only"],
}

PRESETS = {
    "Custom": {},
    "Conservative": dict(lookback_min=15, min_conv=75, confirm_n=2, trend_mode=TREND_FULL,
                         htf_filter=True, cooldown_m=45, session_filter=True, min_live=3),
    "Balanced": dict(lookback_min=15, min_conv=50, confirm_n=2, trend_mode=TREND_PRICE,
                     htf_filter=False, cooldown_m=30, session_filter=True, min_live=3),
    "Aggressive": dict(lookback_min=5, min_conv=50, confirm_n=1, trend_mode=TREND_OFF,
                       htf_filter=False, cooldown_m=15, session_filter=False, min_live=2),
}

# Settings that don't affect scoring/signals (changing them won't reset confirmations)
SIG_EXCLUDE = {"preset", "notify_on", "refresh_s", "poll_s", "yahoo_seed"}

SETTINGS_PATH = Path(__file__).with_name("settings_v4.json")
LOG_PATH = Path(__file__).with_name("signals_log_v4.csv")
BARS_PATH = Path(__file__).with_name("spot_bars_5m.csv")
LOG_COLS = [
    "id", "time_utc", "side", "entry", "stop", "target", "net_score", "atr",
    "entry_bar_utc", "status", "exit_price", "r_result", "closed_utc",
    "px_15m", "px_30m", "px_60m", "drivers", "src",
]
FLOAT_COLS = ["entry", "stop", "target", "net_score", "atr", "exit_price",
              "r_result", "px_15m", "px_30m", "px_60m"]
OBJ_COLS = ["time_utc", "side", "entry_bar_utc", "status", "closed_utc", "drivers", "src"]

st.set_page_config(page_title="Gold Spot Macro Signals", page_icon="🥇", layout="wide")
st.markdown(
    """
    <style>
        .stApp { background-color: #0e1117; color: #e6e6e6; }
        div[data-testid="stMetric"] {
            background: #161b22; border: 1px solid #30363d;
            border-radius: 12px; padding: 16px 18px;
        }
        div[data-testid="stMetricValue"] { font-size: 2.0rem; }
        .status-bar {
            background: #161b22; border: 1px solid #30363d; border-radius: 10px;
            padding: 10px 16px; font-size: 0.95rem; color: #c9d1d9;
        }
        h1, h2, h3 { color: #f5c542; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Spot recorder: background thread polling gold-api.com and building 5m bars
# ----------------------------------------------------------------------------
def _fetch_quote() -> dict:
    headers = {"User-Agent": "gold-macro-dashboard/1.0"}
    try:
        import requests
    except ImportError:
        req = urllib.request.Request(GOLD_API_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode())
    r = requests.get(GOLD_API_URL, headers=headers, timeout=8)
    r.raise_for_status()
    return r.json()


class SpotRecorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.bars = {}                 # bar start (UTC Timestamp) -> [o, h, l, c]
        self.last_price = None
        self.last_tick_ts = None       # provider's updatedAt of the last accepted tick
        self.last_poll_ts = None
        self.last_error = None
        self.poll_s = 15
        self.seeded = False
        self.first_tick = threading.Event()
        self._dirty = False
        self._last_save = 0.0
        self._load()
        threading.Thread(target=self._run, daemon=True, name="gold-api-poller").start()

    # -- persistence
    def _load(self):
        try:
            if BARS_PATH.exists():
                df = pd.read_csv(BARS_PATH)
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=KEEP_DAYS)
                for r in df.itertuples(index=False):
                    t = pd.Timestamp(r.time_utc)
                    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
                    if t >= cutoff:
                        self.bars[t] = [float(r.open), float(r.high), float(r.low), float(r.close)]
        except Exception as e:
            self.last_error = f"Could not load saved bars: {e}"

    def _maybe_save(self, force=False):
        if not self._dirty or (not force and time.time() - self._last_save < 60):
            return
        try:
            with self.lock:
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=KEEP_DAYS)
                self.bars = {t: v for t, v in self.bars.items() if t >= cutoff}
                rows = [(t.isoformat(), *v) for t, v in sorted(self.bars.items())]
            pd.DataFrame(rows, columns=["time_utc", "open", "high", "low", "close"]).to_csv(
                BARS_PATH, index=False)
            self._dirty = False
            self._last_save = time.time()
        except Exception:
            pass

    # -- polling
    def _poll_once(self):
        js = _fetch_quote()
        price = float(js["price"])
        if not price > 0:
            raise ValueError("invalid price in response")
        now = pd.Timestamp.now(tz="UTC")
        raw_ts = js.get("updatedAt") or js.get("updated_at")
        try:
            ts = pd.Timestamp(raw_ts)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        except Exception:
            ts = now
        if ts > now + pd.Timedelta(minutes=2):      # guard against clock skew
            ts = now

        if self.last_price and abs(price / self.last_price - 1) > 0.05:
            raise ValueError(f"ignored outlier price {price}")

        with self.lock:
            self.last_poll_ts = now
            self.last_error = None
            if self.last_tick_ts is None or ts > self.last_tick_ts:   # only genuinely new ticks
                start = ts.floor("5min")
                bar = self.bars.get(start)
                if bar is None:
                    self.bars[start] = [price, price, price, price]
                else:
                    bar[1], bar[2], bar[3] = max(bar[1], price), min(bar[2], price), price
                self.last_price, self.last_tick_ts = price, ts
                self._dirty = True
        self.first_tick.set()

    def _run(self):
        while True:
            try:
                self._poll_once()
            except Exception as e:
                with self.lock:
                    self.last_error = f"{type(e).__name__}: {e}"
            self._maybe_save()
            time.sleep(max(3.0, float(self.poll_s)))

    # -- accessors
    def latest(self):
        with self.lock:
            return self.last_price, self.last_tick_ts, self.last_poll_ts, self.last_error

    def bar_count(self) -> int:
        with self.lock:
            return len(self.bars)

    def snapshot(self):
        with self.lock:
            if not self.bars:
                return None
            items = sorted(self.bars.items())
            arr = np.array([v for _, v in items], dtype=float)
            idx = pd.DatetimeIndex([t for t, _ in items])
        return pd.DataFrame(arr, index=idx, columns=["Open", "High", "Low", "Close"])

    def seed(self, hist: pd.DataFrame):
        with self.lock:
            for t, r in hist.iterrows():
                if t not in self.bars:
                    self.bars[t] = [float(r["Open"]), float(r["High"]),
                                    float(r["Low"]), float(r["Close"])]
            self._dirty = True
        self._maybe_save(force=True)


@st.cache_resource
def get_recorder() -> SpotRecorder:
    return SpotRecorder()


def yahoo_spot_seed(rec: SpotRecorder) -> int:
    """One-time history pre-fill from Yahoo's XAUUSD=X spot pair. Never uses futures."""
    try:
        h = yf.Ticker(YAHOO_SPOT_SEED_SYMBOL).history(period="5d", interval="5m")
        h = h.dropna(subset=["Close"])
        if h.empty:
            return 0
        h.index = h.index.tz_localize("UTC") if h.index.tz is None else h.index.tz_convert("UTC")
        rec.seed(h[["Open", "High", "Low", "Close"]])
        return len(h)
    except Exception:
        return 0


# ----------------------------------------------------------------------------
# Settings persistence
# ----------------------------------------------------------------------------
def init_settings():
    ss = st.session_state
    if ss.get("_settings_ready"):
        return
    saved = {}
    try:
        saved = json.loads(SETTINGS_PATH.read_text())
    except Exception:
        pass
    for k, default in DEFAULTS.items():
        val = saved.get(k, default)
        try:
            if isinstance(default, bool):
                val = bool(val)
            elif isinstance(default, int):
                val = int(val)
            elif isinstance(default, float):
                val = float(val)
            else:
                val = str(val)
        except Exception:
            val = default
        if k in OPTIONS and val not in OPTIONS[k]:
            val = default
        ss[k] = val
    ss["_settings_ready"] = True


def save_settings():
    try:
        txt = json.dumps({k: st.session_state[k] for k in DEFAULTS}, indent=2)
        if not SETTINGS_PATH.exists() or SETTINGS_PATH.read_text() != txt:
            SETTINGS_PATH.write_text(txt)
    except Exception:
        pass


def apply_preset():
    for k, v in PRESETS.get(st.session_state.preset, {}).items():
        st.session_state[k] = v


# ----------------------------------------------------------------------------
# Notifications
# ----------------------------------------------------------------------------
def desktop_notify(title: str, message: str) -> bool:
    try:
        from plyer import notification
        notification.notify(title=title, message=message,
                            app_name="Gold Macro Signals", timeout=15)
        return True
    except Exception:
        pass
    try:
        system = platform.system()
        if system == "Darwin":
            t, m = title.replace('"', "'"), message.replace('"', "'")
            subprocess.run(["osascript", "-e",
                            f'display notification "{m}" with title "{t}" sound name "Glass"'],
                           check=True, timeout=10)
            return True
        if system == "Linux":
            subprocess.run(["notify-send", title, message], check=True, timeout=10)
            return True
    except Exception:
        pass
    return False


# ----------------------------------------------------------------------------
# Market session (approximate; holidays not modelled)
# ----------------------------------------------------------------------------
def is_market_open(now_et: datetime) -> bool:
    wd = now_et.weekday()
    t = now_et.hour + now_et.minute / 60
    if wd == 5:
        return False
    if wd == 4 and t >= 17:
        return False
    if wd == 6 and t < 18:
        return False
    if wd in (0, 1, 2, 3) and 17 <= t < 18:
        return False
    return True


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def fetch_asset(symbol: str):
    try:
        hist = yf.Ticker(symbol).history(period="5d", interval="5m")
        hist = hist.dropna(subset=["Close"])
        if len(hist) < 2:
            return None
        hist.index = hist.index.tz_localize("UTC") if hist.index.tz is None else hist.index.tz_convert("UTC")
        return {"last": float(hist["Close"].iloc[-1]), "bar_time": hist.index[-1], "hist": hist}
    except Exception:
        return None


def fetch_all(rec: SpotRecorder) -> dict:
    """Macro drivers from Yahoo + gold bars from the spot recorder."""
    with ThreadPoolExecutor(max_workers=len(ASSETS)) as pool:
        futs = {k: pool.submit(fetch_asset, ASSETS[k][0]) for k in ASSETS}
        data = {k: f.result() for k, f in futs.items()}
    rec.first_tick.wait(timeout=8)
    price, tick_ts, _, _ = rec.latest()
    bars = rec.snapshot()
    data["GOLD"] = ({"last": price, "bar_time": tick_ts, "hist": bars}
                    if price is not None and tick_ts is not None and bars is not None else None)
    return data


def compute_move(d: dict, lookback_min: int):
    """Price change vs. the bar `lookback_min` minutes earlier. None if no valid reference."""
    closes = d["hist"]["Close"]
    last_t, last = closes.index[-1], float(closes.iloc[-1])
    target = last_t - timedelta(minutes=lookback_min)
    prior = closes[closes.index <= target]
    if prior.empty:
        return None
    if target - prior.index[-1] > timedelta(minutes=15):   # gap in data
        return None
    ref = float(prior.iloc[-1])
    chg = last - ref
    return chg, (chg / ref * 100 if ref else 0.0)


def add_indicators(hist: pd.DataFrame, fast: int, slow: int, atr_n: int) -> pd.DataFrame:
    df = hist.copy()
    df["ema_fast"] = df["Close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=slow, adjust=False).mean()
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / atr_n, adjust=False).mean()
    return df


def htf_trend(gold: pd.DataFrame, fast: int, slow: int):
    """1-hour trend built from the 5m closes: 'up', 'down', 'flat' or None."""
    h = gold["Close"].resample("1h").last().dropna()
    if len(h) < slow + 5:
        return None
    ef = h.ewm(span=fast, adjust=False).mean().iloc[-1]
    es = h.ewm(span=slow, adjust=False).mean().iloc[-1]
    return "up" if ef > es else ("down" if ef < es else "flat")


# ----------------------------------------------------------------------------
# Scoring (weighted; positive = bullish gold)
# ----------------------------------------------------------------------------
def sgn(x) -> int:
    return int(np.sign(x))


def build_scorecard(data: dict):
    ss = st.session_state
    lb, dz = ss.lookback_min, ss.deadzone
    specs = [("DXY", ss.w_dxy), ("TNX", ss.w_tnx), ("VIX", ss.w_vix), ("ES", ss.w_es)]
    now = datetime.now(UTC)
    rows, drivers = [], []
    total, denom, live = 0.0, 0.0, 0

    for key, w in specs:
        symbol, name, dec = ASSETS[key]
        if key == "VIX":
            rule = (f"VIX > {ss.vix_hi:g} +1 / < {ss.vix_lo:g} -1"
                    if ss.vix_mode == VIX_LEVEL else "VIX rising +1 / falling -1")
        else:
            rule = "Falling +1 / rising -1"
        base = {"Driver": name, "Ticker": symbol, "Wt": w, "Rule": rule}
        mcol, pcol = f"{lb}m Move", f"{lb}m %"

        if w <= 0:
            rows.append({**base, "Live Value": "—", mcol: "—", pcol: "—",
                         "Data": "⚪ Disabled", "Score": 0.0})
            drivers.append(f"{key}off")
            continue
        d = data.get(key)
        if d is None:
            rows.append({**base, "Live Value": "N/A", mcol: "N/A", pcol: "N/A",
                         "Data": "❌ No data", "Score": 0.0})
            drivers.append(f"{key}x")
            continue

        age = (now - d["bar_time"]).total_seconds() / 60
        mv = compute_move(d, lb)
        value = f"{d['last']:,.{dec}f}"
        if mv is not None:
            chg, pct = mv
            arrow = "▲" if chg > 0 else ("▼" if chg < 0 else "■")
            move_s, pct_s = f"{arrow} {chg:+.{dec}f}", f"{pct:+.3f}%"
        else:
            chg = pct = None
            move_s = pct_s = "N/A"

        stale = age > ss.stale_m
        needs_move = not (key == "VIX" and ss.vix_mode == VIX_LEVEL)
        if stale:
            status = "🟡 Stale (ignored)"
        elif needs_move and mv is None:
            status = "🟡 Data gap (ignored)"
        else:
            status = "🟢 Live"

        pts = 0
        if status == "🟢 Live":
            if key == "VIX":
                if ss.vix_mode == VIX_LEVEL:
                    pts = 1 if d["last"] > ss.vix_hi else (-1 if d["last"] < ss.vix_lo else 0)
                else:
                    pts = 0 if abs(pct) < dz else sgn(pct)
            else:
                pts = 0 if abs(pct) < dz else -sgn(pct)
            live += 1
            denom += w
            total += w * pts
            drivers.append(f"{key}{pts:+d}")
        else:
            drivers.append(f"{key}x")

        rows.append({**base, "Live Value": value, mcol: move_s, pcol: pct_s,
                     "Data": status, "Score": float(w * pts)})

    return pd.DataFrame(rows), total, denom, live, " ".join(drivers)


def classify(score: float, denom: float):
    signed = score / denom * 100 if denom > 0 else 0.0
    conv = abs(signed)
    tier = ("High Conviction" if conv >= 75
            else "Moderate Conviction" if conv >= 50 else "Low Conviction / Chop")
    bias = "🟢 Bullish" if signed >= 50 else ("🔴 Bearish" if signed <= -50 else "🟡 Neutral")
    return bias, conv, tier


def parse_events(text: str):
    events, bad = [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(datetime.strptime(line[:16], "%Y-%m-%d %H:%M").replace(tzinfo=ET))
        except ValueError:
            bad += 1
    return events, bad


# ----------------------------------------------------------------------------
# Signal engine
# ----------------------------------------------------------------------------
def evaluate_signal(score, denom, live, gold, gold_ts, market_open, now_et, now_utc, events):
    ss = st.session_state
    no = lambda reason: {"side": "NO TRADE", "reason": reason}

    need = ss.ema_slow + 10
    have = 0 if gold is None else len(gold)
    if gold is None or have < need or gold_ts is None:
        return no(f"Warming up gold history: {have}/{need} five-minute bars")
    if not market_open:
        return no("Market closed")
    age = (now_utc - gold_ts).total_seconds() / 60
    if age > ss.gold_age:
        return no(f"Gold quote is {age:.0f} min old")

    enabled = sum(w > 0 for w in (ss.w_dxy, ss.w_tnx, ss.w_vix, ss.w_es))
    if enabled == 0:
        return no("All macro drivers are disabled")
    need_live = min(ss.min_live, enabled)
    if live < need_live:
        return no(f"Only {live}/{enabled} macro feeds live (need {need_live})")

    a, b, h = ss.sess_start, ss.sess_end, now_et.hour
    in_session = (a <= h < b) if a <= b else (h >= a or h < b)
    if ss.session_filter and a != b and not in_session:
        return no(f"Outside session window ({a:02d}:00-{b:02d}:00 ET)")

    for ev in events:
        if ev - timedelta(minutes=ss.blackout_before) <= now_et <= ev + timedelta(minutes=ss.blackout_after):
            return no(f"News blackout around {ev:%H:%M} ET event")

    last = gold.iloc[-1]
    atr = float(last["atr"])
    if np.isnan(atr) or atr < ss.min_atr:
        return no(f"ATR ${atr:.2f} below minimum ${ss.min_atr:.2f} (dead market)")

    if denom <= 0:
        return no("No live macro weight")
    conv = abs(score) / denom * 100
    if score == 0 or conv < ss.min_conv:
        return no(f"Macro conviction {conv:.0f}% is below the {ss.min_conv}% threshold")

    side, sign = ("LONG", 1) if score > 0 else ("SHORT", -1)
    if ss.direction == "Long only" and side == "SHORT":
        return no("Macro says SHORT but short signals are disabled")
    if ss.direction == "Short only" and side == "LONG":
        return no("Macro says LONG but long signals are disabled")

    c, ef, es = last["Close"], last["ema_fast"], last["ema_slow"]
    if ss.trend_mode == TREND_PRICE:
        ok = c > ef if sign == 1 else c < ef
    elif ss.trend_mode == TREND_FULL:
        ok = (c > ef > es) if sign == 1 else (c < ef < es)
    else:
        ok = True
    if not ok:
        return no(f"Macro says {side} but gold trend filter doesn't confirm")

    if ss.htf_filter:
        t = htf_trend(gold, ss.ema_fast, ss.ema_slow)
        if t is None:
            return no("Not enough history for the 1h trend filter")
        if (sign == 1 and t != "up") or (sign == -1 and t != "down"):
            return no(f"1h trend is {t}, against the {side}")

    risk = ss.atr_mult * atr
    entry = float(c)
    return {
        "side": side, "reason": "All conditions met", "entry": entry,
        "stop": entry - sign * risk, "target": entry + sign * ss.rr * risk,
        "atr": atr, "entry_bar_utc": gold.index[-1].isoformat(),
    }


def confirmed_side(history, n):
    if len(history) < n:
        return "NO TRADE"
    last = history[-n:]
    return last[0] if len(set(last)) == 1 and last[0] != "NO TRADE" else "NO TRADE"


# ----------------------------------------------------------------------------
# Signal log + outcome tracking
# ----------------------------------------------------------------------------
def load_log() -> pd.DataFrame:
    try:
        df = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame(columns=LOG_COLS)
    except Exception:
        df = pd.DataFrame(columns=LOG_COLS)
    for c in LOG_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[LOG_COLS].copy()
    df[FLOAT_COLS] = df[FLOAT_COLS].astype(float)
    df[OBJ_COLS] = df[OBJ_COLS].astype(object)
    return df.reset_index(drop=True)


def save_log(df: pd.DataFrame):
    try:
        df.to_csv(LOG_PATH, index=False)
    except Exception as e:
        st.warning(f"Could not write {LOG_PATH.name}: {e}")


def resolve_outcomes(log, gold, now_utc):
    expiry = timedelta(hours=st.session_state.expiry_h)
    for idx in log.index:
        row = log.loc[idx]
        opened = pd.to_datetime(row["time_utc"], utc=True)
        entry_bar = pd.to_datetime(row["entry_bar_utc"], utc=True)
        bars = gold[gold.index > entry_bar]

        for mins, col in ((15, "px_15m"), (30, "px_30m"), (60, "px_60m")):
            if pd.isna(row[col]):
                t_target = entry_bar + timedelta(minutes=mins)
                cand = bars[bars.index >= t_target]
                if not cand.empty and cand.index[0] <= t_target + timedelta(minutes=10):
                    log.at[idx, col] = float(cand.iloc[0]["Open"])

        if row["status"] != "OPEN":
            continue

        side, entry = row["side"], float(row["entry"])
        stop, target = float(row["stop"]), float(row["target"])
        risk = abs(entry - stop) or 1e-9
        rr = abs(target - entry) / risk
        resolved = False
        for t, bar in bars.iterrows():
            if side == "LONG":
                hit_stop, hit_tgt = bar["Low"] <= stop, bar["High"] >= target
            else:
                hit_stop, hit_tgt = bar["High"] >= stop, bar["Low"] <= target
            if hit_stop:   # conservative: stop wins if both hit in one bar
                log.at[idx, "status"], log.at[idx, "exit_price"] = "LOSS", stop
                log.at[idx, "r_result"], log.at[idx, "closed_utc"] = -1.0, t.isoformat()
                resolved = True
                break
            if hit_tgt:
                log.at[idx, "status"], log.at[idx, "exit_price"] = "WIN", target
                log.at[idx, "r_result"], log.at[idx, "closed_utc"] = rr, t.isoformat()
                resolved = True
                break
        if not resolved and now_utc - opened >= expiry:
            exit_px = float(bars["Close"].iloc[-1]) if not bars.empty else entry
            sign = 1 if side == "LONG" else -1
            log.at[idx, "status"], log.at[idx, "exit_price"] = "EXPIRED", exit_px
            log.at[idx, "r_result"] = sign * (exit_px - entry) / risk
            log.at[idx, "closed_utc"] = now_utc.isoformat()
    return log


def maybe_log_new_signal(raw, score, drivers, now_utc) -> bool:
    ss = st.session_state
    log = ss.log
    if int((log["status"] == "OPEN").sum()) >= ss.max_open:
        return False
    if not log.empty and ss.cooldown_m > 0:
        last_t = pd.to_datetime(log["time_utc"], utc=True).max()
        if now_utc - last_t < timedelta(minutes=ss.cooldown_m):
            return False
    new_id = int(log["id"].max()) + 1 if not log.empty and log["id"].notna().any() else 1
    row = {
        "id": new_id, "time_utc": now_utc.isoformat(), "side": raw["side"],
        "entry": raw["entry"], "stop": raw["stop"], "target": raw["target"],
        "net_score": score, "atr": raw["atr"], "entry_bar_utc": raw["entry_bar_utc"],
        "status": "OPEN", "exit_price": np.nan, "r_result": np.nan, "closed_utc": np.nan,
        "px_15m": np.nan, "px_30m": np.nan, "px_60m": np.nan,
        "drivers": drivers, "src": SRC_NAME,
    }
    log.loc[len(log)] = [row[c] for c in LOG_COLS]
    ss.log = log
    save_log(log)
    return True


def rebuild_gold_df():
    ss = st.session_state
    raw = ss.get("gold_raw")
    ss.gold_df = (add_indicators(raw, ss.ema_fast, ss.ema_slow, ss.atr_period)
                  if raw is not None else None)


def process_signals(market_open: bool, now_et: datetime, counts: bool):
    ss = st.session_state
    now_utc = datetime.now(UTC)
    _, score, denom, live, drivers = build_scorecard(ss.data)
    gold = ss.gold_df
    gold_ts = ss.data["GOLD"]["bar_time"] if ss.data.get("GOLD") else None

    if gold is not None:
        ss.log = resolve_outcomes(ss.log, gold, now_utc)
        save_log(ss.log)

    raw = evaluate_signal(score, denom, live, gold, gold_ts, market_open, now_et, now_utc, ss.events)
    ss.raw = raw
    if counts:
        ss.raw_history = (ss.raw_history + [raw["side"]])[-10:]
        ss.last_confirm_ts = time.time()
    conf = confirmed_side(ss.raw_history, ss.confirm_n)
    ss.confirmed = conf

    if counts and conf != "NO TRADE" and raw["side"] == conf:
        if maybe_log_new_signal(raw, score, drivers, now_utc):
            msg = (f"{raw['side']} XAU/USD spot @ {raw['entry']:.2f} | "
                   f"SL {raw['stop']:.2f} | TP {raw['target']:.2f}")
            st.toast(f"🚨 {msg}", icon="🥇")
            if ss.notify_on and not desktop_notify(f"Gold {raw['side']} signal", msg):
                st.toast("Desktop notification failed - is plyer installed?", icon="⚠️")


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
init_settings()
get_recorder()   # starts the background spot poller as soon as the app launches

with st.sidebar:
    st.header("⚙️ Settings")
    st.selectbox("Preset", OPTIONS["preset"], key="preset", on_change=apply_preset,
                 help="Loads a bundle of settings. Tweak anything afterwards. "
                      "Aggressive = most signals, Conservative = fewest.")
    st.caption("Changes apply instantly and are saved to settings_v4.json")

    with st.expander("📡 Data & timing"):
        st.caption("Gold: XAU/USD spot via gold-api.com (locked)")
        st.slider("Spot poll interval (seconds)", min_value=5, max_value=60, step=5, key="poll_s")
        st.toggle("Pre-fill history from Yahoo XAUUSD=X (spot)", key="yahoo_seed",
                  help="One-time, at startup, so EMAs/ATR work immediately. "
                       "Turn off to build history purely from gold-api.com (takes hours).")
        st.slider("Macro refresh interval (seconds)", min_value=60, max_value=900, step=30, key="refresh_s")
        st.slider("Ignore macro feeds older than (min)", min_value=10, max_value=180, key="stale_m")
        st.slider("Max gold quote age for signals (min)", min_value=2, max_value=60, key="gold_age")

    with st.expander("🧮 Macro scoring", expanded=True):
        st.select_slider("Move lookback (minutes)", options=OPTIONS["lookback_min"], key="lookback_min",
                         help="Longer = smoother and more stable, shorter = faster but noisier.")
        st.slider("Dead-zone: ignore moves smaller than (%)", min_value=0.0, max_value=0.2,
                  step=0.005, format="%.3f", key="deadzone")
        st.caption("Driver weights (0 = disabled)")
        wc1, wc2 = st.columns(2)
        wc1.slider("DXY", min_value=0.0, max_value=2.0, step=0.25, key="w_dxy")
        wc2.slider("US10Y", min_value=0.0, max_value=2.0, step=0.25, key="w_tnx")
        wc1.slider("VIX", min_value=0.0, max_value=2.0, step=0.25, key="w_vix")
        wc2.slider("S&P", min_value=0.0, max_value=2.0, step=0.25, key="w_es")
        st.selectbox("VIX scoring mode", OPTIONS["vix_mode"], key="vix_mode")
        st.slider("VIX risk-off level (>)", min_value=12.0, max_value=40.0, step=0.5, key="vix_hi")
        st.slider("VIX complacent level (<)", min_value=8.0, max_value=25.0, step=0.5, key="vix_lo")
        st.slider("Min macro conviction to signal (%)", min_value=25, max_value=100, step=5, key="min_conv")
        st.slider("Min live drivers required", min_value=1, max_value=4, key="min_live")

    with st.expander("📈 Gold trend & volatility filters"):
        st.selectbox("Trend filter", OPTIONS["trend_mode"], key="trend_mode")
        st.slider("Fast EMA", min_value=3, max_value=50, key="ema_fast")
        st.slider("Slow EMA", min_value=10, max_value=200, key="ema_slow")
        if st.session_state.ema_fast >= st.session_state.ema_slow:
            st.warning("Fast EMA should be shorter than Slow EMA")
        st.slider("ATR period", min_value=5, max_value=30, key="atr_period")
        st.toggle("Require 1h trend alignment", key="htf_filter",
                  help="Built from the 5m data: 1h fast EMA vs slow EMA must agree with the trade.")
        st.slider("Min ATR ($) — skip dead markets", min_value=0.0, max_value=10.0, step=0.1, key="min_atr")

    with st.expander("🎯 Signal rules & risk"):
        st.selectbox("Allowed direction", OPTIONS["direction"], key="direction")
        st.slider("Confirmations (consecutive refreshes)", min_value=1, max_value=5, key="confirm_n")
        st.slider("Cooldown between signals (min)", min_value=0, max_value=240, step=5, key="cooldown_m")
        st.slider("Max simultaneous open signals", min_value=1, max_value=3, key="max_open")
        st.slider("Stop distance (× ATR)", min_value=0.5, max_value=4.0, step=0.1, key="atr_mult")
        st.slider("Reward : Risk", min_value=0.5, max_value=4.0, step=0.5, key="rr")
        st.slider("Expire open signals after (hours)", min_value=1, max_value=24, key="expiry_h")

    with st.expander("🕒 Session & news filters"):
        st.toggle("Restrict to a session window (ET)", key="session_filter")
        sc1, sc2 = st.columns(2)
        sc1.slider("Start hour", min_value=0, max_value=23, key="sess_start")
        sc2.slider("End hour", min_value=1, max_value=24, key="sess_end")
        bc1, bc2 = st.columns(2)
        bc1.slider("Blackout before (min)", min_value=0, max_value=180, step=5, key="blackout_before")
        bc2.slider("Blackout after (min)", min_value=0, max_value=180, step=5, key="blackout_after")
        st.text_area("News events (ET), one per line: YYYY-MM-DD HH:MM", key="events_text",
                     height=100, placeholder="2026-11-06 08:30\n2026-12-16 14:00")
        _events, _bad = parse_events(st.session_state.events_text)
        st.session_state.events = _events
        st.caption(f"{len(_events)} blackout event(s) loaded")
        if _bad:
            st.warning(f"{_bad} line(s) couldn't be parsed")

    with st.expander("🔔 Alerts"):
        st.toggle("Desktop notifications", key="notify_on")
        if st.button("Send test notification"):
            ok = desktop_notify("Gold Macro Signals", "Test notification - it works!")
            st.success("Sent!") if ok else st.error("Failed. Run: pip install plyer")

get_recorder().poll_s = st.session_state.poll_s
save_settings()


# ----------------------------------------------------------------------------
# Dashboard fragment
# ----------------------------------------------------------------------------
@st.fragment(run_every="1s")
def dashboard():
    ss = st.session_state
    rec = get_recorder()
    now_et = datetime.now(ET)
    market_open = is_market_open(now_et)
    now_ts = time.time()

    for k, v in {"data": None, "gold_raw": None, "gold_df": None,
                 "last_fetch_ts": 0.0, "last_update": None,
                 "last_confirm_ts": 0.0, "raw_history": [], "confirmed": "NO TRADE",
                 "raw": {"side": "NO TRADE", "reason": "Waiting for data"},
                 "params_sig": None, "events": []}.items():
        ss.setdefault(k, v)
    if "log" not in ss:
        ss.log = load_log()

    # Detect settings changes (so they apply instantly without a re-download)
    sig = tuple(ss[k] for k in DEFAULTS if k not in SIG_EXCLUDE)
    params_changed = ss.params_sig is not None and ss.params_sig != sig
    ss.params_sig = sig

    top_l, top_r = st.columns([5, 1])
    with top_r:
        manual = st.button("🔄 Refresh now", use_container_width=True)

    first_load = ss.data is None
    due = (now_ts - ss.last_fetch_ts) >= ss.refresh_s

    if first_load or manual or (market_open and due):
        with st.spinner("Fetching market data..."):
            if first_load and ss.yahoo_seed and not rec.seeded:
                rec.seeded = True
                if rec.bar_count() < SEED_IF_BARS_BELOW:
                    yahoo_spot_seed(rec)
            data = fetch_all(rec)
        ss.data = data
        ss.gold_raw = data["GOLD"]["hist"] if data.get("GOLD") else None
        rebuild_gold_df()
        ss.last_fetch_ts = time.time()
        ss.last_update = datetime.now().astimezone()
        now_ts = ss.last_fetch_ts
        counts = first_load or (time.time() - ss.last_confirm_ts) >= 0.8 * ss.refresh_s
        process_signals(market_open, now_et, counts)
    elif params_changed and ss.data is not None:
        rebuild_gold_df()
        ss.raw_history = []
        process_signals(market_open, now_et, False)

    # ---- live gold tick (updates every second)
    g_price, g_ts, g_poll, g_err = rec.latest()
    tick_age_s = (datetime.now(UTC) - g_ts).total_seconds() if g_ts is not None else None

    # ---- status bar
    remaining = max(0, int(ss.refresh_s - (now_ts - ss.last_fetch_ts)))
    mins, secs = divmod(remaining, 60)
    if market_open:
        countdown = f"⏱ Macro refresh in <b>{mins:02d}:{secs:02d}</b>"
        session = "🟢 Market OPEN"
    else:
        countdown = "⏸ Auto-refresh paused (market closed)"
        session = "🔴 Market CLOSED"
    if g_price is not None:
        age_txt = f"{int(tick_age_s)}s ago" if tick_age_s < 120 else f"{tick_age_s / 60:.0f} min ago"
        gold_txt = f"🥇 XAU/USD <b>${g_price:,.2f}</b> ({age_txt})"
    else:
        gold_txt = "🥇 waiting for gold-api.com…"
    with top_l:
        st.markdown(
            f'<div class="status-bar">🕒 Macro updated: <b>{ss.last_update.strftime("%H:%M:%S %Z")}</b> '
            f'&nbsp;|&nbsp; {countdown} &nbsp;|&nbsp; {session} &nbsp;|&nbsp; '
            f'{gold_txt} &nbsp;|&nbsp; ET {now_et.strftime("%a %H:%M")}</div>',
            unsafe_allow_html=True)
    if g_err:
        st.warning(f"gold-api.com: {g_err}")

    data = ss.data
    df_sc, score, denom, live, drivers = build_scorecard(data)
    bias, conv, tier = classify(score, denom)
    raw, confirmed = ss.raw, ss.confirmed

    # ---- headline metrics
    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Macro Bias", bias, f"Net score {score:+.2f} / {denom:.2f}", delta_color="off")
    c2.metric("Confidence", f"{conv:.0f}%")
    c3.metric("Rating Tier", tier)
    if confirmed != "NO TRADE":
        label, sub = ("🟢 " if confirmed == "LONG" else "🔴 ") + confirmed, "Confirmed"
    elif raw["side"] != "NO TRADE":
        label, sub = "⏳ " + raw["side"], f"Pending {ss.confirm_n}x confirmation"
    else:
        label, sub = "⚪ NO TRADE", None
    c4.metric("Signal", label, sub, delta_color="off")

    # ---- active signals
    st.subheader("Active Signals")
    open_sigs = ss.log[ss.log["status"] == "OPEN"]
    for _, s in open_sigs.iterrows():
        opened = pd.to_datetime(s["time_utc"], utc=True)
        age_min = int((datetime.now(UTC) - opened).total_seconds() // 60)
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Side", ("🟢 " if s["side"] == "LONG" else "🔴 ") + s["side"])
        a2.metric("Entry", f"{s['entry']:,.2f}")
        a3.metric("Stop", f"{s['stop']:,.2f}")
        a4.metric("Target", f"{s['target']:,.2f}")
        a5.metric("Open for", f"{age_min} min")
    st.info(f"Scanner status: {raw.get('reason', '')}"
            + ("" if raw["side"] == "NO TRADE" else f" ({raw['side']} candidate)"))

    # ---- price strip (gold is live from the recorder)
    st.subheader("Live Prices")
    strip = []
    gold_bars = rec.snapshot() if g_price is not None else None
    strip.append((GOLD_LABEL, {"last": g_price, "hist": gold_bars} if gold_bars is not None else None, 2))
    for key, (symbol, name, dec) in ASSETS.items():
        strip.append((name, data.get(key), dec))
    cols = st.columns(len(strip))
    for col, (name, d, dec) in zip(cols, strip):
        if d is None:
            col.metric(name, "N/A")
            continue
        mv = compute_move(d, ss.lookback_min)
        if mv:
            col.metric(name, f"{d['last']:,.{dec}f}", f"{mv[0]:+.{dec}f} ({mv[1]:+.2f}%)")
        else:
            col.metric(name, f"{d['last']:,.{dec}f}")

    # ---- gold chart
    gold = ss.gold_df
    if gold is not None and len(gold) > 20:
        st.subheader(f"{GOLD_LABEL} 5m — Price vs EMA{ss.ema_fast} / EMA{ss.ema_slow}")
        chart = gold[["Close", "ema_fast", "ema_slow"]].tail(150).copy()
        chart.columns = ["Close", f"EMA{ss.ema_fast}", f"EMA{ss.ema_slow}"]
        chart.index = chart.index.tz_convert(ET).tz_localize(None)
        st.line_chart(chart, height=280)

    # ---- driver table
    st.subheader("Driver Breakdown")
    st.dataframe(df_sc, hide_index=True, use_container_width=True,
                 column_config={"Score": st.column_config.NumberColumn("Score", format="%+.2f"),
                                "Wt": st.column_config.NumberColumn("Wt", format="%.2f")})
    if any(data.get(k) is None for k in ASSETS):
        st.warning("Some Yahoo macro feeds returned no data (possible rate-limiting). "
                   "Missing/stale drivers are excluded from the score.")

    # ---- history
    st.subheader("Signal History")
    log = ss.log
    closed = log[log["status"].isin(["WIN", "LOSS", "EXPIRED"])]
    wins = int((closed["status"] == "WIN").sum())
    losses = int((closed["status"] == "LOSS").sum())
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Total signals", len(log))
    h2.metric("Closed", len(closed))
    h3.metric("Win rate (TP vs SL)", f"{wins / (wins + losses) * 100:.0f}%" if wins + losses else "—")
    h4.metric("Avg result", f"{closed['r_result'].mean():+.2f}R" if len(closed) else "—")
    h5.metric("Total result", f"{closed['r_result'].sum():+.2f}R" if len(closed) else "—")

    if log.empty:
        st.caption(f"No signals yet. Confirmed signals appear here and in {LOG_PATH.name}.")
    else:
        v = log.copy().iloc[::-1]
        sign = np.where(v["side"] == "LONG", 1, -1)
        out = pd.DataFrame({
            "Time (ET)": pd.to_datetime(v["time_utc"], utc=True).dt.tz_convert(ET).dt.strftime("%m-%d %H:%M"),
            "Side": v["side"],
            "Entry": v["entry"].round(2),
            "Stop": v["stop"].round(2),
            "Target": v["target"].round(2),
            "Score": v["net_score"].round(2),
            "Drivers": v["drivers"],
            "Status": v["status"].map({"OPEN": "🔵 OPEN", "WIN": "✅ WIN", "LOSS": "❌ LOSS",
                                       "EXPIRED": "⌛ EXPIRED"}),
            "R": v["r_result"].round(2),
            "15m P/L $": ((v["px_15m"] - v["entry"]) * sign).round(2),
            "30m P/L $": ((v["px_30m"] - v["entry"]) * sign).round(2),
            "60m P/L $": ((v["px_60m"] - v["entry"]) * sign).round(2),
        })
        st.dataframe(out, hide_index=True, use_container_width=True)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.title("🥇 Gold (XAU/USD Spot) Macro Signals — v4")
st.caption("Gold: XAU/USD spot via gold-api.com. Macro drivers via Yahoo Finance (delayed/unofficial). "
           "Paper trade first — not financial advice.")
dashboard()
