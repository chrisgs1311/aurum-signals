"""
aurum_state.py — Shared global state for all Aurum modules.
All mutable globals live here as mutable containers (list/dict) so cross-module
imports via `from aurum_state import x` share the same object by reference.
Scalars that were formerly reassigned are now wrapped in mutable dicts.
"""
import threading, json, time, os, math
import datetime as _dt
from datetime import datetime, timezone

# ── THREAD SAFETY ─────────────────────────────────────────
_data_lock    = threading.Lock()  # price_history, _scalp_prices_*, _ohlc_candles
_control_lock = threading.Lock()  # _current_control
_news_lock    = threading.Lock()  # _news_cache, _news_alerted
_risk_lock    = threading.Lock()  # RISK_STATE
_cache_lock   = threading.Lock()  # _cache_store
_sse_lock     = threading.Lock()  # _sse_clients

# ── RETRY HELPER ──────────────────────────────────────────
import urllib.request

def _fetch_with_retry(url, headers=None, timeout=10, retries=3, backoff=1.0):
    """HTTP GET con retry + exponential backoff."""
    hdrs = headers or {"User-Agent": "Mozilla/5.0"}
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
    print(f"  ⚠ Fetch failed after {retries} retries: {url[:80]}... | {last_err}")
    return None

# ── CACHE EN MEMORIA ─────────────────────────────────────
_cache_store = {}

def cached(key, ttl=30):
    with _cache_lock:
        e = _cache_store.get(key)
        if e and time.time() - e["ts"] < ttl: return e["val"]
    return None

def set_cache(key, val):
    with _cache_lock:
        _cache_store[key] = {"val": val, "ts": time.time()}

# ── HISTORIAL GLOBAL DE PRECIOS ───────────────────────────
price_history = []

def push_price(p):
    with _data_lock:
        price_history.append(p)
        if len(price_history) > 2000:
            del price_history[:-2000]

# ── LIVE DATA CACHE (workers → RAM → endpoints) ───────────
STALE_PRICE_SEC = 30
STALE_OHLC_SEC  = 180

_live_cache = {
    "price": None, "price_ts": 0,
    "ohlc_5m": [], "ohlc_15m": [], "ohlc_ts": 0,
    "signal": None, "signal_ts": 0,
    "ai_prob": None, "ai_signal": None,
    "price_stale": False, "ohlc_stale": False,
    "price_fails": 0,
}
_sse_clients = []

def is_data_stale():
    now = time.time()
    price_age = now - _live_cache["price_ts"] if _live_cache["price_ts"] else 999
    ohlc_age  = now - _live_cache["ohlc_ts"]  if _live_cache["ohlc_ts"]  else 999
    _live_cache["price_stale"] = price_age > STALE_PRICE_SEC
    _live_cache["ohlc_stale"]  = ohlc_age  > STALE_OHLC_SEC
    return _live_cache["price_stale"] or _live_cache["ohlc_stale"]

def _push_sse(event_type, data):
    msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    with _sse_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.append(msg)
            except:
                dead.append(q)
        for d in dead:
            _sse_clients.remove(d)

# ── FILTRO DE NOTICIAS ────────────────────────────────────
_news_cache   = {"events": [], "fetched_at": 0}
_news_alerted = set()
PAUSE_MINUTES_BEFORE = 30
PAUSE_MINUTES_AFTER  = 30
ALERT_MINUTES_BEFORE = [60, 15, 5]
HIGH_IMPACT_KEYWORDS = [
    "Non-Farm", "NFP", "CPI", "Core CPI", "PPI", "FOMC", "Fed Rate",
    "Federal Funds", "Unemployment Rate", "ADP", "GDP", "Retail Sales",
    "Core PCE", "PCE", "JOLTS", "ISM", "Powell", "Jackson Hole",
    "ECB", "Lagarde", "Initial Jobless Claims",
]

# ── PRICE HISTORY BUFFERS ─────────────────────────────────
_scalp_prices_1m  = []
_scalp_prices_5m  = []
_scalp_prices_15m = []
_ohlc_candles_5m  = []
_ohlc_candles_15m = []

# Timer dicts (replace reassigned scalars — safe for cross-module mutation)
_ohlc_timers = {"last_5m": 0.0, "last_15m": 0.0}
_ai_timers   = {"last_train": 0.0, "last_backup": 0.0}

# ── CONTROL STATE ─────────────────────────────────────────
_current_control = {
    "state": "FULL",
    "thresholds": {"buy": 0.63, "sell": 0.37},
    "lock_until": 0,
    "combined": 1.0,
}

# ── RISK STATE ────────────────────────────────────────────
RISK_STATE = {
    "daily_loss_limit_r": -3.0,
    "weekly_loss_limit_r": -8.0,
    "max_consecutive_losses": 3,
    "circuit_breaker_hours": 4,
    "paused_until": 0,
    "pause_reason": "",
}

# ── PAPER TRADING STATE ───────────────────────────────────
PAPER_TRADES_FILE = "/data/paper_trades.json"
PAPER_SPREAD_USD  = float(os.environ.get("PAPER_SPREAD_USD", "0.30"))
PAPER_SLIP_USD    = float(os.environ.get("PAPER_SLIP_USD",   "0.10"))
_paper_trades = []
_paper_lock   = threading.Lock()

# ── SIGNAL DEDUP ──────────────────────────────────────────
_last_scalp_signal  = {"key": "", "time": 0}
_last_any_signal_ts = 0   # global cooldown — cualquier señal

# ── PRE-SIGNAL STATE ──────────────────────────────────────
_pre_signal = {
    "state": "IDLE",
    "bias": None,
    "sweep_level": 0,
    "fvg_zone": None,
    "ts": 0,
}

# ── PERFORMANCE TRACKING ──────────────────────────────────
_signal_history = []
_perf_counters  = {
    "cons_losses": 0, "cons_wins": 0,
    "daily_signals": 0, "last_signal_date": "",
}
MAX_CONSECUTIVE_LOSSES = 3
MAX_DAILY_SIGNALS      = 4
_performance_monitor = {
    "rolling_wr": 0.0, "rolling_pf": 1.0, "rolling_sharpe": 0.0,
    "status": "OK",
    "baseline_wr": 55.0, "baseline_pf": 1.5, "baseline_sharpe": 1.0,
    "last_update": 0,
    "shutdown_until": 0, "shutdown_reason": "",
}
PERF_WINDOW = 50
PERF_MIN_N  = 20

# ── SCORING THRESHOLDS ────────────────────────────────────
SCORE_SNIPER        = 165
SCORE_NORMAL        = 125
SCORE_EARLY         = 999    # desactivado
SIGNAL_COOLDOWN_SEC = 3600   # 1 hora entre cualquier señal

# ── MACRO CACHES ─────────────────────────────────────────
_mtf_cache  = {"prices_1h": [], "last_update": 0}
_dxy_cache  = {"trend": "neutral", "change_pct": 0.0, "last_update": 0}
_us10y_cache = {"trend": "neutral", "change_bps": 0.0, "last_update": 0, "level": 0.0}

# ── API CONFIG ────────────────────────────────────────────
MASSIVE_API_KEY  = os.environ.get("MASSIVE_API_KEY", "")
TWELVE_API_KEY   = os.environ.get("TWELVE_API_KEY", "")
TWELVE_DAILY_LIMIT = 750

_api_counter = {
    "twelve_calls": 0, "twelve_date": "",
    "gold_calls":   0, "gold_date":   "",
}

# ── APP CONSTANTS ─────────────────────────────────────────
PORT             = int(os.environ.get("PORT", 8765))
LOG_FILE         = "aurum_operaciones.csv"
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── BACKTEST STATE ────────────────────────────────────────
_backtest_result    = {"data": None}
_walkforward_result = {"data": None}
