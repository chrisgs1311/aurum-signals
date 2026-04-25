"""aurum_signal.py — ICT signal engine, session helpers, signal detection."""
import math, time
import datetime as _dt

from aurum_state import (
    _data_lock, _control_lock,
    _scalp_prices_1m, _scalp_prices_5m, _scalp_prices_15m,
    _ohlc_candles_5m, _ohlc_candles_15m,
    _ohlc_timers, _last_scalp_signal, _pre_signal,
    _current_control, _live_cache,
    SCORE_SNIPER, SCORE_NORMAL, SCORE_EARLY, SIGNAL_COOLDOWN_SEC,
    is_data_stale,
)
import aurum_state as _state
from aurum_models import _ai, ai_predict
from aurum_news import (
    is_news_time, check_dxy_correlation, check_yields_for_gold,
)


# ── SESIONES ─────────────────────────────────────────────
def is_trading_session():
    h = _dt.datetime.utcnow().hour
    return (3 <= h < 11) or (12 <= h < 17)


def get_session_name():
    h = _dt.datetime.utcnow().hour
    if  3 <= h < 11: return "Londres 🇬🇧"
    if 12 <= h < 17: return "New York 🇺🇸"
    return "Fuera de sesión 🌙"


# ── SWING STRUCTURE ───────────────────────────────────────
def _swing_highs_ohlc(candles, n=3):
    highs = []
    for i in range(n, len(candles)-n):
        h = candles[i]["h"]
        if all(h >= candles[i-j]["h"] for j in range(1,n+1)) and \
           all(h >= candles[i+j]["h"] for j in range(1,n+1)):
            highs.append((i, h))
    return highs


def _swing_lows_ohlc(candles, n=3):
    lows = []
    for i in range(n, len(candles)-n):
        lo = candles[i]["l"]
        if all(lo <= candles[i-j]["l"] for j in range(1,n+1)) and \
           all(lo <= candles[i+j]["l"] for j in range(1,n+1)):
            lows.append((i, lo))
    return lows


def _swing_highs(prices, n=3):
    highs = []
    for i in range(n, len(prices)-n):
        if all(prices[i] >= prices[i-j] for j in range(1,n+1)) and \
           all(prices[i] >= prices[i+j] for j in range(1,n+1)):
            highs.append((i, prices[i]))
    return highs


def _swing_lows(prices, n=3):
    lows = []
    for i in range(n, len(prices)-n):
        if all(prices[i] <= prices[i-j] for j in range(1,n+1)) and \
           all(prices[i] <= prices[i+j] for j in range(1,n+1)):
            lows.append((i, prices[i]))
    return lows


# ── SWEEP DETECTION ───────────────────────────────────────
def _is_real_sweep(prices, sweep_level, bias, candles=None, atr=None):
    if not atr or atr <= 0:
        if len(prices) >= 15:
            atr = sum(abs(prices[i]-prices[i-1]) for i in range(len(prices)-14, len(prices))) / 14
        else:
            return True
    if bias == "bullish":
        penetration = sweep_level - min(prices[-5:])
    else:
        penetration = max(prices[-5:]) - sweep_level
    if penetration < atr * 0.5:
        return False
    if len(prices) >= 3:
        body = abs(prices[-1] - prices[-2])
        rng  = max(prices[-3:]) - min(prices[-3:]) + 1e-9
        if body / rng < 0.45:
            return False
    if bias == "bullish" and prices[-1] < prices[-2]: return False
    if bias == "bearish" and prices[-1] > prices[-2]: return False
    return True


def _classify_sweep_type(prices, bias, candles=None):
    if len(prices) < 20: return "UNKNOWN", 0.5
    pre_trend  = prices[-10] - prices[-15] if len(prices) >= 15 else 0
    post_trend = prices[-1]  - prices[-3]  if len(prices) >= 3  else 0
    if bias == "bullish":
        if pre_trend < 0 and post_trend > 0:
            return "REVERSA", round(min(1.0, abs(post_trend / (abs(pre_trend) + 1e-9))), 2)
        elif pre_trend > 0:
            return "CONTINUATION", 0.3
        return "NEUTRAL", 0.5
    if bias == "bearish":
        if pre_trend > 0 and post_trend < 0:
            return "REVERSA", round(min(1.0, abs(post_trend / (abs(pre_trend) + 1e-9))), 2)
        elif pre_trend < 0:
            return "CONTINUATION", 0.3
        return "NEUTRAL", 0.5
    return "UNKNOWN", 0.5


def detect_sweep_and_rejection(prices, bias, candles=None):
    if candles and len(candles) >= 20:
        return _detect_sweep_ohlc(candles, bias)
    if len(prices) < 20: return False, 0, 0
    p = prices
    if bias == "bullish":
        lows = _swing_lows(p[:-3], n=2)
        if not lows: return False, 0, 0
        prev_low_idx, prev_low = lows[-1]
        recent_low   = min(p[-5:])
        recent_close = p[-1]
        if recent_low < prev_low and recent_close > prev_low:
            candle_range = max(p[-4:]) - min(p[-4:]) + 1e-9
            wick = prev_low - recent_low
            return True, round(prev_low, 2), round(wick / candle_range, 2)
    if bias == "bearish":
        highs = _swing_highs(p[:-3], n=2)
        if not highs: return False, 0, 0
        prev_high_idx, prev_high = highs[-1]
        recent_high  = max(p[-5:])
        recent_close = p[-1]
        if recent_high > prev_high and recent_close < prev_high:
            candle_range = max(p[-4:]) - min(p[-4:]) + 1e-9
            wick = recent_high - prev_high
            return True, round(prev_high, 2), round(wick / candle_range, 2)
    return False, 0, 0


def _detect_sweep_ohlc(candles, bias):
    c = candles
    if bias == "bullish":
        lows = _swing_lows_ohlc(c[:-3], n=2)
        if not lows: return False, 0, 0
        prev_low_idx, prev_low = lows[-1]
        recent_lows  = [x["l"] for x in c[-5:]]
        recent_close = c[-1]["c"]
        min_wick = min(recent_lows)
        if min_wick < prev_low and recent_close > prev_low:
            sweep_candle = min(range(len(c[-5:])), key=lambda j: c[-5+j]["l"])
            sc = c[-5 + sweep_candle]
            candle_range = sc["h"] - sc["l"] + 1e-9
            return True, round(prev_low, 2), round((prev_low - sc["l"]) / candle_range, 2)
    if bias == "bearish":
        highs = _swing_highs_ohlc(c[:-3], n=2)
        if not highs: return False, 0, 0
        prev_high_idx, prev_high = highs[-1]
        recent_highs = [x["h"] for x in c[-5:]]
        recent_close = c[-1]["c"]
        max_wick = max(recent_highs)
        if max_wick > prev_high and recent_close < prev_high:
            sweep_candle = max(range(len(c[-5:])), key=lambda j: c[-5+j]["h"])
            sc = c[-5 + sweep_candle]
            candle_range = sc["h"] - sc["l"] + 1e-9
            return True, round(prev_high, 2), round((sc["h"] - prev_high) / candle_range, 2)
    return False, 0, 0


# ── BOS / FVG ─────────────────────────────────────────────
def detect_bos_scalp(prices, bias):
    if len(prices) < 20: return False, "none"
    p = prices
    if bias == "bullish":
        highs = _swing_highs(p[:-3], n=3)
        if highs and p[-1] > highs[-1][1]:
            return True, "BOS alcista"
        # CHOCH: requiere que previo swing low sea roto con momentum real (2% ATR minimo)
        if len(p) >= 15:
            atr = sum(abs(p[i]-p[i-1]) for i in range(len(p)-14, len(p))) / 14
            prev_swing_lows = _swing_lows(p[:-5], n=3)
            if prev_swing_lows:
                last_low = prev_swing_lows[-1][1]
                # precio rompe swing high despues de hacer un low mas alto
                local_lows = _swing_lows(p[-15:], n=2)
                if local_lows and local_lows[-1][1] > last_low and (p[-1] - local_lows[-1][1]) > atr * 1.5:
                    return True, "CHOCH"
    if bias == "bearish":
        lows = _swing_lows(p[:-3], n=3)
        if lows and p[-1] < lows[-1][1]:
            return True, "BOS bajista"
        if len(p) >= 15:
            atr = sum(abs(p[i]-p[i-1]) for i in range(len(p)-14, len(p))) / 14
            prev_swing_highs = _swing_highs(p[:-5], n=3)
            if prev_swing_highs:
                last_high = prev_swing_highs[-1][1]
                local_highs = _swing_highs(p[-15:], n=2)
                if local_highs and local_highs[-1][1] < last_high and (local_highs[-1][1] - p[-1]) > atr * 1.5:
                    return True, "CHOCH"
    return False, "none"


def detect_fvg_scalp(prices, bias, candles=None):
    if candles and len(candles) >= 5:
        return _detect_fvg_ohlc(candles, bias)
    if len(prices) < 5: return False, 0, 0
    for i in range(len(prices)-4, len(prices)-1):
        if i < 2: continue
        low1  = prices[i-2]
        high3 = prices[i]
        if bias == "bullish" and high3 > low1:
            gap = high3 - low1
            if gap > prices[-1] * 0.0015:
                return True, round(low1, 2), round(high3, 2)
        if bias == "bearish" and high3 < low1:
            gap = low1 - high3
            if gap > prices[-1] * 0.0015:
                return True, round(high3, 2), round(low1, 2)
    return False, 0, 0


def _detect_fvg_ohlc(candles, bias):
    for i in range(len(candles)-3, len(candles)):
        if i < 2: continue
        c1, c2, c3 = candles[i-2], candles[i-1], candles[i]
        if bias == "bullish":
            if c3["l"] > c1["h"]:
                gap = c3["l"] - c1["h"]
                if gap > c3["c"] * 0.001:
                    return True, round(c1["h"], 2), round(c3["l"], 2)
        if bias == "bearish":
            if c3["h"] < c1["l"]:
                gap = c1["l"] - c3["h"]
                if gap > c3["c"] * 0.001:
                    return True, round(c3["h"], 2), round(c1["l"], 2)
    return False, 0, 0


# ── TECHNICAL INDICATORS ──────────────────────────────────
def get_ema_bias(prices):
    if len(prices) < 22: return None
    def ema(arr, n):
        k = 2/(n+1); e = sum(arr[:n])/n
        for v in arr[n:]: e = v*k+e*(1-k)
        return e
    e9, e21 = ema(prices, 9), ema(prices, 21)
    if   e9 > e21 * 1.001: return "bullish"
    elif e9 < e21 * 0.999: return "bearish"
    return None


def get_rsi_scalp(prices, n=14):
    if len(prices) < n+1: return 50
    gains = losses = 0
    for i in range(len(prices)-n, len(prices)):
        d = prices[i] - prices[i-1]
        if d > 0: gains += d
        else: losses -= d
    ag, al = gains/n, losses/n
    return 100 - 100/(1+ag/al) if al > 0 else 100


def detect_chop(prices, n=20):
    if len(prices) < n: return True
    recent = prices[-n:]
    atr = sum(abs(recent[i]-recent[i-1]) for i in range(1,len(recent))) / (len(recent)-1)
    rng = max(recent) - min(recent)
    if rng < atr * 5: return True
    highs = _swing_highs(recent, n=2)
    lows  = _swing_lows(recent, n=2)
    if len(highs) < 2 or len(lows) < 2: return True
    return False


def detect_market_regime(prices, n=30):
    if len(prices) < n:
        return {"regime": "UNKNOWN", "strength": 0, "should_trade": False, "score_mult": 0}
    recent = prices[-n:]
    atr = sum(abs(recent[i]-recent[i-1]) for i in range(1,len(recent))) / (len(recent)-1)
    atr_pct = atr / (recent[-1] + 1e-9) * 100
    rng = max(recent) - min(recent)
    rng_atr_ratio = rng / (atr + 1e-9)
    direction     = recent[-1] - recent[0]
    directionality = abs(direction) / (rng + 1e-9)
    def _ema(arr, n):
        if len(arr) < n: return arr[-1] if arr else 0
        k = 2/(n+1); e = sum(arr[:n])/n
        for v in arr[n:]: e = v*k+e*(1-k)
        return e
    if len(prices) >= 50:
        ema20_now = _ema(prices[-30:], 20)
        ema20_old = _ema(prices[-50:-20], 20)
        ema_slope_pct = abs(ema20_now - ema20_old) / (ema20_old + 1e-9) * 100
    else:
        ema_slope_pct = 0
    if atr_pct < 0.03:
        return {"regime": "QUIET", "strength": 0, "should_trade": False, "score_mult": 0,
                "atr_pct": atr_pct, "reason": "volatilidad muy baja (spread > edge)"}
    if atr_pct > 1.5:
        return {"regime": "VOLATILE", "strength": 1.0, "should_trade": False, "score_mult": 0,
                "atr_pct": atr_pct, "reason": "volatilidad extrema — probable news spike"}
    if directionality > 0.5 and rng_atr_ratio > 5 and ema_slope_pct > 0.15:
        return {"regime": "TRENDING_STRONG", "strength": min(1.0, ema_slope_pct * 3),
                "should_trade": True, "score_mult": 1.2,
                "atr_pct": atr_pct, "directionality": directionality,
                "reason": f"tendencia fuerte ({ema_slope_pct:.2f}% slope)"}
    if directionality > 0.3 and rng_atr_ratio > 4:
        return {"regime": "TRENDING_WEAK", "strength": 0.5,
                "should_trade": True, "score_mult": 1.0,
                "atr_pct": atr_pct, "reason": "tendencia moderada"}
    if directionality < 0.25 and rng_atr_ratio > 3:
        return {"regime": "RANGING", "strength": 0.3,
                "should_trade": True, "score_mult": 0.8,
                "atr_pct": atr_pct, "reason": "mercado lateral — solo reversals extremos"}
    return {"regime": "MIXED", "strength": 0.2, "should_trade": True, "score_mult": 0.9,
            "atr_pct": atr_pct, "reason": "régimen indefinido"}


def detect_confirmation_candle(prices, bias):
    if len(prices) < 4: return False
    last4 = prices[-4:]
    body  = abs(last4[-1] - last4[-2])
    rng   = max(last4) - min(last4) + 1e-9
    momentum = body / rng
    if bias == "bullish" and last4[-1] > last4[-2] and momentum > 0.4: return True
    if bias == "bearish" and last4[-1] < last4[-2] and momentum > 0.4: return True
    return False


def get_ml_filter(prices, direction=None):
    if not _ai.trained or len(prices) < 35:
        return "MED", 0.5
    prob = _ai.predict_proba(prices)
    if prob is None: return "MED", 0.5
    effective_prob = (1.0 - prob) if direction == "VENDER" else prob
    if   effective_prob >= 0.65: return "HIGH", round(prob, 2)
    elif effective_prob >= 0.45: return "MED",  round(prob, 2)
    else:                        return "LOW",  round(prob, 2)


# ── PRE-SIGNAL ────────────────────────────────────────────
def detect_pre_signal(prices, candles=None):
    if len(prices) < 22: return _pre_signal
    bias = get_ema_bias(prices)
    if not bias:
        _pre_signal["state"] = "IDLE"
        return _pre_signal
    _pre_signal["bias"]  = bias
    price = prices[-1]
    if bias == "bullish":
        lows = _swing_lows(prices[:-3], n=2)
        if lows:
            nearest_low = lows[-1][1]
            dist_pct = (price - nearest_low) / (price + 1e-9) * 100
            if 0 < dist_pct < 0.15:
                _pre_signal["state"] = "LIQUIDITY_NEAR"
                _pre_signal["sweep_level"] = nearest_low
                _pre_signal["ts"] = time.time()
    elif bias == "bearish":
        highs = _swing_highs(prices[:-3], n=2)
        if highs:
            nearest_high = highs[-1][1]
            dist_pct = (nearest_high - price) / (price + 1e-9) * 100
            if 0 < dist_pct < 0.15:
                _pre_signal["state"] = "LIQUIDITY_NEAR"
                _pre_signal["sweep_level"] = nearest_high
                _pre_signal["ts"] = time.time()
    swept, lvl, rej = detect_sweep_and_rejection(prices, bias, candles=candles)
    if swept:
        _pre_signal["state"] = "SWEEP_DETECTED"
        _pre_signal["sweep_level"] = lvl
    fvg, flo, fhi = detect_fvg_scalp(prices, bias, candles=candles)
    if fvg:
        _pre_signal["fvg_zone"] = (flo, fhi)
        if swept:
            _pre_signal["state"] = "READY"
    return _pre_signal


# ── ICT PRICE BUFFERS ─────────────────────────────────────
def update_ict_prices(price, candle=None):
    with _data_lock:
        _scalp_prices_1m.append(price)
        if len(_scalp_prices_1m) > 500:
            del _scalp_prices_1m[:-500]
        now = time.time()
        if now - _ohlc_timers["last_5m"] >= 300:
            _scalp_prices_5m.append(price)
            if len(_scalp_prices_5m) > 300:
                del _scalp_prices_5m[:-300]
            _ohlc_timers["last_5m"] = now
        if now - _ohlc_timers["last_15m"] >= 900:
            _scalp_prices_15m.append(price)
            if len(_scalp_prices_15m) > 200:
                del _scalp_prices_15m[:-200]
            _ohlc_timers["last_15m"] = now


def update_ohlc_candles(candles_list, tf="5m"):
    with _data_lock:
        target = _ohlc_candles_5m if tf == "5m" else _ohlc_candles_15m
        target.clear()
        for c in candles_list:
            target.append({
                "o": c.get("open", c["close"]),
                "h": c.get("high", c["close"]),
                "l": c.get("low",  c["close"]),
                "c": c["close"],
            })


# backward compat aliases
_ict_prices_5m  = _scalp_prices_5m
_ict_prices_15m = _scalp_prices_15m


# ── ICT ORDER BLOCKS ─────────────────────────────────────
def detect_order_block(candles, bias, atr):
    """Último candle opuesto antes de un displacement. Zona institucional de entrada."""
    if len(candles) < 15 or atr <= 0: return False, 0, 0, 0
    if bias == "bullish":
        for i in range(len(candles)-4, max(2, len(candles)-30), -1):
            c = candles[i]
            if c["c"] >= c["o"]: continue
            after = candles[i+1:min(i+6, len(candles))]
            if len(after) < 2: continue
            up_move = max(a["h"] for a in after) - c["l"]
            if up_move < atr * 1.5: continue
            if any(x["c"] < c["l"] for x in candles[i+1:]): continue
            return True, round(c["h"], 2), round(c["l"], 2), round(min(1.0, up_move/(atr*4)), 2)
    if bias == "bearish":
        for i in range(len(candles)-4, max(2, len(candles)-30), -1):
            c = candles[i]
            if c["c"] <= c["o"]: continue
            after = candles[i+1:min(i+6, len(candles))]
            if len(after) < 2: continue
            dn_move = c["h"] - min(a["l"] for a in after)
            if dn_move < atr * 1.5: continue
            if any(x["c"] > c["h"] for x in candles[i+1:]): continue
            return True, round(c["h"], 2), round(c["l"], 2), round(min(1.0, dn_move/(atr*4)), 2)
    return False, 0, 0, 0


# ── OPTIMAL TRADE ENTRY — Fibonacci 61.8–78.6% ───────────
def detect_ote(candles, bias, current_price):
    """Zona óptima de entrada: retroceso del 61.8% al 78.6% del impulso."""
    if len(candles) < 15: return False, 0, 0
    prices = [c["c"] for c in candles[-30:]]
    lows   = _swing_lows(prices, n=3)
    highs  = _swing_highs(prices, n=3)
    if not lows or not highs: return False, 0, 0
    sl, sh = lows[-1][1], highs[-1][1]
    if sh <= sl or (sh - sl) < prices[-1] * 0.002: return False, 0, 0
    imp = sh - sl
    if bias == "bullish":
        lo, hi = sh - imp * 0.786, sh - imp * 0.618
    else:
        lo, hi = sl + imp * 0.618, sl + imp * 0.786
    return lo <= current_price <= hi, round(lo, 2), round(hi, 2)


# ── DISPLACEMENT — huella institucional ──────────────────
def detect_displacement(candles, bias, atr):
    """Vela de impulso fuerte (body > 1.5 ATR, wick ratio > 60%)."""
    if len(candles) < 5 or atr <= 0: return False
    for c in candles[-6:]:
        body = abs(c["c"] - c["o"])
        if body < atr * 1.5: continue
        if body / (c["h"] - c["l"] + 1e-9) < 0.6: continue
        if bias == "bullish" and c["c"] > c["o"]: return True
        if bias == "bearish" and c["c"] < c["o"]: return True
    return False


# ── EQUAL HIGHS / EQUAL LOWS — pools de liquidez ─────────
def detect_equal_levels(candles, bias, tol=0.0004):
    """EQH / EQL: niveles casi iguales que acumulan liquidez."""
    if len(candles) < 10: return False, 0
    if bias == "bullish":
        lows = sorted([c["l"] for c in candles[-20:]])
        for i in range(len(lows)-1):
            if abs(lows[i]-lows[i+1])/(lows[i]+1e-9) < tol:
                return True, round((lows[i]+lows[i+1])/2, 2)
    if bias == "bearish":
        highs = sorted([c["h"] for c in candles[-20:]], reverse=True)
        for i in range(len(highs)-1):
            if abs(highs[i]-highs[i+1])/(highs[i]+1e-9) < tol:
                return True, round((highs[i]+highs[i+1])/2, 2)
    return False, 0


# ── ICT KILLZONES ─────────────────────────────────────────
def get_killzone():
    """Ventanas de mayor probabilidad según ICT."""
    t = _dt.datetime.utcnow()
    m = t.hour * 60 + t.minute
    if 480 <= m <= 570: return True, "London Open 🇬🇧", 20
    if 780 <= m <= 840: return True, "New York Open 🇺🇸", 18
    if 600 <= m <= 660: return True, "London Close",    10
    if 900 <= m <= 960: return True, "NY PM",           10
    return False, "", 0


# ── INDUCEMENT ────────────────────────────────────────────
def detect_inducement(candles, bias):
    """Swing menor barrido antes del movimiento real — confirma manipulación."""
    if len(candles) < 15: return False
    prices = [c["c"] for c in candles]
    if bias == "bullish":
        minor = _swing_highs(prices[-15:-5], n=1)
        if minor and max(prices[-5:]) > minor[-1][1]: return True
    if bias == "bearish":
        minor = _swing_lows(prices[-15:-5], n=1)
        if minor and min(prices[-5:]) < minor[-1][1]: return True
    return False


# ── SMART TP — siguiente pool de liquidez ─────────────────
def find_liquidity_target(candles, bias, entry, atr_15m):
    """Apunta el TP al siguiente swing H/L o EQH/EQL en lugar de múltiplo fijo."""
    prices = [c["c"] for c in candles]
    if bias == "bullish":
        candidates = [h for _, h in _swing_highs(prices, n=2) if h > entry + atr_15m * 0.8]
        highs_raw  = [c["h"] for c in candles]
        for i in range(len(highs_raw)):
            for j in range(i+3, min(i+15, len(highs_raw))):
                if abs(highs_raw[i]-highs_raw[j])/(highs_raw[i]+1e-9) < 0.0005:
                    lv = (highs_raw[i]+highs_raw[j])/2
                    if lv > entry + atr_15m: candidates.append(lv)
        if candidates: return round(min(candidates), 2), "SWING HIGH"
        return round(entry + atr_15m * 3.0, 2), "ATR 3x"
    if bias == "bearish":
        candidates = [l for _, l in _swing_lows(prices, n=2) if l < entry - atr_15m * 0.8]
        lows_raw   = [c["l"] for c in candles]
        for i in range(len(lows_raw)):
            for j in range(i+3, min(i+15, len(lows_raw))):
                if abs(lows_raw[i]-lows_raw[j])/(lows_raw[i]+1e-9) < 0.0005:
                    lv = (lows_raw[i]+lows_raw[j])/2
                    if lv < entry - atr_15m: candidates.append(lv)
        if candidates: return round(max(candidates), 2), "SWING LOW"
        return round(entry - atr_15m * 3.0, 2), "ATR 3x"
    return round(entry + atr_15m * 2.5, 2), "ATR 2.5x"


# ── MAIN SIGNAL ENGINE ────────────────────────────────────
def run_ict_engine(current_price, atr):
    """AURUM TRADE v8.0 — ICT completo: OB, OTE, Displacement, Killzones, Smart TP."""
    if len(_ohlc_candles_5m) < 20:  return None
    if len(_ohlc_candles_15m) < 20: return None

    now_g = time.time()
    if now_g - _state._last_any_signal_ts < SIGNAL_COOLDOWN_SEC: return None

    ohlc5  = list(_ohlc_candles_5m)
    ohlc15 = list(_ohlc_candles_15m)
    p5  = [c["c"] for c in ohlc5]
    if current_price and abs(current_price - p5[-1]) > 0.01:
        p5 = p5 + [current_price]
    p15 = [c["c"] for c in ohlc15]

    if not is_trading_session(): return None

    regime_info = detect_market_regime(p15)
    if not regime_info["should_trade"]: return None

    atr_15m = sum(abs(ohlc15[i]["c"] - ohlc15[i-1]["c"])
                  for i in range(len(ohlc15)-14, len(ohlc15))) / 14
    if atr_15m <= 0: atr_15m = atr * 3

    bias_15m = get_ema_bias(p15)
    if not bias_15m: return None

    bias_5m = get_ema_bias(p5)
    if bias_5m and bias_5m != bias_15m: return None

    with _data_lock:
        ohlc4h = list(_live_cache.get("ohlc_4h", []))
        ohlc1h = list(_live_cache.get("ohlc_1h", []))
    mtf_4h = get_ema_bias([c["close"] for c in ohlc4h]) if len(ohlc4h) >= 21 else None
    mtf_1h = get_ema_bias([c["close"] for c in ohlc1h]) if len(ohlc1h) >= 21 else None

    bias      = bias_15m
    direction = "COMPRAR" if bias == "bullish" else "VENDER"
    score     = 0
    details   = {"bias": bias, "direction": direction, "tf": "15M", "ver": "v8.0"}

    # 0. KILLZONE — ventana de mayor probabilidad ICT
    in_kz, kz_name, kz_pts = get_killzone()
    if in_kz:
        score += kz_pts
        details["killzone"] = f"✓ {kz_name}"

    # 1. HTF confluence
    htf_count = 0
    if mtf_4h == bias: htf_count += 1; score += 12; details["htf_4h"] = "✓ 4H"
    if mtf_1h == bias: htf_count += 1; score += 10; details["htf_1h"] = "✓ 1H"
    if htf_count == 2: score += 8; details["htf_full"] = "✓ HTF completo"

    detect_pre_signal(p15, candles=ohlc15)

    # 2. DISPLACEMENT — obligatorio para confirmar presencia institucional
    displaced = detect_displacement(ohlc15, bias, atr_15m)
    if not displaced: return None
    score += 15
    details["displacement"] = "✓ Displacement"

    # 3. LIQUIDITY SWEEP en 15m — OBLIGATORIO
    swept, sweep_level, rejection = detect_sweep_and_rejection(p15, bias, candles=ohlc15)
    if not swept: return None
    if not _is_real_sweep(p15, sweep_level, bias, candles=ohlc15, atr=atr_15m): return None
    score += 35
    details["sweep"] = f"Sweep ${sweep_level}"

    sweep_type, sweep_conf = _classify_sweep_type(p15, bias, candles=ohlc15)
    if sweep_type == "CONTINUATION": return None
    elif sweep_type == "REVERSA":
        score += int(sweep_conf * 12)
        details["sweep_q"] = f"Reversa {sweep_conf:.0%}"
    elif sweep_type == "NEUTRAL" and sweep_conf < 0.5: return None

    # 4. BOS / CHOCH en 15m — OBLIGATORIO
    bos, bos_type = detect_bos_scalp(p15, bias)
    if not bos: return None
    score += 30
    details["bos"] = bos_type

    # 5. INDUCEMENT — confirma la narrativa de manipulación
    if detect_inducement(ohlc15, bias):
        score += 12
        details["inducement"] = "✓ Inducement"

    # 6. ORDER BLOCK — zona de entrada institucional
    ob_found, ob_hi, ob_lo, ob_str = detect_order_block(ohlc15, bias, atr_15m)
    if ob_found:
        score += 20
        details["ob"] = f"OB ${ob_lo}–${ob_hi}"
        if ob_lo <= current_price <= ob_hi:
            score += 20
            details["ob_entry"] = "✓ En Order Block"

    # 7. OTE — Fibonacci 61.8–78.6%
    in_ote, ote_lo, ote_hi = detect_ote(ohlc15, bias, current_price)
    if in_ote:
        score += 15
        details["ote"] = f"✓ OTE ${ote_lo}–${ote_hi}"

    # 8. FVG en 15m
    fvg, fvg_lo, fvg_hi = detect_fvg_scalp(p15, bias, candles=ohlc15)
    if fvg:
        score += 12
        details["fvg"] = f"FVG ${fvg_lo}–${fvg_hi}"
        if fvg_lo <= current_price <= fvg_hi:
            score += 8
            details["fvg_entry"] = "✓ En FVG"

    # 9. EQH / EQL — target de liquidez visible
    eq_found, eq_level = detect_equal_levels(ohlc15, bias)
    if eq_found:
        score += 8
        details["eq"] = f"{'EQH' if bias=='bearish' else 'EQL'} ${eq_level}"

    # 10. EMA 15m alineada
    if bias == get_ema_bias(p15):
        score += 8
        details["ema"] = "EMA 9/21 15M"

    # Filtro intermedio — no continuar si la estructura es débil
    if score < SCORE_NORMAL - 20: return None

    # 11. CONFIRMACIÓN en 5m — OBLIGATORIA
    if not detect_confirmation_candle(p5, bias): return None
    score += 10
    details["candle"] = "✅ Confirmación 5M"

    # 12. RSI 15m
    rsi = get_rsi_scalp(p15)
    details["rsi"] = round(rsi, 1)
    if bias == "bullish":
        if rsi > 78: return None
        elif rsi < 40: score += 8
        elif rsi > 65: score -= 5
    if bias == "bearish":
        if rsi < 22: return None
        elif rsi > 60: score += 8
        elif rsi < 35: score -= 5

    # 13. ML FILTER
    ml_label, ml_prob = get_ml_filter(p5, direction=direction)
    if ml_label == "LOW": return None
    if ml_label == "HIGH":
        score += 15
        details["ml"] = f"✓ IA ({ml_prob:.0%})"
    else:
        details["ml"] = f"IA neutral ({ml_prob:.0%})"

    # 14. MACRO — DXY + US10Y
    dxy_check    = check_dxy_correlation(direction)
    yields_check = check_yields_for_gold(direction)
    details["dxy"]   = dxy_check["note"]
    details["us10y"] = yields_check["note"]
    if dxy_check["aligned"] is False and yields_check["aligned"] is False: return None
    score += dxy_check["bonus"] + yields_check["bonus"]
    if dxy_check.get("aligned") and yields_check.get("aligned"):
        score += 5; details["macro"] = "✓ DXY+Yields"

    score = int(score * regime_info["score_mult"])
    details["regime"] = regime_info["regime"]
    if score < SCORE_NORMAL: return None

    # COOLDOWN por clave
    sig_key = f"{direction}_{round(current_price/10)*10}"
    now = time.time()
    if sig_key == _last_scalp_signal["key"] and now - _last_scalp_signal["time"] < 3600:
        return None
    _last_scalp_signal["key"]  = sig_key
    _last_scalp_signal["time"] = now
    _state._last_any_signal_ts = now

    is_buy    = direction == "COMPRAR"
    is_sniper = score >= SCORE_SNIPER

    # TP inteligente — siguiente pool de liquidez
    tp_smart, tp_label = find_liquidity_target(ohlc15, bias, current_price, atr_15m)
    sl = current_price - atr_15m if is_buy else current_price + atr_15m
    rr_real = abs(tp_smart - current_price) / (abs(sl - current_price) + 1e-9)
    if rr_real < 2.0:
        tp_mult  = 3.5 if is_sniper else 2.5
        tp_smart = current_price + atr_15m * tp_mult if is_buy else current_price - atr_15m * tp_mult
        rr_real  = tp_mult
        tp_label = f"ATR {tp_mult}x"
    rr = round(rr_real, 1)

    session  = get_session_name()
    mode_tag = "SNIPER 🎯" if is_sniper else "TRADE"

    reasons = []
    if details.get("killzone"):   reasons.append(f"✔ {kz_name}")
    if details.get("htf_full"):   reasons.append("✔ 4H + 1H confluencia")
    elif details.get("htf_4h"):   reasons.append("✔ 4H alineado")
    reasons.append("✔ Displacement institucional")
    reasons.append(f"✔ Liquidity Sweep ({sweep_type})")
    reasons.append(f"✔ {bos_type}")
    if details.get("inducement"): reasons.append("✔ Inducement")
    if details.get("ob_entry"):   reasons.append(f"✔ Order Block ${ob_lo}–${ob_hi}")
    elif details.get("ob"):       reasons.append(f"✔ OB detectado")
    if details.get("ote"):        reasons.append(details["ote"])
    if details.get("fvg_entry"):  reasons.append(f"✔ FVG entry ${fvg_lo}–${fvg_hi}")
    elif details.get("fvg"):      reasons.append("✔ FVG presente")
    if details.get("ml"):         reasons.append(f"✔ {details['ml']}")
    if details.get("macro"):      reasons.append("✔ DXY + Yields")

    header = "🥇 *AURUM — SNIPER*" if is_sniper else "📊 *AURUM TRADE*"
    dirstr = "BUY" if is_buy else "SELL"
    msg = (
        header + "\n\nPAIR: XAUUSD\nDIRECTION: " + dirstr + "\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "ENTRY:  $" + f"{current_price:.2f}" + "\n"
        "SL:     $" + f"{sl:.2f}" + "\n"
        "TP:     $" + f"{tp_smart:.2f}" + "  (" + tp_label + ")\n\n"
        "RR:         1:" + str(rr) + "\n"
        "CONFIDENCE: " + str(score) + "%\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "REASON:\n" + "\n".join(reasons) + "\n\n"
        "SESSION: " + session + "\nMODE: " + mode_tag
    )
    return {"direction": direction, "score": score, "msg": msg,
            "tp": tp_smart, "sl": sl, "rr": rr, "details": details, "session": session}
