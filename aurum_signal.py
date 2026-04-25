"""aurum_signal.py — ICT signal engine, session helpers, signal detection."""
import math, time
import datetime as _dt

from aurum_state import (
    _data_lock, _control_lock,
    _scalp_prices_1m, _scalp_prices_5m, _scalp_prices_15m,
    _ohlc_candles_5m, _ohlc_candles_15m,
    _ohlc_timers, _last_scalp_signal, _pre_signal,
    _current_control, _live_cache,
    SCORE_SNIPER, SCORE_NORMAL, SCORE_EARLY,
    is_data_stale,
)
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
    if penetration < atr * 0.3:
        return False
    if len(prices) >= 3:
        body = abs(prices[-1] - prices[-2])
        rng  = max(prices[-3:]) - min(prices[-3:]) + 1e-9
        if body / rng < 0.35:
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
    if len(prices) < 15: return False, "none"
    p = prices
    if bias == "bullish":
        highs = _swing_highs(p[:-2], n=2)
        if highs and p[-1] > highs[-1][1]:
            return True, "BOS alcista"
        if len(p) >= 10 and p[-1] > p[-5] and p[-5] < p[-10]:
            return True, "CHOCH"
    if bias == "bearish":
        lows = _swing_lows(p[:-2], n=2)
        if lows and p[-1] < lows[-1][1]:
            return True, "BOS bajista"
        if len(p) >= 10 and p[-1] < p[-5] and p[-5] > p[-10]:
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
            if gap > prices[-1] * 0.0008:
                return True, round(low1, 2), round(high3, 2)
        if bias == "bearish" and high3 < low1:
            gap = low1 - high3
            if gap > prices[-1] * 0.0008:
                return True, round(high3, 2), round(low1, 2)
    return False, 0, 0


def _detect_fvg_ohlc(candles, bias):
    for i in range(len(candles)-3, len(candles)):
        if i < 2: continue
        c1, c2, c3 = candles[i-2], candles[i-1], candles[i]
        if bias == "bullish":
            if c3["l"] > c1["h"]:
                gap = c3["l"] - c1["h"]
                if gap > c3["c"] * 0.0005:
                    return True, round(c1["h"], 2), round(c3["l"], 2)
        if bias == "bearish":
            if c3["h"] < c1["l"]:
                gap = c1["l"] - c3["h"]
                if gap > c3["c"] * 0.0005:
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
    if   e9 > e21 * 1.0005: return "bullish"
    elif e9 < e21 * 0.9995: return "bearish"
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


# ── MAIN SIGNAL ENGINE ────────────────────────────────────
def run_ict_engine(current_price, atr):
    """AURUM SCALP v6.2 — Motor principal ICT."""
    if len(_ohlc_candles_5m) < 20:  return None
    if len(_ohlc_candles_15m) < 15: return None

    ohlc5  = _ohlc_candles_5m
    ohlc15 = _ohlc_candles_15m
    p5  = [c["c"] for c in ohlc5]
    if current_price and abs(current_price - p5[-1]) > 0.01:
        p5 = p5 + [current_price]
    p15 = [c["c"] for c in ohlc15]

    if not is_trading_session(): return None

    regime_info = detect_market_regime(p5)
    if not regime_info["should_trade"]: return None

    # Multi-timeframe confluence
    mtf_bias_4h = mtf_bias_1h = None
    with _data_lock:
        ohlc4h = list(_live_cache.get("ohlc_4h", []))
        ohlc1h = list(_live_cache.get("ohlc_1h", []))
    if ohlc4h and len(ohlc4h) >= 21:
        mtf_bias_4h = get_ema_bias([c["close"] for c in ohlc4h])
    if ohlc1h and len(ohlc1h) >= 21:
        mtf_bias_1h = get_ema_bias([c["close"] for c in ohlc1h])
    mtf_bias_15m = get_ema_bias(p15)

    bias_5m = get_ema_bias(p5)
    if not bias_5m: return None

    htf_biases = [b for b in [mtf_bias_4h, mtf_bias_1h, mtf_bias_15m] if b]
    if len(htf_biases) >= 2:
        aligned = sum(1 for b in htf_biases if b == bias_5m)
        if aligned < 2: return None

    bias = mtf_bias_15m or get_ema_bias(p15)
    if not bias: return None

    detect_pre_signal(p5, candles=ohlc5)

    direction = "COMPRAR" if bias == "bullish" else "VENDER"
    score   = 0
    details = {"bias": bias, "direction": direction}

    # 1. LIQUIDITY SWEEP — OBLIGATORIO
    swept, sweep_level, rejection = detect_sweep_and_rejection(p5, bias, candles=ohlc5)
    if not swept: return None
    if not _is_real_sweep(p5, sweep_level, bias, candles=ohlc5, atr=atr):
        details["filtered"] = "micro-sweep filtrado"
        return None
    score += 40
    details["sweep"] = f"Sweep ${sweep_level} | rej={rejection:.1f}"

    sweep_type, sweep_conf = _classify_sweep_type(p5, bias, candles=ohlc5)
    details["sweep_type"] = sweep_type
    details["sweep_conf"] = sweep_conf
    if sweep_type == "CONTINUATION":
        details["filtered"] = "⚠ Sweep continuación (trampa) bloqueado"
        return None
    elif sweep_type == "REVERSA":
        score += int(sweep_conf * 15)
        details["sweep_quality"] = f"✓ Reversa ({sweep_conf:.0%})"
    elif sweep_type == "NEUTRAL" and sweep_conf < 0.5:
        details["filtered"] = "sweep tipo neutral sin confianza"
        return None

    # 2. BOS / CHOCH — OBLIGATORIO
    bos, bos_type = detect_bos_scalp(p5, bias)
    if not bos:
        fvg_early, fvg_lo, fvg_hi = detect_fvg_scalp(p5, bias, candles=ohlc5)
        if fvg_early and sweep_type == "REVERSA" and sweep_conf >= 0.6:
            if fvg_lo <= current_price <= fvg_hi:
                score += 15 + 5
                details["entry_type"] = "⚡ EARLY FVG (sin BOS)"
                details["fvg"] = f"FVG ${fvg_lo}–${fvg_hi}"
                details["fvg_entry"] = "✓ Precio en FVG"
                if score < SCORE_EARLY: return None
                ml_label, ml_prob = get_ml_filter(p5)
                if ml_label == "LOW": return None
                if ml_label == "HIGH":
                    score += 10
                    details["ml"] = f"ML HIGH ({ml_prob:.0%})"
                details["score"] = score
                is_buy = direction == "COMPRAR"
                tp = current_price + atr * 1.2 if is_buy else current_price - atr * 1.2
                sl = current_price - atr * 1.0 if is_buy else current_price + atr * 1.0
                rr      = 1.2
                session = get_session_name()
                sig_key = f"EARLY_{direction}_{round(current_price/5)*5}"
                now = time.time()
                if sig_key == _last_scalp_signal["key"] and now - _last_scalp_signal["time"] < 60:
                    return None
                _last_scalp_signal["key"]  = sig_key
                _last_scalp_signal["time"] = now
                reasons = [f"✔ Sweep REVERSA ({sweep_conf:.0%})", "✔ FVG entry (early)", f"✔ {details.get('ml','')}"]
                dirstr  = "BUY" if is_buy else "SELL"
                msg = (
                    "⚡ *AURUM EARLY ENTRY*\n\n"
                    "PAIR: XAUUSD\nDIRECTION: " + dirstr + "\n\n"
                    "━━━━━━━━━━━━━━━━━━\n"
                    "ENTRY:  $" + f"{current_price:.2f}" + "\n"
                    "SL:     $" + f"{sl:.2f}" + "\n"
                    "TP:     $" + f"{tp:.2f}" + "\n\n"
                    "RR:         1:" + str(rr) + "\n"
                    "CONFIDENCE: " + str(score) + "%\n"
                    "━━━━━━━━━━━━━━━━━━\n"
                    "REASON:\n" + "\n".join(reasons) + "\n\n"
                    "SESSION: " + session + "\nMODE: EARLY ⚡"
                )
                return {"direction": direction, "score": score, "msg": msg,
                        "tp": tp, "sl": sl, "rr": rr, "details": details, "session": session}
        return None
    score += 25
    details["bos"] = bos_type
    details["entry_type"] = "STANDARD"

    # 3. FVG
    fvg, fvg_lo, fvg_hi = detect_fvg_scalp(p5, bias, candles=ohlc5)
    if fvg:
        score += 15
        details["fvg"] = f"FVG ${fvg_lo}–${fvg_hi}"
        if fvg_lo <= current_price <= fvg_hi:
            score += 5
            details["fvg_entry"] = "✓ Precio en FVG"

    # 4. EMA ALIGNMENT
    if bias == get_ema_bias(p5):
        score += 10
        details["ema"] = "EMA 9/21 alineada"

    # 5. ML FILTER
    ml_label, ml_prob = get_ml_filter(p5, direction=direction)
    if ml_label == "LOW":
        details["ml_contradiction"] = f"IA contradice — prob={ml_prob:.0%} para {direction}"
        return None
    if ml_label == "HIGH":
        score += 15
        details["ml"] = f"✓ IA confirma {direction} ({ml_prob:.0%})"
    else:
        details["ml"] = f"IA neutral ({ml_prob:.0%})"
        if score < SCORE_SNIPER - 10: return None

    # DXY CORRELATION
    dxy_check = check_dxy_correlation(direction)
    details["dxy"] = dxy_check["note"]
    if dxy_check["aligned"] is False and score + dxy_check["bonus"] < SCORE_SNIPER:
        details["filtered"] = f"conflicto DXY — {dxy_check['note']}"
        return None
    score += dxy_check["bonus"]

    # US10Y YIELDS
    yields_check = check_yields_for_gold(direction)
    details["us10y"] = yields_check["note"]
    if yields_check["aligned"] is False and score + yields_check["bonus"] < SCORE_SNIPER:
        details["filtered"] = f"conflicto yields — {yields_check['note']}"
        return None
    score += yields_check["bonus"]
    if dxy_check.get("aligned") and yields_check.get("aligned"):
        score += 5
        details["macro_confluence"] = "✓ DXY + US10Y confirman"

    details["score"] = score
    if score < SCORE_NORMAL: return None

    score = int(score * regime_info["score_mult"])
    details["regime"]      = regime_info["regime"]
    details["regime_mult"] = regime_info["score_mult"]
    if score < SCORE_NORMAL: return None

    # RSI SOFT FILTER
    rsi = get_rsi_scalp(p5)
    details["rsi"] = round(rsi, 1)
    if bias == "bullish":
        if   rsi > 75: score -= 15
        elif rsi > 65: score -= 5
        elif rsi < 40: score += 5
    if bias == "bearish":
        if   rsi < 25: score -= 15
        elif rsi < 35: score -= 5
        elif rsi > 60: score += 5
    if score < SCORE_NORMAL: return None

    # CANDLE CONFIRMATION
    if len(p5) >= 4:
        confirmed = detect_confirmation_candle(p5, bias)
        details["candle"] = "✅ Confirmación 5M" if confirmed else "⏳ Sin confirmación"
        if not confirmed and score < 95:
            details["filtered"] = "sin confirmación de vela 5M"
            return None

    # COOLDOWN
    sig_key = f"{direction}_{round(current_price/5)*5}"
    now = time.time()
    if sig_key == _last_scalp_signal["key"] and now - _last_scalp_signal["time"] < 60:
        return None
    _last_scalp_signal["key"]  = sig_key
    _last_scalp_signal["time"] = now

    is_buy = direction == "COMPRAR"
    tp_mult, sl_mult = (2.5, 1.0) if score >= SCORE_SNIPER else (1.5, 1.0)
    tp  = current_price + atr * tp_mult if is_buy else current_price - atr * tp_mult
    sl  = current_price - atr * sl_mult if is_buy else current_price + atr * sl_mult
    rr  = round(tp_mult / sl_mult, 1)
    session  = get_session_name()
    mode_tag = "SNIPER 🎯" if score >= SCORE_SNIPER else "NORMAL"

    reasons = []
    if details.get("sweep"): reasons.append("✔ Liquidity Sweep")
    if details.get("bos"):   reasons.append(f"✔ {details['bos']}")
    if details.get("fvg"):   reasons.append("✔ FVG entry")
    if details.get("ema"):   reasons.append("✔ EMA alignment")
    if details.get("ml"):    reasons.append(f"✔ {details['ml']}")

    header = "🥇 *AURUM SCALP — SNIPER*" if score >= SCORE_SNIPER else "⚡ *AURUM SCALP SIGNAL*"
    dirstr = "BUY" if is_buy else "SELL"
    msg = (
        header + "\n\nPAIR: XAUUSD\nDIRECTION: " + dirstr + "\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "ENTRY:  $" + f"{current_price:.2f}" + "\n"
        "SL:     $" + f"{sl:.2f}" + "\n"
        "TP:     $" + f"{tp:.2f}" + "\n\n"
        "RR:         1:" + str(rr) + "\n"
        "CONFIDENCE: " + str(score) + "%\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "REASON:\n" + "\n".join(reasons) + "\n\n"
        "SESSION: " + session + "\nMODE: " + mode_tag
    )

    return {"direction": direction, "score": score, "msg": msg,
            "tp": tp, "sl": sl, "rr": rr, "details": details, "session": session}
