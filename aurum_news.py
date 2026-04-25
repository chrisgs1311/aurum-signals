"""aurum_news.py — News filter, DXY correlation, US10Y yields."""
import time
import datetime as _dt

from aurum_state import (
    _news_lock, _news_cache, _news_alerted,
    PAUSE_MINUTES_BEFORE, PAUSE_MINUTES_AFTER, ALERT_MINUTES_BEFORE,
    HIGH_IMPACT_KEYWORDS,
    TWELVE_API_KEY,
    _dxy_cache, _us10y_cache,
    _fetch_with_retry,
)


def _fetch_forex_factory_calendar():
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        data = _fetch_with_retry(url, timeout=8, retries=2)
        if not data: return []
        events = []
        for ev in data:
            title    = ev.get("title", "")
            impact   = ev.get("impact", "")
            currency = ev.get("country", "")
            date_str = ev.get("date", "")
            if currency != "USD": continue
            if impact != "High": continue
            if not any(kw.lower() in title.lower() for kw in HIGH_IMPACT_KEYWORDS):
                continue
            try:
                ev_dt  = _dt.datetime.fromisoformat(date_str)
                ev_utc = ev_dt.astimezone(_dt.timezone.utc).replace(tzinfo=None)
                events.append({"name": title, "datetime_utc": ev_utc, "impact": impact})
            except:
                continue
        print(f"  ✓ Forex Factory: {len(events)} eventos high-impact USD esta semana")
        return events
    except Exception as e:
        print(f"  ⚠ Forex Factory fetch error: {e}")
        return []


def _get_news_events():
    now = time.time()
    with _news_lock:
        if now - _news_cache["fetched_at"] > 21600 or not _news_cache["events"]:
            events = _fetch_forex_factory_calendar()
            if events:
                _news_cache["events"] = events
                _news_cache["fetched_at"] = now
        return list(_news_cache["events"])


def is_news_time():
    now = _dt.datetime.utcnow()
    for ev in _get_news_events():
        diff_min = (now - ev["datetime_utc"]).total_seconds() / 60
        if -PAUSE_MINUTES_BEFORE <= diff_min <= PAUSE_MINUTES_AFTER:
            return True, ev["name"]
    return False, None


def check_upcoming_news():
    now = _dt.datetime.utcnow()
    alerts = []
    for ev in _get_news_events():
        diff_min = (now - ev["datetime_utc"]).total_seconds() / 60
        for minutes_before in ALERT_MINUTES_BEFORE:
            key = f"{ev['name']}_{minutes_before}"
            if -minutes_before - 1 <= diff_min <= -minutes_before + 1:
                with _news_lock:
                    if key not in _news_alerted:
                        _news_alerted.add(key)
                        alerts.append({
                            "name": ev["name"],
                            "minutes_away": abs(int(diff_min)),
                            "impact": ev.get("impact", "High"),
                        })
        with _news_lock:
            if len(_news_alerted) > 200:
                _news_alerted.clear()
    return alerts


def get_dxy_trend():
    now = time.time()
    if now - _dxy_cache["last_update"] < 300:
        return {"trend": _dxy_cache["trend"], "change_pct": _dxy_cache["change_pct"]}
    try:
        url = (f"https://api.twelvedata.com/time_series?symbol=DXY&interval=1h"
               f"&outputsize=24&apikey={TWELVE_API_KEY}")
        data = _fetch_with_retry(url, timeout=8, retries=2, backoff=1.0)
        if data and "values" in data:
            closes = [float(v["close"]) for v in reversed(data["values"])]
            if len(closes) >= 20:
                def ema(arr, n):
                    k = 2/(n+1); e = sum(arr[:n])/n
                    for v in arr[n:]: e = v*k + e*(1-k)
                    return e
                e9, e21 = ema(closes, 9), ema(closes, 21)
                change_pct = (closes[-1] - closes[-6]) / closes[-6] * 100 if len(closes) >= 6 else 0
                if e9 > e21 * 1.001 and change_pct > 0.1:
                    trend = "up"
                elif e9 < e21 * 0.999 and change_pct < -0.1:
                    trend = "down"
                else:
                    trend = "neutral"
                _dxy_cache.update({"trend": trend, "change_pct": round(change_pct, 2), "last_update": now})
                return {"trend": trend, "change_pct": round(change_pct, 2)}
    except Exception as e:
        print(f"  ⚠ DXY fetch: {e}")
    _dxy_cache["last_update"] = now
    return {"trend": "neutral", "change_pct": 0.0}


def check_dxy_correlation(gold_direction):
    dxy = get_dxy_trend()
    if gold_direction == "COMPRAR":
        if dxy["trend"] == "down":
            return {"aligned": True,  "bonus": 10,  "note": f"DXY bajando ({dxy['change_pct']}%) → favorece oro ↑"}
        elif dxy["trend"] == "up" and abs(dxy["change_pct"]) > 0.3:
            return {"aligned": False, "bonus": -25, "note": f"⚠ DXY subiendo fuerte ({dxy['change_pct']}%) → conflicto con oro ↑"}
    if gold_direction == "VENDER":
        if dxy["trend"] == "up":
            return {"aligned": True,  "bonus": 10,  "note": f"DXY subiendo ({dxy['change_pct']}%) → favorece oro ↓"}
        elif dxy["trend"] == "down" and abs(dxy["change_pct"]) > 0.3:
            return {"aligned": False, "bonus": -25, "note": f"⚠ DXY bajando fuerte ({dxy['change_pct']}%) → conflicto con oro ↓"}
    return {"aligned": None, "bonus": 0, "note": f"DXY neutral ({dxy['change_pct']}%)"}


def get_us10y_trend():
    now = time.time()
    if now - _us10y_cache["last_update"] < 300:
        return {"trend": _us10y_cache["trend"], "change_bps": _us10y_cache["change_bps"],
                "level": _us10y_cache["level"]}
    try:
        url = (f"https://api.twelvedata.com/time_series?symbol=US10Y&interval=1h"
               f"&outputsize=24&apikey={TWELVE_API_KEY}")
        data = _fetch_with_retry(url, timeout=8, retries=2, backoff=1.0)
        if data and "values" in data:
            closes = [float(v["close"]) for v in reversed(data["values"])]
            if len(closes) >= 6:
                current  = closes[-1]
                prev_6h  = closes[-6]
                change_bps = (current - prev_6h) * 100
                if change_bps > 3:
                    trend = "rising"
                elif change_bps < -3:
                    trend = "falling"
                else:
                    trend = "stable"
                _us10y_cache.update({
                    "trend": trend, "change_bps": round(change_bps, 1),
                    "level": round(current, 3), "last_update": now
                })
                return {"trend": trend, "change_bps": round(change_bps, 1), "level": round(current, 3)}
    except Exception as e:
        print(f"  ⚠ US10Y fetch: {e}")
    _us10y_cache["last_update"] = now
    return {"trend": "stable", "change_bps": 0.0, "level": 0.0}


def check_yields_for_gold(gold_direction):
    y = get_us10y_trend()
    if gold_direction == "COMPRAR":
        if y["trend"] == "falling":
            return {"aligned": True,  "bonus": 8,   "note": f"US10Y bajando ({y['change_bps']}bps) → favorece oro ↑"}
        elif y["trend"] == "rising" and abs(y["change_bps"]) > 5:
            return {"aligned": False, "bonus": -20,  "note": f"⚠ US10Y subiendo ({y['change_bps']}bps) → presiona oro ↓"}
    elif gold_direction == "VENDER":
        if y["trend"] == "rising":
            return {"aligned": True,  "bonus": 8,   "note": f"US10Y subiendo ({y['change_bps']}bps) → favorece oro ↓"}
        elif y["trend"] == "falling" and abs(y["change_bps"]) > 5:
            return {"aligned": False, "bonus": -20,  "note": f"⚠ US10Y bajando ({y['change_bps']}bps) → presiona oro ↑"}
    return {"aligned": None, "bonus": 0, "note": f"US10Y estable ({y['change_bps']}bps @ {y['level']}%)"}
