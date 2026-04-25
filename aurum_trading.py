"""aurum_trading.py — Risk manager and paper trading engine."""
import time, json, os
import datetime as _dt

from aurum_state import (
    RISK_STATE, _risk_lock,
    _paper_trades, _paper_lock,
    PAPER_TRADES_FILE, PAPER_SPREAD_USD, PAPER_SLIP_USD,
    _live_cache,
)
from aurum_models import register_signal_result


# ── RISK MANAGER ──────────────────────────────────────────
def risk_get_today_stats():
    today = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    with _paper_lock:
        trades_today = [
            t for t in _paper_trades
            if t["status"] == "CLOSED"
            and t.get("closed_at")
            and _dt.datetime.utcfromtimestamp(t["closed_at"]).strftime("%Y-%m-%d") == today
        ]
    if not trades_today:
        return {"count": 0, "total_r": 0, "wins": 0, "losses": 0, "consecutive_losses": 0}
    total_r = sum(t["pnl_r"] for t in trades_today)
    wins    = sum(1 for t in trades_today if t["result"] == "WIN")
    losses  = sum(1 for t in trades_today if t["result"] == "LOSS")
    consec  = 0
    for t in reversed(trades_today):
        if t["result"] == "LOSS": consec += 1
        else: break
    return {"count": len(trades_today), "total_r": round(total_r, 2),
            "wins": wins, "losses": losses, "consecutive_losses": consec}


def risk_get_week_stats():
    now      = time.time()
    week_ago = now - (7 * 86400)
    with _paper_lock:
        trades_week = [
            t for t in _paper_trades
            if t["status"] == "CLOSED" and t.get("closed_at", 0) > week_ago
        ]
    total_r = sum(t["pnl_r"] for t in trades_week) if trades_week else 0
    return {"count": len(trades_week), "total_r": round(total_r, 2)}


def risk_can_trade():
    now = time.time()
    with _risk_lock:
        paused_until = RISK_STATE["paused_until"]
        pause_reason = RISK_STATE["pause_reason"]
    if paused_until > now:
        remaining = int((paused_until - now) / 60)
        return False, f"🚫 Circuit breaker: {pause_reason} — {remaining}min restantes"
    day = risk_get_today_stats()
    if day["total_r"] <= RISK_STATE["daily_loss_limit_r"]:
        tomorrow = (_dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                    + _dt.timedelta(days=1))
        with _risk_lock:
            RISK_STATE["paused_until"] = tomorrow.timestamp()
            RISK_STATE["pause_reason"] = f"Daily loss limit ({day['total_r']}R)"
            paused_until = RISK_STATE["paused_until"]
        remaining = int((paused_until - now) / 60)
        return False, f"🚫 DAILY LOSS LIMIT alcanzado ({day['total_r']}R) — pausa {remaining}min hasta medianoche UTC"
    week = risk_get_week_stats()
    if week["total_r"] <= RISK_STATE["weekly_loss_limit_r"]:
        with _risk_lock:
            RISK_STATE["paused_until"] = now + 86400
            RISK_STATE["pause_reason"] = f"Weekly loss ({week['total_r']}R)"
        return False, f"🚫 WEEKLY LOSS LIMIT ({week['total_r']}R) — pausa 24h"
    if day["consecutive_losses"] >= RISK_STATE["max_consecutive_losses"]:
        with _risk_lock:
            RISK_STATE["paused_until"] = now + (RISK_STATE["circuit_breaker_hours"] * 3600)
            RISK_STATE["pause_reason"] = f"{day['consecutive_losses']} pérdidas consecutivas"
        return False, f"🚫 CIRCUIT BREAKER: {day['consecutive_losses']} pérdidas seguidas — pausa {RISK_STATE['circuit_breaker_hours']}h"
    return True, "OK"


def risk_get_status():
    day = risk_get_today_stats()
    week = risk_get_week_stats()
    can_trade, reason = risk_can_trade()
    now = time.time()
    return {
        "can_trade": can_trade,
        "reason": reason,
        "paused_until": RISK_STATE["paused_until"],
        "pause_remaining_min": max(0, int((RISK_STATE["paused_until"] - now) / 60)) if RISK_STATE["paused_until"] > now else 0,
        "pause_reason": RISK_STATE["pause_reason"],
        "daily": day,
        "weekly": week,
        "limits": {
            "daily_loss_r":        RISK_STATE["daily_loss_limit_r"],
            "weekly_loss_r":       RISK_STATE["weekly_loss_limit_r"],
            "max_consec_losses":   RISK_STATE["max_consecutive_losses"],
            "circuit_breaker_hours": RISK_STATE["circuit_breaker_hours"],
        }
    }


# ── PAPER TRADING ─────────────────────────────────────────
def _load_paper_trades():
    try:
        if os.path.exists(PAPER_TRADES_FILE):
            with open(PAPER_TRADES_FILE, "r") as f:
                loaded = json.load(f)
            with _paper_lock:
                _paper_trades.clear()
                _paper_trades.extend(loaded)
            print(f"  ✓ Paper trading: {len(_paper_trades)} trades cargados del disco")
        else:
            with _paper_lock:
                _paper_trades.clear()
    except Exception as e:
        print(f"  ⚠ Paper load error: {e}")
        with _paper_lock:
            _paper_trades.clear()


def _save_paper_trades():
    try:
        os.makedirs("/data", exist_ok=True)
        tmp = PAPER_TRADES_FILE + ".tmp"
        with _paper_lock:
            data = list(_paper_trades)
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, PAPER_TRADES_FILE)
    except Exception as e:
        print(f"  ⚠ Paper save error: {e}")


def paper_register_signal(signal_data, entry_price):
    """Registra paper trade con spread y slippage realistas."""
    is_buy = signal_data["direction"] == "COMPRAR"
    half_spread = PAPER_SPREAD_USD / 2
    adj_entry = (float(entry_price) + half_spread + PAPER_SLIP_USD if is_buy
                 else float(entry_price) - half_spread - PAPER_SLIP_USD)
    trade = {
        "id":          f"T{int(time.time() * 1000)}",
        "opened_at":   time.time(),
        "opened_iso":  _dt.datetime.utcnow().isoformat(),
        "direction":   signal_data["direction"],
        "entry":       round(adj_entry, 2),
        "tp":          float(signal_data["tp"]),
        "sl":          float(signal_data["sl"]),
        "rr_target":   float(signal_data["rr"]),
        "score":       int(signal_data["score"]),
        "session":     signal_data.get("session", ""),
        "tier":        "SNIPER" if signal_data["score"] >= 85 else "NORMAL",
        "status":      "OPEN",
        "closed_at":   None,
        "exit_price":  None,
        "result":      None,
        "pnl_r":       0.0,
        "duration_sec": 0,
    }
    with _paper_lock:
        _paper_trades.append(trade)
    _save_paper_trades()
    print(f"  📝 Paper trade #{trade['id']}: {trade['direction']} mid=${entry_price:.2f} fill=${adj_entry:.2f}")
    return trade


def paper_check_open_trades(current_price):
    if not current_price or current_price <= 0:
        return
    now = time.time()
    half_spread  = PAPER_SPREAD_USD / 2
    trades_changed = False
    with _paper_lock:
        for t in _paper_trades:
            if t["status"] != "OPEN":
                continue
            is_buy = t["direction"] == "COMPRAR"
            tp     = t["tp"]
            sl     = t["sl"]
            entry  = t["entry"]
            exit_mid = current_price - half_spread if is_buy else current_price + half_spread
            hit_tp = (is_buy and exit_mid >= tp) or (not is_buy and exit_mid <= tp)
            hit_sl = (is_buy and exit_mid <= sl) or (not is_buy and exit_mid >= sl)
            if hit_tp:
                t["status"] = "CLOSED"; t["closed_at"] = now
                t["exit_price"] = tp; t["result"] = "WIN"
                t["pnl_r"] = t["rr_target"]
                t["duration_sec"] = int(now - t["opened_at"])
                trades_changed = True
                print(f"  ✅ WIN #{t['id']}: {t['direction']} +{t['pnl_r']}R ({t['duration_sec']}s)")
                register_signal_result(True, t["direction"], t["pnl_r"], t.get("score", 0))
            elif hit_sl:
                t["status"] = "CLOSED"; t["closed_at"] = now
                t["exit_price"] = sl; t["result"] = "LOSS"
                t["pnl_r"] = -1.0
                t["duration_sec"] = int(now - t["opened_at"])
                trades_changed = True
                print(f"  ❌ LOSS #{t['id']}: {t['direction']} -1.0R ({t['duration_sec']}s)")
                register_signal_result(False, t["direction"], -1.0, t.get("score", 0))
            elif now - t["opened_at"] > 86400:
                t["status"] = "CLOSED"; t["closed_at"] = now
                t["exit_price"] = round(exit_mid, 2); t["result"] = "TIMEOUT"
                if is_buy:
                    progress = (exit_mid - entry) / (tp - entry) if tp != entry else 0
                else:
                    progress = (entry - exit_mid) / (entry - tp) if tp != entry else 0
                t["pnl_r"] = round(max(-1.0, min(t["rr_target"], progress * t["rr_target"])), 2)
                t["duration_sec"] = int(now - t["opened_at"])
                trades_changed = True
                print(f"  ⏰ TIMEOUT #{t['id']}: {t['pnl_r']}R")
    if trades_changed:
        _save_paper_trades()


def paper_get_stats():
    with _paper_lock:
        all_trades = list(_paper_trades)
    closed = [t for t in all_trades if t["status"] == "CLOSED"]
    opened = [t for t in all_trades if t["status"] == "OPEN"]
    if not closed:
        return {
            "total": 0, "open": len(opened),
            "wins": 0, "losses": 0, "timeouts": 0,
            "win_rate": 0, "total_r": 0, "avg_r": 0,
            "best_r": 0, "worst_r": 0,
            "open_trades": opened[-5:],
            "recent_trades": [],
        }
    wins     = [t for t in closed if t["result"] == "WIN"]
    losses   = [t for t in closed if t["result"] == "LOSS"]
    timeouts = [t for t in closed if t["result"] == "TIMEOUT"]
    total_r  = sum(t["pnl_r"] for t in closed)
    win_rate = len(wins) / len(closed) * 100 if closed else 0
    pnl_list = [t["pnl_r"] for t in closed]
    return {
        "total":    len(closed),
        "open":     len(opened),
        "wins":     len(wins),
        "losses":   len(losses),
        "timeouts": len(timeouts),
        "win_rate": round(win_rate, 1),
        "total_r":  round(total_r, 2),
        "avg_r":    round(total_r / len(closed), 2) if closed else 0,
        "best_r":   round(max(pnl_list), 2) if pnl_list else 0,
        "worst_r":  round(min(pnl_list), 2) if pnl_list else 0,
        "open_trades":   opened[-5:],
        "recent_trades": closed[-10:],
    }


def _worker_paper():
    import time as _t
    while True:
        try:
            cp = _live_cache.get("price")
            if cp:
                paper_check_open_trades(cp)
        except Exception as e:
            print(f"  ⚠ Paper worker: {e}")
        _t.sleep(1)
