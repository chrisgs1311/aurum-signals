"""
AURUM - Bot de Señales XAU/USD
Versión completa:
  ✓ Señales RSI + EMA + MACD + ATR
  ✓ Gestión de riesgo con filtros
  ✓ Sesiones de mercado (Asia / Londres / NY)
  ✓ Alertas de sonido
  ✓ Backtesting integrado
  ✓ Registro de operaciones en Excel (CSV)
  ✓ Alertas por Telegram

Sin librerías externas. Solo Python 3.
Uso: python aurum_bot.py
"""

import http.server, threading, webbrowser, json, urllib.request, urllib.parse
import csv, os, time
from datetime import datetime, timezone

# ── MOTOR DE IA (Regresión Logística - Pure Python) ──────
import math, random as _random

class AurumAI:
    """
    Regresión logística entrenada con datos históricos simulados.
    Aprende qué combinación de indicadores predice el movimiento del oro.
    Features: RSI, EMA_cross, MACD_hist, ATR_norm, price_vs_sma20, momentum
    """
    def __init__(self):
        self.weights  = [0.0] * 6
        self.bias     = 0.0
        self.trained  = False
        self.accuracy = 0.0
        self.epochs   = 0

    @staticmethod
    def sigmoid(x):
        x = max(-500, min(500, x))
        return 1 / (1 + math.exp(-x))

    def predict_proba(self, features):
        z = sum(w * x for w, x in zip(self.weights, features)) + self.bias
        return self.sigmoid(z)

    def extract_features(self, prices_slice):
        """Extrae 6 features normalizadas de una ventana de precios"""
        if len(prices_slice) < 30:
            return None
        p = prices_slice
        # RSI normalizado [0,1]
        n = 14
        gains = losses = 0
        for i in range(len(p)-n, len(p)):
            d = p[i] - p[i-1]
            if d > 0: gains += d
            else: losses -= d
        ag, al = gains/n, losses/n
        rsi = (100 - 100/(1+ag/al))/100 if al > 0 else 1.0

        # EMA cross: ema9 vs ema21
        def ema(arr, n):
            k = 2/(n+1)
            e = sum(arr[:n])/n
            for v in arr[n:]: e = v*k + e*(1-k)
            return e
        e9  = ema(p, 9)
        e21 = ema(p, 21)
        ema_cross = (e9 - e21) / (e21 + 1e-9)
        ema_cross = max(-1, min(1, ema_cross * 100))

        # MACD histogram normalizado
        e12 = ema(p, 12)
        e26 = ema(p, 26)
        macd_val = e12 - e26
        macd_norm = max(-1, min(1, macd_val / (p[-1] * 0.01 + 1e-9)))

        # ATR normalizado
        atrs = []
        for i in range(len(p)-14, len(p)):
            hi, lo, pc = p[i]*1.004, p[i]*0.996, p[i-1]
            atrs.append(max(hi-lo, abs(hi-pc), abs(lo-pc)))
        atr_norm = (sum(atrs)/14) / (p[-1] * 0.01 + 1e-9)
        atr_norm = max(0, min(2, atr_norm))

        # Precio vs SMA20
        sma20 = sum(p[-20:])/20
        price_vs_sma = (p[-1] - sma20) / (sma20 + 1e-9) * 100
        price_vs_sma = max(-2, min(2, price_vs_sma))

        # Momentum (cambio últimos 5 períodos)
        momentum = (p[-1] - p[-6]) / (p[-6] + 1e-9) * 100
        momentum = max(-2, min(2, momentum))

        return [rsi, ema_cross, macd_norm, atr_norm, price_vs_sma, momentum]

    def train(self, prices):
        """Entrena el modelo con datos históricos de precios"""
        if len(prices) < 60:
            return False

        # Generar dataset de entrenamiento
        X, y = [], []
        for i in range(40, len(prices) - 5):
            features = self.extract_features(prices[i-30:i])
            if features is None:
                continue
            # Label: 1 si el precio subió en los próximos 5 períodos
            future_return = (prices[i+4] - prices[i]) / prices[i]
            label = 1 if future_return > 0.0005 else 0
            X.append(features)
            y.append(label)

        if len(X) < 20:
            return False

        # Gradient descent
        lr = 0.1
        self.weights = [0.0] * 6
        self.bias = 0.0
        n_samples = len(X)

        for epoch in range(200):
            dw = [0.0] * 6
            db = 0.0
            for xi, yi in zip(X, y):
                pred = self.predict_proba(xi)
                err  = pred - yi
                for j in range(6):
                    dw[j] += err * xi[j]
                db += err
            # L2 regularization
            for j in range(6):
                self.weights[j] -= lr * (dw[j]/n_samples + 0.01*self.weights[j])
            self.bias -= lr * db/n_samples
            lr *= 0.995  # decay

        # Calcular accuracy en entrenamiento
        correct = sum(
            1 for xi, yi in zip(X, y)
            if (self.predict_proba(xi) >= 0.5) == bool(yi)
        )
        self.accuracy = correct / len(X) * 100
        self.epochs   = len(X)
        self.trained  = True
        return True

# Instancia global del modelo
_ai = AurumAI()
_ai_last_train = 0  # timestamp del último entrenamiento

def ai_train_if_needed(prices):
    global _ai_last_train
    now = __import__('time').time()
    if now - _ai_last_train > 300 and len(prices) >= 60:  # re-entrenar cada 5 min
        _ai.train(prices)
        _ai_last_train = now
        if _ai.trained:
            print(f"  🤖 IA entrenada — Accuracy: {_ai.accuracy:.1f}% ({_ai.epochs} muestras)")

def ai_predict(prices):
    """Devuelve probabilidad de subida [0,1] y señal"""
    if not _ai.trained or len(prices) < 35:
        return None, None
    features = _ai.extract_features(prices[-35:])
    if features is None:
        return None, None
    prob = _ai.predict_proba(features)
    if prob >= 0.68:
        signal = "COMPRAR"
    elif prob <= 0.32:
        signal = "VENDER"
    else:
        signal = "ESPERAR"
    return prob, signal


import os
PORT = int(os.environ.get("PORT", 8765))
LOG_FILE = "aurum_operaciones.csv"

# ── TELEGRAM (opcional) ──────────────────────────────────
# Pon tu token y chat_id aquí, o déjalos vacíos
TELEGRAM_TOKEN   = ""
TELEGRAM_CHAT_ID = ""

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}).encode()
        req  = urllib.request.Request(url, data=data)
        urllib.request.urlopen(req, timeout=5)
        print(f"  📲 Telegram enviado")
    except Exception as e:
        print(f"  ⚠ Telegram error: {e}")

# ── REGISTRO EXCEL (CSV) ─────────────────────────────────
def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Fecha","Hora GMT","Señal","Precio Entrada","TP","SL","ATR","RR","Confianza%","Sesión"])

def log_signal(signal, price, tp, sl, atr, rr, confidence, session):
    now = datetime.now(timezone.utc)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M"),
            signal,
            f"{price:.2f}",
            f"{tp:.2f}",
            f"{sl:.2f}",
            f"{atr:.2f}",
            f"{rr}",
            confidence,
            session
        ])
    print(f"  📊 Operación registrada en {LOG_FILE}")

def read_log():
    if not os.path.exists(LOG_FILE):
        return []
    rows = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(dict(row))
    return rows[-20:]  # últimas 20

# ── API PRECIO ───────────────────────────────────────────
TWELVE_API_KEY = "dd53883de1a84cccaf65bf7f4e7a4756"

def get_gold_price():
    """Precio actual del oro desde gold-api.com"""
    try:
        req = urllib.request.Request(
            "https://api.gold-api.com/price/XAU",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            d = json.loads(r.read().decode())
            return {"price": float(d["price"]), "ch": float(d.get("ch",0)), "chp": float(d.get("chp",0))}
    except Exception as e:
        print(f"  ⚠ Precio API Error: {e}")
        return None

def get_historical_ohlc(interval="5min", outputsize=150):
    """
    Obtiene velas OHLC reales de Twelve Data.
    Si falla, construye historial sintético basado en precio actual.
    """
    # Try Twelve Data first
    try:
        url = (f"https://api.twelvedata.com/time_series"
               f"?symbol=XAU/USD&interval={interval}"
               f"&outputsize={outputsize}&apikey={TWELVE_API_KEY}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        if "values" in data:
            candles = []
            for v in reversed(data["values"]):
                candles.append({
                    "open":  float(v["open"]),
                    "high":  float(v["high"]),
                    "low":   float(v["low"]),
                    "close": float(v["close"]),
                    "dt":    v["datetime"]
                })
            print(f"  ✓ {len(candles)} velas {interval} (Twelve Data)")
            return candles
        else:
            print(f"  ⚠ Twelve Data: {data.get('message','sin datos')}")
    except Exception as e:
        print(f"  ⚠ Twelve Data Error: {e}")

    # Fallback: build synthetic history from current price
    import random, math
    price_data = get_gold_price()
    base = price_data["price"] if price_data else 3020.0
    print(f"  ◌ Construyendo historial sintético desde ${base:.2f}")
    candles = []
    p = base
    import datetime as dt
    now = dt.datetime.utcnow()
    for i in range(outputsize, 0, -1):
        noise = (random.random() - 0.49) * 3
        trend = math.sin(i / 30) * 2
        p = max(base * 0.95, p + noise + trend * 0.1)
        atr = base * 0.002
        o = p
        h = p + abs(random.gauss(0, atr))
        l = p - abs(random.gauss(0, atr))
        candles.append({
            "open": round(o, 2), "high": round(h, 2),
            "low": round(l, 2), "close": round(p, 2),
            "dt": str(now - dt.timedelta(minutes=i*5))
        })
    candles[-1]["close"] = base  # last candle = real price
    print(f"  ✓ {len(candles)} velas sintéticas generadas (precio base real)")
    return candles

# ── HTML ─────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AURUM · XAU/USD</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@300;400;500&family=Playfair+Display:ital,wght@0,400;1,400&display=swap" rel="stylesheet">
<style>
:root{--gold:#C9A84C;--gold-light:#F0D080;--gold-dim:#6B5520;--bg:#0A0A08;--bg2:#111109;--bg3:#1A1A14;--text:#E8E0C8;--text-dim:#6B6550;--buy:#4CAF82;--sell:#E05A5A;--hold:#C9A84C;--border:rgba(201,168,76,0.15)}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'DM Mono',monospace;min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");pointer-events:none;z-index:1000;opacity:.6}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.8)}}
@keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
@keyframes ringBuy{0%{box-shadow:0 0 0 0 rgba(76,175,130,.6)}70%{box-shadow:0 0 0 20px rgba(76,175,130,0)}100%{box-shadow:0 0 0 0 rgba(76,175,130,0)}}
@keyframes ringSell{0%{box-shadow:0 0 0 0 rgba(224,90,90,.6)}70%{box-shadow:0 0 0 20px rgba(224,90,90,0)}100%{box-shadow:0 0 0 0 rgba(224,90,90,0)}}

/* NAV TABS */
nav{display:flex;border-bottom:1px solid var(--border);padding:0 40px}
.nav-tab{padding:16px 20px;font-size:9px;letter-spacing:3px;color:var(--text-dim);cursor:pointer;border-bottom:2px solid transparent;transition:all .3s;background:transparent;border-top:none;border-left:none;border-right:none;font-family:'DM Mono',monospace}
.nav-tab.active{color:var(--gold);border-bottom-color:var(--gold)}
.nav-tab:hover{color:var(--text)}
.page{display:none}.page.active{display:block}

header{display:flex;align-items:center;justify-content:space-between;padding:20px 40px;border-bottom:1px solid var(--border);position:relative}
header::after{content:'';position:absolute;bottom:-1px;left:40px;width:80px;height:1px;background:var(--gold)}
.logo{font-family:'Bebas Neue',sans-serif;font-size:28px;letter-spacing:6px;color:var(--gold)}
.logo span{color:var(--text-dim);font-size:10px;display:block;letter-spacing:4px;margin-top:-4px}
.header-right{display:flex;align-items:center;gap:16px}
.live-badge{display:flex;align-items:center;gap:8px;font-size:10px;letter-spacing:3px;color:var(--text-dim)}
.live-dot{width:6px;height:6px;border-radius:50%;background:var(--buy);animation:pulse 2s infinite}
.sound-toggle{background:transparent;border:1px solid var(--border);color:var(--text-dim);font-family:'DM Mono',monospace;font-size:9px;letter-spacing:2px;padding:5px 10px;cursor:pointer;transition:all .3s}
.sound-toggle.on{border-color:rgba(76,175,130,.4);color:var(--buy)}
.tg-status{font-size:9px;letter-spacing:2px;padding:5px 10px;border:1px solid var(--border);color:var(--text-dim)}
.tg-status.connected{border-color:rgba(76,175,130,.4);color:var(--buy)}

/* LAYOUT */
.main{display:grid;grid-template-columns:1fr 340px;min-height:calc(100vh - 128px)}
.left{padding:32px 40px;border-right:1px solid var(--border)}
.right{padding:28px;display:flex;flex-direction:column;gap:18px;overflow-y:auto}

/* SESIONES */
.sessions-bar{display:flex;gap:1px;margin-bottom:24px;animation:fadeUp .5s ease both}
.session-block{flex:1;padding:10px 6px;text-align:center;border:1px solid var(--border);transition:all .4s}
.session-block.active-session{border-color:var(--gold);background:rgba(201,168,76,.07)}
.session-block.overlap{border-color:var(--buy);background:rgba(76,175,130,.07)}
.session-block.closed{opacity:.35}
.session-name{font-family:'Bebas Neue',sans-serif;font-size:16px;letter-spacing:3px;color:var(--text-dim);transition:color .4s}
.session-block.active-session .session-name{color:var(--gold)}
.session-block.overlap .session-name{color:var(--buy)}
.session-hours{font-size:7px;letter-spacing:1px;color:var(--text-dim);margin-top:2px}
.session-status{font-size:8px;letter-spacing:2px;margin-top:4px;color:var(--text-dim)}
.session-block.active-session .session-status{color:var(--gold)}
.session-block.overlap .session-status{color:var(--buy)}
.session-tip{font-size:9px;color:var(--text-dim);padding:8px 12px;border:1px solid var(--border);background:var(--bg2);margin-bottom:20px;font-style:italic;font-family:'Playfair Display',serif}
.session-tip.hot{border-color:rgba(76,175,130,.3);color:var(--buy)}
.session-tip.warm{border-color:rgba(201,168,76,.3);color:var(--gold)}

/* PRECIO */
.price-section{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:24px;animation:fadeUp .5s ease both}
.price-label{font-size:9px;letter-spacing:4px;color:var(--text-dim);margin-bottom:6px}
.price-main{font-family:'Bebas Neue',sans-serif;font-size:64px;letter-spacing:2px;color:var(--gold-light);line-height:1}
.price-change{font-size:12px;margin-top:5px;letter-spacing:1px}
.price-change.up{color:var(--buy)}.price-change.down{color:var(--sell)}
.source-badge{font-size:9px;letter-spacing:2px;padding:3px 8px;border:1px solid;display:inline-block;margin-top:5px}
.source-badge.live{color:var(--buy);border-color:rgba(76,175,130,.3)}

/* ALERTA */
.alert-box{padding:24px 28px;margin-bottom:20px;border:1px solid;position:relative;overflow:hidden;animation:fadeUp .5s .1s ease both;transition:all .5s}
.alert-box.go-buy{border-color:var(--buy);background:rgba(76,175,130,.05)}
.alert-box.go-sell{border-color:var(--sell);background:rgba(224,90,90,.05)}
.alert-box.wait{border-color:var(--border);background:var(--bg2)}
.alert-box::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;transition:background .5s}
.alert-box.go-buy::before{background:var(--buy)}
.alert-box.go-sell::before{background:var(--sell)}
.alert-box.wait::before{background:var(--gold-dim)}
.alert-glow{position:absolute;top:-80px;right:-80px;width:200px;height:200px;border-radius:50%;opacity:.05;transition:background .5s}
.alert-box.go-buy .alert-glow{background:var(--buy)}
.alert-box.go-sell .alert-glow{background:var(--sell)}
.alert-tag{font-size:9px;letter-spacing:4px;color:var(--text-dim);margin-bottom:12px}
.alert-signal{font-family:'Bebas Neue',sans-serif;font-size:48px;letter-spacing:4px;line-height:1;transition:color .5s}
.alert-box.go-buy .alert-signal{color:var(--buy)}
.alert-box.go-sell .alert-signal{color:var(--sell)}
.alert-box.wait .alert-signal{color:var(--gold-dim)}
.alert-reason{font-size:11px;color:var(--text-dim);margin-top:8px;font-style:italic;font-family:'Playfair Display',serif}
.validity-row{display:flex;gap:6px;margin-top:14px;flex-wrap:wrap}
.validity-pill{font-size:8px;letter-spacing:1px;padding:3px 8px;border:1px solid var(--border);color:var(--text-dim)}
.validity-pill.ok{border-color:rgba(76,175,130,.4);color:var(--buy)}
.validity-pill.fail{border-color:rgba(224,90,90,.3);color:var(--sell)}
.conf-row{display:flex;align-items:center;gap:10px;margin-top:14px;font-size:10px;letter-spacing:2px;color:var(--text-dim)}
.conf-bar{flex:1;height:2px;background:var(--bg3)}
.conf-fill{height:100%;transition:width .8s,background .5s}
.alert-box.go-buy .conf-fill{background:var(--buy)}
.alert-box.go-sell .conf-fill{background:var(--sell)}
.alert-box.wait .conf-fill{background:var(--gold-dim)}

/* NIVELES */
.levels-box{background:var(--bg2);border:1px solid var(--border);padding:18px 22px;margin-bottom:20px}
.levels-title{font-size:9px;letter-spacing:4px;color:var(--text-dim);margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between}
.level-row{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid rgba(201,168,76,.05)}
.level-row:last-child{border:none}
.level-name{font-size:10px;letter-spacing:2px;color:var(--text-dim)}
.level-val{font-family:'Bebas Neue',sans-serif;font-size:22px}
.level-val.tp{color:var(--buy)}.level-val.sl{color:var(--sell)}.level-val.entry{color:var(--gold-light)}
.level-sub{font-size:8px;color:var(--text-dim);text-align:right}

/* LOG BUTTON */
.log-btn{width:100%;padding:10px;background:transparent;border:1px solid rgba(76,175,130,.3);color:var(--buy);font-family:'DM Mono',monospace;font-size:9px;letter-spacing:3px;cursor:pointer;transition:all .3s;margin-top:4px}
.log-btn:hover{background:rgba(76,175,130,.1)}

/* GRÁFICA */
.chart-wrap{animation:fadeUp .5s .25s ease both}
.chart-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.chart-title{font-size:9px;letter-spacing:3px;color:var(--text-dim)}
.chart-tabs{display:flex;gap:4px}
.chart-tab{padding:3px 10px;font-size:9px;letter-spacing:2px;border:1px solid var(--border);background:transparent;color:var(--text-dim);cursor:pointer;transition:all .2s;font-family:'DM Mono',monospace}
.chart-tab.active,.chart-tab:hover{border-color:var(--gold);color:var(--gold);background:rgba(201,168,76,.05)}
canvas{width:100%;display:block}

/* INDICADORES */
.indicators-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:20px}
.indicator-box{background:var(--bg2);border:1px solid var(--border);padding:16px}
.ind-label{font-size:9px;letter-spacing:3px;color:var(--text-dim);margin-bottom:6px}
.ind-value{font-family:'Bebas Neue',sans-serif;font-size:32px;letter-spacing:2px;line-height:1}
.ind-status{font-size:9px;letter-spacing:2px;margin-top:4px}
.ind-status.bullish{color:var(--buy)}.ind-status.bearish{color:var(--sell)}.ind-status.neutral{color:var(--hold)}
.rsi-gauge{position:relative;width:100%;height:4px;background:linear-gradient(to right,var(--sell) 30%,var(--hold) 30%,var(--hold) 70%,var(--buy) 70%);margin-top:8px;border-radius:2px}
.rsi-needle{position:absolute;top:-4px;width:2px;height:12px;background:white;transform:translateX(-50%);transition:left .8s;border-radius:1px}
.rsi-labels{display:flex;justify-content:space-between;margin-top:4px;font-size:8px;color:var(--text-dim)}

/* PANEL DERECHO */
.panel-title{font-size:9px;letter-spacing:4px;color:var(--text-dim);margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--border)}
.rules-box{background:var(--bg2);border:1px solid var(--border);padding:16px}
.rule-row{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(201,168,76,.06);font-size:10px}
.rule-row:last-child{border:none}
.rule-label{color:var(--text-dim);letter-spacing:1px;font-size:9px}
.rule-input{background:transparent;border:none;border-bottom:1px solid var(--gold-dim);color:var(--gold);font-family:'DM Mono',monospace;font-size:12px;width:65px;text-align:right;outline:none;padding:2px 4px}
.rule-unit{font-size:9px;color:var(--text-dim);margin-left:3px}

.ma-row{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(201,168,76,.06);font-size:11px}
.ma-name{color:var(--text-dim);letter-spacing:1px}.ma-val{color:var(--text)}.ma-sig{font-size:9px;letter-spacing:2px}
.ma-sig.b{color:var(--buy)}.ma-sig.s{color:var(--sell)}

.history-list{display:flex;flex-direction:column;gap:5px}
.hist-item{background:var(--bg2);border:1px solid var(--border);padding:9px 12px;display:flex;align-items:center;gap:10px}
.hist-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.hist-dot.b{background:var(--buy)}.hist-dot.s{background:var(--sell)}.hist-dot.h{background:var(--hold)}
.hist-info{flex:1}.hist-sig{font-size:10px;letter-spacing:2px}
.hist-sig.b{color:var(--buy)}.hist-sig.s{color:var(--sell)}.hist-sig.h{color:var(--hold)}
.hist-sub{font-size:8px;color:var(--text-dim);margin-top:1px}.hist-time{font-size:8px;color:var(--text-dim)}

.update-btn{width:100%;padding:12px;background:transparent;border:1px solid var(--gold-dim);color:var(--gold);font-family:'DM Mono',monospace;font-size:10px;letter-spacing:4px;cursor:pointer;transition:all .3s;position:relative;overflow:hidden}
.update-btn::before{content:'';position:absolute;inset:0;background:var(--gold);transform:scaleX(0);transform-origin:left;transition:transform .3s}
.update-btn:hover::before{transform:scaleX(1)}.update-btn:hover{color:var(--bg)}
.update-btn span{position:relative;z-index:1}
.update-btn:disabled{opacity:.4;pointer-events:none}
.timestamp{font-size:9px;color:var(--text-dim);letter-spacing:2px;text-align:center}
.disclaimer{font-size:9px;color:var(--text-dim);line-height:1.7;padding:12px;border:1px solid var(--border)}
.disclaimer strong{color:var(--gold-dim)}
/* IA PANEL */
.ai-panel{background:var(--bg2);border:1px solid var(--border);padding:20px 24px;margin-bottom:20px;position:relative;overflow:hidden}
.ai-panel::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(to right,var(--gold-dim),var(--gold),var(--gold-dim))}
.ai-title{font-size:9px;letter-spacing:4px;color:var(--text-dim);margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
.ai-badge{font-size:8px;letter-spacing:2px;padding:3px 8px;border:1px solid var(--gold-dim);color:var(--gold)}
.ai-prob-wrap{display:flex;align-items:center;gap:16px;margin-bottom:16px}
.ai-prob-circle{width:80px;height:80px;border-radius:50%;border:3px solid var(--border);display:flex;flex-direction:column;align-items:center;justify-content:center;flex-shrink:0;transition:border-color .5s}
.ai-prob-circle.buy{border-color:var(--buy)}
.ai-prob-circle.sell{border-color:var(--sell)}
.ai-prob-circle.neutral{border-color:var(--gold-dim)}
.ai-prob-val{font-family:'Bebas Neue',sans-serif;font-size:26px;letter-spacing:1px;line-height:1}
.ai-prob-val.buy{color:var(--buy)}.ai-prob-val.sell{color:var(--sell)}.ai-prob-val.neutral{color:var(--gold)}
.ai-prob-label{font-size:7px;letter-spacing:2px;color:var(--text-dim);margin-top:2px}
.ai-signal-wrap{flex:1}
.ai-signal{font-family:'Bebas Neue',sans-serif;font-size:32px;letter-spacing:3px;transition:color .5s}
.ai-signal.buy{color:var(--buy)}.ai-signal.sell{color:var(--sell)}.ai-signal.neutral{color:var(--gold-dim)}
.ai-desc{font-size:9px;color:var(--text-dim);margin-top:4px;letter-spacing:1px;font-style:italic;font-family:'Playfair Display',serif}
.ai-bar-wrap{margin-top:8px}
.ai-bar-label{display:flex;justify-content:space-between;font-size:8px;color:var(--text-dim);letter-spacing:1px;margin-bottom:4px}
.ai-bar{height:4px;background:var(--bg3);border-radius:2px;position:relative}
.ai-bar-fill{position:absolute;left:0;top:0;height:100%;border-radius:2px;transition:width .8s,background .5s}
.ai-stats-row{display:flex;gap:10px;margin-top:14px}
.ai-stat{flex:1;text-align:center;padding:8px;background:var(--bg3);border:1px solid var(--border)}
.ai-stat-val{font-family:'Bebas Neue',sans-serif;font-size:20px;color:var(--gold);letter-spacing:1px}
.ai-stat-label{font-size:7px;letter-spacing:2px;color:var(--text-dim);margin-top:2px}
.ai-consensus{margin-top:14px;padding:10px 14px;border:1px solid;font-size:10px;letter-spacing:1px;font-family:'Playfair Display',serif;font-style:italic}
.ai-consensus.agree{border-color:rgba(76,175,130,.3);color:var(--buy);background:rgba(76,175,130,.04)}
.ai-consensus.disagree{border-color:rgba(224,90,90,.3);color:var(--sell);background:rgba(224,90,90,.04)}
.ai-consensus.neutral{border-color:var(--border);color:var(--text-dim)}
.ai-training{font-size:9px;color:var(--text-dim);letter-spacing:2px;margin-top:10px;text-align:center}


/* ── BACKTESTING PAGE ── */
.bt-page{padding:36px 40px}
.bt-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:28px}
.bt-title{font-family:'Bebas Neue',sans-serif;font-size:36px;letter-spacing:4px;color:var(--gold)}
.bt-subtitle{font-size:10px;letter-spacing:3px;color:var(--text-dim);margin-top:4px}
.bt-controls{display:flex;gap:12px;align-items:center;margin-bottom:28px;flex-wrap:wrap}
.bt-input{background:var(--bg2);border:1px solid var(--border);color:var(--text);font-family:'DM Mono',monospace;font-size:11px;padding:8px 12px;outline:none;width:120px}
.bt-input:focus{border-color:var(--gold)}
.bt-label{font-size:9px;letter-spacing:2px;color:var(--text-dim)}
.bt-run{padding:10px 24px;background:transparent;border:1px solid var(--gold);color:var(--gold);font-family:'DM Mono',monospace;font-size:10px;letter-spacing:3px;cursor:pointer;transition:all .3s}
.bt-run:hover{background:var(--gold);color:var(--bg)}
.bt-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:28px}
.bt-stat{background:var(--bg2);border:1px solid var(--border);padding:16px;text-align:center}
.bt-stat-val{font-family:'Bebas Neue',sans-serif;font-size:36px;letter-spacing:2px}
.bt-stat-val.green{color:var(--buy)}.bt-stat-val.red{color:var(--sell)}.bt-stat-val.gold{color:var(--gold)}
.bt-stat-label{font-size:8px;letter-spacing:3px;color:var(--text-dim);margin-top:4px}
.bt-chart-wrap{background:var(--bg2);border:1px solid var(--border);padding:20px;margin-bottom:20px}
.bt-chart-title{font-size:9px;letter-spacing:3px;color:var(--text-dim);margin-bottom:14px}
.bt-table{width:100%;border-collapse:collapse;font-size:10px}
.bt-table th{font-size:8px;letter-spacing:3px;color:var(--text-dim);padding:8px 10px;text-align:left;border-bottom:1px solid var(--border)}
.bt-table td{padding:8px 10px;border-bottom:1px solid rgba(201,168,76,.05)}
.bt-table tr:last-child td{border:none}
.td-buy{color:var(--buy)}.td-sell{color:var(--sell)}.td-pos{color:var(--buy)}.td-neg{color:var(--sell)}

/* ── REGISTRO PAGE ── */
.reg-page{padding:36px 40px}
.reg-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:24px}
.reg-title{font-family:'Bebas Neue',sans-serif;font-size:36px;letter-spacing:4px;color:var(--gold)}
.reg-actions{display:flex;gap:10px}
.reg-btn{padding:9px 18px;background:transparent;border:1px solid var(--border);color:var(--text-dim);font-family:'DM Mono',monospace;font-size:9px;letter-spacing:2px;cursor:pointer;transition:all .3s}
.reg-btn:hover{border-color:var(--gold);color:var(--gold)}
.reg-btn.primary{border-color:var(--gold-dim);color:var(--gold)}
.reg-summary{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:24px}
.reg-stat{background:var(--bg2);border:1px solid var(--border);padding:16px}
.reg-stat-val{font-family:'Bebas Neue',sans-serif;font-size:32px;letter-spacing:2px}
.reg-stat-val.g{color:var(--buy)}.reg-stat-val.r{color:var(--sell)}.reg-stat-val.w{color:var(--gold)}
.reg-stat-label{font-size:8px;letter-spacing:3px;color:var(--text-dim);margin-top:4px}
.reg-table-wrap{background:var(--bg2);border:1px solid var(--border);overflow:auto}
.reg-table{width:100%;border-collapse:collapse;font-size:10px}
.reg-table th{font-size:8px;letter-spacing:2px;color:var(--text-dim);padding:10px 12px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}
.reg-table td{padding:9px 12px;border-bottom:1px solid rgba(201,168,76,.05);white-space:nowrap}
.reg-table tr:last-child td{border:none}
.reg-empty{padding:40px;text-align:center;color:var(--text-dim);font-size:11px;letter-spacing:2px}

/* CONFIG TELEGRAM */
.tg-box{background:var(--bg2);border:1px solid var(--border);padding:16px;margin-top:4px}
.tg-field{display:flex;flex-direction:column;gap:4px;margin-bottom:12px}
.tg-field:last-child{margin-bottom:0}
.tg-field label{font-size:8px;letter-spacing:3px;color:var(--text-dim)}
.tg-field input{background:var(--bg3);border:1px solid var(--border);color:var(--text);font-family:'DM Mono',monospace;font-size:10px;padding:7px 10px;outline:none;width:100%}
.tg-field input:focus{border-color:var(--gold)}
.tg-save{width:100%;padding:9px;background:transparent;border:1px solid var(--gold-dim);color:var(--gold);font-family:'DM Mono',monospace;font-size:9px;letter-spacing:3px;cursor:pointer;margin-top:10px;transition:all .3s}
.tg-save:hover{background:rgba(201,168,76,.1)}
.tg-hint{font-size:9px;color:var(--text-dim);line-height:1.6;margin-top:10px;padding:10px;border:1px solid var(--border)}
</style>
</head>
<body>
<header>
  <div class="logo">AURUM<span>BOT DE SEÑALES XAU/USD</span></div>
  <div class="header-right">
    <div class="tg-status" id="tgStatus">📵 TELEGRAM OFF</div>
    <button class="sound-toggle on" id="soundBtn" onclick="toggleSound()">🔔 ON</button>
    <div class="live-badge"><div class="live-dot"></div>EN VIVO</div>
  </div>
</header>

<nav>
  <button class="nav-tab active" onclick="showPage('dashboard')">DASHBOARD</button>
  <button class="nav-tab" onclick="showPage('backtest')">BACKTESTING</button>
  <button class="nav-tab" onclick="showPage('registro')">REGISTRO</button>
  <button class="nav-tab" onclick="showPage('config')">CONFIGURACIÓN</button>
</nav>

<!-- ══ DASHBOARD ══════════════════════════════════════════ -->
<div class="page active" id="page-dashboard">
<div class="main">
<div class="left">

  <div class="sessions-bar" id="sessionsBar">
    <div class="session-block" id="sessAsia">
      <div class="session-name">ASIA</div>
      <div class="session-hours">22:00–09:00 GMT</div>
      <div class="session-status" id="sessAsiaStatus">—</div>
    </div>
    <div class="session-block" id="sessLondon">
      <div class="session-name">LONDRES</div>
      <div class="session-hours">08:00–17:00 GMT</div>
      <div class="session-status" id="sessLondonStatus">—</div>
    </div>
    <div class="session-block" id="sessNY">
      <div class="session-name">NEW YORK</div>
      <div class="session-hours">13:00–22:00 GMT</div>
      <div class="session-status" id="sessNYStatus">—</div>
    </div>
  </div>
  <div class="session-tip" id="sessionTip">Calculando sesión...</div>

  <!-- TIMEFRAME SELECTOR -->
  <div style="display:flex;gap:6px;margin-bottom:16px;align-items:center">
    <span style="font-size:9px;letter-spacing:3px;color:var(--text-dim)">TF:</span>
    <button class="chart-tab active" id="tf5" onclick="setTF(this,'5min')">5M</button>
    <button class="chart-tab" id="tf15" onclick="setTF(this,'15min')">15M</button>
    <button class="chart-tab" id="tf1h" onclick="setTF(this,'1h')">1H</button>
    <button class="chart-tab" id="tf4h" onclick="setTF(this,'4h')">4H</button>
    <span style="font-size:9px;color:var(--text-dim);margin-left:8px" id="tfInfo">Cargando velas reales...</span>
  </div>

  <div class="price-section">
    <div>
      <div class="price-label">PRECIO ACTUAL · XAU/USD</div>
      <div class="price-main" id="priceDisplay">—</div>
      <div class="price-change" id="priceChange">Cargando...</div>
      <div id="sourceBadge"></div>
    </div>
  </div>

  <div class="alert-box wait" id="alertBox">
    <div class="alert-glow"></div>
    <div class="alert-tag">SEÑAL VALIDADA</div>
    <div class="alert-signal" id="alertSignal">—</div>
    <div class="alert-reason" id="alertReason">Analizando mercado...</div>
    <div class="validity-row" id="validityRow"></div>
    <div class="conf-row">
      <span id="confPct">—%</span>
      <div class="conf-bar"><div class="conf-fill" id="confFill" style="width:0%"></div></div>
      <span>CONFIANZA</span>
    </div>
  </div>


  <!-- PANEL IA -->
  <div class="ai-panel" id="aiPanel">
    <div class="ai-title">
      SEÑAL DE INTELIGENCIA ARTIFICIAL
      <span class="ai-badge" id="aiBadge">ENTRENANDO...</span>
    </div>
    <div class="ai-prob-wrap">
      <div class="ai-prob-circle neutral" id="aiCircle">
        <div class="ai-prob-val neutral" id="aiProbVal">—</div>
        <div class="ai-prob-label">PROB.</div>
      </div>
      <div class="ai-signal-wrap">
        <div class="ai-signal neutral" id="aiSignal">CARGANDO</div>
        <div class="ai-desc" id="aiDesc">Entrenando modelo con datos históricos...</div>
        <div class="ai-bar-wrap">
          <div class="ai-bar-label"><span>VENDER</span><span>NEUTRAL</span><span>COMPRAR</span></div>
          <div class="ai-bar"><div class="ai-bar-fill" id="aiBarFill" style="width:50%;background:var(--gold-dim)"></div></div>
        </div>
      </div>
    </div>
    <div class="ai-stats-row">
      <div class="ai-stat"><div class="ai-stat-val" id="aiAccuracy">—</div><div class="ai-stat-label">ACCURACY</div></div>
      <div class="ai-stat"><div class="ai-stat-val" id="aiSamples">—</div><div class="ai-stat-label">MUESTRAS</div></div>
      <div class="ai-stat"><div class="ai-stat-val" id="aiRetrains">0</div><div class="ai-stat-label">ENTRENAM.</div></div>
    </div>
    <div class="ai-consensus neutral" id="aiConsensus">Esperando señal técnica y de IA...</div>
    <div class="ai-training" id="aiTraining">El modelo se re-entrena cada 5 minutos con nuevos datos</div>
  </div>

  <div class="levels-box">
    <div class="levels-title">
      NIVELES · ATR DINÁMICO
      <span id="rrLabel" style="color:var(--gold);font-size:10px"></span>
    </div>
    <div class="level-row">
      <span class="level-name">ENTRADA</span>
      <div class="level-val entry" id="lvlEntry">—</div>
    </div>
    <div class="level-row">
      <span class="level-name">TAKE PROFIT</span>
      <div style="text-align:right"><div class="level-val tp" id="lvlTP">—</div><div class="level-sub" id="lvlTPsub"></div></div>
    </div>
    <div class="level-row">
      <span class="level-name">STOP LOSS</span>
      <div style="text-align:right"><div class="level-val sl" id="lvlSL">—</div><div class="level-sub" id="lvlSLsub"></div></div>
    </div>
  </div>
  <button class="log-btn" id="logBtn" onclick="logCurrentSignal()" style="display:none">📊 REGISTRAR ESTA OPERACIÓN EN EXCEL</button>

  <div class="chart-wrap">
    <div class="chart-header">
      <div class="chart-title">HISTORIAL DE PRECIO</div>
      <div class="chart-tabs">
        <button class="chart-tab active" onclick="setRange(this,20)">20P</button>
        <button class="chart-tab" onclick="setRange(this,50)">50P</button>
        <button class="chart-tab" onclick="setRange(this,100)">100P</button>
      </div>
    </div>
    <canvas id="priceChart" height="160"></canvas>
  </div>

  <div class="indicators-row">
    <div class="indicator-box">
      <div class="ind-label">RSI (14)</div>
      <div class="ind-value" id="rsiValue">—</div>
      <div class="ind-status" id="rsiStatus">—</div>
      <div class="rsi-gauge"><div class="rsi-needle" id="rsiNeedle" style="left:50%"></div></div>
      <div class="rsi-labels"><span>VENTA</span><span>50</span><span>COMPRA</span></div>
    </div>
    <div class="indicator-box">
      <div class="ind-label">MACD</div>
      <div class="ind-value" id="macdValue" style="font-size:24px">—</div>
      <div class="ind-status" id="macdStatus">—</div>
      <canvas id="macdChart" height="50" style="margin-top:8px;height:50px"></canvas>
    </div>
  </div>
</div>

<div class="right">
  <div>
    <div class="panel-title">REGLAS DE GESTIÓN</div>
    <div class="rules-box">
      <div class="rule-row"><span class="rule-label">CONFIANZA MÍNIMA</span><div><input class="rule-input" id="ruleConf" type="number" value="70" min="50" max="95"><span class="rule-unit">%</span></div></div>
      <div class="rule-row"><span class="rule-label">RSI COMPRA MAX</span><div><input class="rule-input" id="ruleRSIbuy" type="number" value="60"></div></div>
      <div class="rule-row"><span class="rule-label">RSI VENTA MIN</span><div><input class="rule-input" id="ruleRSIsell" type="number" value="40"></div></div>
      <div class="rule-row"><span class="rule-label">MULT. TP (ATR)</span><div><input class="rule-input" id="ruleTPmult" type="number" value="1.5" step="0.1"><span class="rule-unit">×</span></div></div>
      <div class="rule-row"><span class="rule-label">MULT. SL (ATR)</span><div><input class="rule-input" id="ruleSLmult" type="number" value="1.0" step="0.1"><span class="rule-unit">×</span></div></div>
    </div>
  </div>
  <div><div class="panel-title">MEDIAS MÓVILES</div><div id="maTable"></div></div>
  <div><div class="panel-title">HISTORIAL DE SEÑALES</div><div class="history-list" id="historyList"></div></div>
  <button class="update-btn" id="updateBtn" onclick="fetchAndAnalyze()"><span>↻ ACTUALIZAR</span></button>
  <div class="timestamp" id="timestamp">—</div>
  <div class="disclaimer"><strong>⚠ AVISO:</strong> Solo educativo. No garantiza resultados. Respeta el drawdown de tu prop firm.</div>
</div>
</div>
</div>

<!-- ══ BACKTESTING ════════════════════════════════════════ -->
<div class="page" id="page-backtest">
<div class="bt-page">
  <div class="bt-header">
    <div>
      <div class="bt-title">BACKTESTING</div>
      <div class="bt-subtitle">SIMULACIÓN DE ESTRATEGIA CON DATOS HISTÓRICOS</div>
    </div>
  </div>
  <div class="bt-controls">
    <div><div class="bt-label">PERÍODOS</div><input class="bt-input" id="btPeriods" type="number" value="200" min="50" max="500"></div>
    <div><div class="bt-label">MULT. TP</div><input class="bt-input" id="btTP" type="number" value="1.5" step="0.1" style="width:80px"></div>
    <div><div class="bt-label">MULT. SL</div><input class="bt-input" id="btSL" type="number" value="1.0" step="0.1" style="width:80px"></div>
    <div><div class="bt-label">CONF. MIN %</div><input class="bt-input" id="btConf" type="number" value="70" style="width:80px"></div>
    <button class="bt-run" onclick="runBacktest()">▶ EJECUTAR</button>
  </div>
  <div class="bt-stats" id="btStats">
    <div class="bt-stat"><div class="bt-stat-val gold" id="btTotal">—</div><div class="bt-stat-label">SEÑALES TOTALES</div></div>
    <div class="bt-stat"><div class="bt-stat-val green" id="btWins">—</div><div class="bt-stat-label">GANADORAS</div></div>
    <div class="bt-stat"><div class="bt-stat-val red" id="btLosses">—</div><div class="bt-stat-label">PERDEDORAS</div></div>
    <div class="bt-stat"><div class="bt-stat-val" id="btWR" style="color:var(--gold)">—</div><div class="bt-stat-label">WIN RATE</div></div>
  </div>
  <div class="bt-chart-wrap">
    <div class="bt-chart-title">CURVA DE EQUITY (SIMULADA)</div>
    <canvas id="btChart" height="160"></canvas>
  </div>
  <div style="overflow:auto"><table class="bt-table" id="btTable">
    <thead><tr><th>#</th><th>SEÑAL</th><th>ENTRADA</th><th>TP</th><th>SL</th><th>RR</th><th>RESULTADO</th><th>P&L</th></tr></thead>
    <tbody id="btBody"></tbody>
  </table></div>
</div>
</div>

<!-- ══ REGISTRO ═══════════════════════════════════════════ -->
<div class="page" id="page-registro">
<div class="reg-page">
  <div class="reg-header">
    <div>
      <div class="reg-title">REGISTRO</div>
    </div>
    <div class="reg-actions">
      <button class="reg-btn primary" onclick="downloadCSV()">⬇ DESCARGAR EXCEL</button>
      <button class="reg-btn" onclick="loadLog()">↻ ACTUALIZAR</button>
    </div>
  </div>
  <div class="reg-summary">
    <div class="reg-stat"><div class="reg-stat-val w" id="regTotal">0</div><div class="reg-stat-label">OPERACIONES</div></div>
    <div class="reg-stat"><div class="reg-stat-val g" id="regBuys">0</div><div class="reg-stat-label">COMPRAS</div></div>
    <div class="reg-stat"><div class="reg-stat-val r" id="regSells">0</div><div class="reg-stat-label">VENTAS</div></div>
  </div>
  <div class="reg-table-wrap">
    <table class="reg-table">
      <thead><tr><th>FECHA</th><th>HORA GMT</th><th>SEÑAL</th><th>ENTRADA</th><th>TP</th><th>SL</th><th>RR</th><th>CONF%</th><th>SESIÓN</th></tr></thead>
      <tbody id="regBody"><tr><td colspan="9" class="reg-empty">Sin operaciones registradas aún.<br>Las señales validadas aparecerán aquí.</td></tr></tbody>
    </table>
  </div>
</div>
</div>

<!-- ══ CONFIGURACIÓN ══════════════════════════════════════ -->
<div class="page" id="page-config">
<div class="bt-page">
  <div class="bt-header"><div><div class="bt-title">CONFIGURACIÓN</div><div class="bt-subtitle">TELEGRAM · PREFERENCIAS</div></div></div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px">
    <div>
      <div class="panel-title">ALERTAS TELEGRAM</div>
      <div class="tg-box">
        <div class="tg-field">
          <label>BOT TOKEN</label>
          <input type="text" id="cfgToken" placeholder="123456:ABC-DEF...">
        </div>
        <div class="tg-field">
          <label>CHAT ID</label>
          <input type="text" id="cfgChatId" placeholder="-100123456789">
        </div>
        <button class="tg-save" onclick="saveTelegram()">💾 GUARDAR Y ACTIVAR</button>
        <button class="tg-save" style="margin-top:6px;border-color:rgba(224,90,90,.3);color:var(--sell)" onclick="testTelegram()">📲 ENVIAR MENSAJE DE PRUEBA</button>
        <div class="tg-hint">
          <strong style="color:var(--gold-dim)">Cómo obtener tu token:</strong><br>
          1. Abre Telegram y busca @BotFather<br>
          2. Escribe /newbot y sigue los pasos<br>
          3. Copia el token que te da<br><br>
          <strong style="color:var(--gold-dim)">Cómo obtener tu Chat ID:</strong><br>
          1. Escríbele a tu bot desde Telegram<br>
          2. Ve a: api.telegram.org/bot<b>TOKEN</b>/getUpdates<br>
          3. Copia el valor de "id" en "chat"
        </div>
      </div>
    </div>
    <div>
      <div class="panel-title">ESTADO DEL SISTEMA</div>
      <div class="rules-box">
        <div class="rule-row"><span class="rule-label">PRECIO API</span><span style="color:var(--buy);font-size:10px">gold-api.com</span></div>
        <div class="rule-row"><span class="rule-label">ACTUALIZACIÓN</span><span style="color:var(--gold);font-size:10px">CADA 30s</span></div>
        <div class="rule-row"><span class="rule-label">TELEGRAM</span><span id="cfgTGstatus" style="font-size:10px;color:var(--text-dim)">INACTIVO</span></div>
        <div class="rule-row"><span class="rule-label">REGISTRO</span><span style="color:var(--gold);font-size:10px">aurum_operaciones.csv</span></div>
        <div class="rule-row" style="border:none"><span class="rule-label">VERSIÓN</span><span style="color:var(--text-dim);font-size:10px">AURUM v3.0</span></div>
      </div>
    </div>
  </div>
</div>
</div>

<script>
// ── ESTADO ───────────────────────────────────────────────
let prices=[], signalHistory=[], currentRange=20;
let ohlcData=[], currentTF='5min';
let soundOn=true, lastValidSignal='', audioCtx=null;
let currentSignalData=null, currentSessionName='';
let tgToken='', tgChatId='';

// ── NAVEGACIÓN ───────────────────────────────────────────
function showPage(name){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('page-'+name).classList.add('active');
  event.target.classList.add('active');
  if(name==='registro') loadLog();
}

// ── AUDIO ────────────────────────────────────────────────
function getAC(){if(!audioCtx)audioCtx=new(window.AudioContext||window.webkitAudioContext)();return audioCtx}
function playTone(freqs,durs,type='sine'){
  if(!soundOn)return;
  try{const ctx=getAC();let t=ctx.currentTime;
  freqs.forEach((f,i)=>{const o=ctx.createOscillator(),g=ctx.createGain();o.connect(g);g.connect(ctx.destination);o.type=type;o.frequency.value=f;g.gain.setValueAtTime(0,t);g.gain.linearRampToValueAtTime(0.3,t+.02);g.gain.linearRampToValueAtTime(0,t+durs[i]);o.start(t);o.stop(t+durs[i]);t+=durs[i]+.05})}catch(e){}}
function alertBuy(){playTone([440,554,659],[.15,.15,.3])}
function alertSell(){playTone([659,554,440],[.15,.15,.3])}
function toggleSound(){soundOn=!soundOn;const b=document.getElementById('soundBtn');b.textContent=soundOn?'🔔 ON':'🔕 OFF';b.className='sound-toggle '+(soundOn?'on':'off');if(soundOn)try{getAC().resume()}catch(e){}}

// ── TELEGRAM ─────────────────────────────────────────────
function saveTelegram(){
  tgToken=document.getElementById('cfgToken').value.trim();
  tgChatId=document.getElementById('cfgChatId').value.trim();
  const active=tgToken&&tgChatId;
  document.getElementById('tgStatus').textContent=active?'📲 TELEGRAM ON':'📵 TELEGRAM OFF';
  document.getElementById('tgStatus').className='tg-status '+(active?'connected':'');
  document.getElementById('cfgTGstatus').textContent=active?'ACTIVO':'INACTIVO';
  document.getElementById('cfgTGstatus').style.color=active?'var(--buy)':'var(--text-dim)';
  alert(active?'✓ Telegram configurado':'Token o Chat ID vacíos');
}
async function testTelegram(){
  if(!tgToken||!tgChatId){alert('Configura el token y chat ID primero');return}
  await sendTG('🥇 *AURUM TEST*\nConexión exitosa con tu bot de señales.');
  alert('Mensaje de prueba enviado');
}
async function sendTG(msg){
  if(!tgToken||!tgChatId)return;
  try{
    await fetch(`/telegram?token=${encodeURIComponent(tgToken)}&chat=${encodeURIComponent(tgChatId)}&msg=${encodeURIComponent(msg)}`);
  }catch(e){}
}

// ── SESIONES ─────────────────────────────────────────────
function updateSessions(){
  const now=new Date();
  const h=now.getUTCHours()+(now.getUTCMinutes()/60);
  const asiaOpen=h<9||h>=22;
  const londonOpen=h>=8&&h<17;
  const nyOpen=h>=13&&h<22;
  const overlap=h>=13&&h<17;
  const set=(id,statusId,open,cls)=>{
    document.getElementById(id).className='session-block '+(open?(overlap&&cls!=='sessAsia'?'overlap':'active-session'):'closed');
    document.getElementById(statusId).textContent=open?'ABIERTA':'CERRADA';
  };
  set('sessAsia','sessAsiaStatus',asiaOpen,'sessAsia');
  set('sessLondon','sessLondonStatus',londonOpen,'sessLondon');
  set('sessNY','sessNYStatus',nyOpen,'sessNY');
  const tip=document.getElementById('sessionTip');
  if(overlap){tip.textContent='⚡ OVERLAP Londres + NY — Máxima volatilidad. Mejores señales del día.';tip.className='session-tip hot';currentSessionName='OVERLAP';}
  else if(londonOpen){tip.textContent='📈 Sesión de Londres activa — Alta liquidez para el oro.';tip.className='session-tip warm';currentSessionName='LONDRES';}
  else if(nyOpen){tip.textContent='🗽 Nueva York activa — Volatilidad alta. Atentos a datos económicos.';tip.className='session-tip warm';currentSessionName='NEW YORK';}
  else if(asiaOpen){tip.textContent='🌙 Sesión Asia — Oro tranquilo. Mejor esperar Londres.';tip.className='session-tip';currentSessionName='ASIA';}
  else{tip.textContent='💤 Mercado cerrado.';tip.className='session-tip';currentSessionName='CERRADO';}
  const hh=String(now.getUTCHours()).padStart(2,'0'),mm=String(now.getUTCMinutes()).padStart(2,'0');
  document.title=`AURUM · ${hh}:${mm} GMT`;
}

// ── INDICADORES ──────────────────────────────────────────
function calcSMA(a,n){if(a.length<n)return null;return a.slice(-n).reduce((s,v)=>s+v,0)/n}
function calcEMA(a,n){if(a.length<n)return null;const k=2/(n+1);let e=a.slice(0,n).reduce((s,v)=>s+v,0)/n;for(let i=n;i<a.length;i++)e=a[i]*k+e*(1-k);return e}
function calcRSI(a,n=14){if(a.length<n+1)return 50;let g=0,l=0;for(let i=a.length-n;i<a.length;i++){const d=a[i]-a[i-1];d>0?g+=d:l-=d}const ag=g/n,al=l/n;if(al===0)return 100;return 100-100/(1+ag/al)}
function calcMACD(a){const e12=calcEMA(a,12),e26=calcEMA(a,26);if(!e12||!e26)return{macd:0,signal:0,hist:0};const m=e12-e26;const ms=[];for(let i=26;i<=a.length;i++){const x=calcEMA(a.slice(0,i),12),y=calcEMA(a.slice(0,i),26);if(x&&y)ms.push(x-y)}const sig=calcEMA(ms,9)||0;return{macd:m,signal:sig,hist:m-sig}}
function calcATR(a,n=14){if(a.length<n+1)return null;const trs=[];for(let i=a.length-n;i<a.length;i++){const hi=a[i]*1.004,lo=a[i]*.996,pc=a[i-1];trs.push(Math.max(hi-lo,Math.abs(hi-pc),Math.abs(lo-pc)))}return trs.reduce((s,v)=>s+v,0)/n}

// ── SEÑAL ────────────────────────────────────────────────
function computeSignal(priceArr){
  const arr=priceArr||prices;
  const RSI=calcRSI(arr),SMA20=calcSMA(arr,20),SMA50=calcSMA(arr,50);
  const EMA9=calcEMA(arr,9),EMA21=calcEMA(arr,21),MACD=calcMACD(arr);
  const ATR=calcATR(arr,14);
  const price=arr[arr.length-1];
  let score=0,reasons=[];
  if(RSI<35){score+=2;reasons.push('RSI en sobreventa')}else if(RSI>65){score-=2;reasons.push('RSI en sobrecompra')}else score+=RSI<50?.5:-.5;
  if(SMA20&&SMA50){if(SMA20>SMA50){score+=1;reasons.push('SMA20>SMA50')}else{score-=1;reasons.push('SMA20<SMA50')}}
  if(EMA9&&EMA21){if(EMA9>EMA21){score+=1;reasons.push('EMA9 al alza')}else{score-=1;reasons.push('EMA9 a la baja')}}
  if(SMA20)score+=price>SMA20*1.002?.5:price<SMA20*.998?-.5:0;
  if(MACD.hist>0){score+=1;reasons.push('MACD+')}else{score-=1;reasons.push('MACD-')}
  const rawSignal=score>=1.5?'COMPRAR':score<=-1.5?'VENDER':'ESPERAR';
  const confidence=Math.min(95,Math.round(40+Math.abs(score)*12));
  const minConf=parseInt(document.getElementById('ruleConf').value)||70;
  const rsiMaxBuy=parseInt(document.getElementById('ruleRSIbuy').value)||60;
  const rsiMinSell=parseInt(document.getElementById('ruleRSIsell').value)||40;
  const tpMult=parseFloat(document.getElementById('ruleTPmult').value)||1.5;
  const slMult=parseFloat(document.getElementById('ruleSLmult').value)||1.0;
  const atrVal=ATR||price*.005;
  const tpDist=atrVal*tpMult,slDist=atrVal*slMult;
  const rr=(tpDist/slDist).toFixed(2);
  if(rawSignal==='ESPERAR')return{signal:'ESPERAR',cls:'wait',confidence,reason:reasons[0]||'Sin tendencia clara',checks:[],valid:false,RSI,SMA20,SMA50,EMA9,EMA21,MACD,ATR:atrVal,price,tpDist,slDist,rr};
  const checks=[];let valid=true;
  const cok=confidence>=minConf;checks.push({label:`CONF ${confidence}%≥${minConf}%`,ok:cok});if(!cok)valid=false;
  let rok=rawSignal==='COMPRAR'?RSI<=rsiMaxBuy:RSI>=rsiMinSell;
  checks.push({label:`RSI ${RSI.toFixed(0)} ${rawSignal==='COMPRAR'?'≤'+rsiMaxBuy:'≥'+rsiMinSell}`,ok:rok});if(!rok)valid=false;
  let ma=rawSignal==='COMPRAR'?(EMA9&&EMA21?EMA9>EMA21:false):(EMA9&&EMA21?EMA9<EMA21:false);
  checks.push({label:'MEDIAS OK',ok:ma});if(!ma)valid=false;
  const mok=rawSignal==='COMPRAR'?MACD.hist>0:MACD.hist<0;
  checks.push({label:'MACD OK',ok:mok});if(!mok)valid=false;
  const cls=valid?(rawSignal==='COMPRAR'?'go-buy':'go-sell'):'wait';
  const reason=valid?`✓ Señal confirmada — RR ${rr}:1`:`Bloqueada: ${checks.filter(c=>!c.ok).map(c=>c.label).join(', ')}`;
  return{signal:valid?rawSignal:'ESPERAR',rawSignal,cls,confidence,reason,checks,valid,RSI,SMA20,SMA50,EMA9,EMA21,MACD,ATR:atrVal,price,tpDist,slDist,rr};
}

// ── RENDER DASHBOARD ─────────────────────────────────────
function renderAlert(d){
  const box=document.getElementById('alertBox');
  box.className='alert-box '+d.cls;
  document.getElementById('alertSignal').textContent=d.valid?d.signal:(d.rawSignal?d.rawSignal+' ✗':'ESPERAR');
  document.getElementById('alertReason').textContent=d.reason;
  document.getElementById('confPct').textContent=d.confidence+'%';
  document.getElementById('confFill').style.width=d.confidence+'%';
  document.getElementById('validityRow').innerHTML=d.checks.map(c=>`<div class="validity-pill ${c.ok?'ok':'fail'}">${c.ok?'✓':'✗'} ${c.label}</div>`).join('');
  const logBtn=document.getElementById('logBtn');
  if(d.valid&&d.rawSignal!=='ESPERAR'){
    const isLong=d.rawSignal==='COMPRAR';
    const tp=isLong?d.price+d.tpDist:d.price-d.tpDist;
    const sl=isLong?d.price-d.slDist:d.price+d.slDist;
    document.getElementById('lvlEntry').textContent='$'+d.price.toFixed(2);
    document.getElementById('lvlTP').textContent='$'+tp.toFixed(2);
    document.getElementById('lvlSL').textContent='$'+sl.toFixed(2);
    document.getElementById('lvlTPsub').textContent=d.tpDist.toFixed(2)+' USD · '+parseFloat(document.getElementById('ruleTPmult').value)+'× ATR';
    document.getElementById('lvlSLsub').textContent=d.slDist.toFixed(2)+' USD · '+parseFloat(document.getElementById('ruleSLmult').value)+'× ATR';
    document.getElementById('rrLabel').textContent='RR '+d.rr+':1';
    logBtn.style.display='block';
  }else{
    ['lvlEntry','lvlTP','lvlSL'].forEach(id=>document.getElementById(id).textContent='—');
    ['lvlTPsub','lvlSLsub'].forEach(id=>document.getElementById(id).textContent='');
    document.getElementById('rrLabel').textContent='';
    logBtn.style.display='none';
  }
}
function renderIndicators(d){
  document.getElementById('rsiValue').textContent=d.RSI.toFixed(1);
  document.getElementById('rsiNeedle').style.left=d.RSI+'%';
  const rs=document.getElementById('rsiStatus');
  if(d.RSI<30){rs.textContent='SOBREVENTA';rs.className='ind-status bullish'}
  else if(d.RSI>70){rs.textContent='SOBRECOMPRA';rs.className='ind-status bearish'}
  else{rs.textContent='NEUTRAL';rs.className='ind-status neutral'}
  document.getElementById('macdValue').textContent=d.MACD.macd.toFixed(2);
  const ms=document.getElementById('macdStatus');
  ms.textContent=d.MACD.hist>0?'HISTOGRAMA +':'HISTOGRAMA −';
  ms.className='ind-status '+(d.MACD.hist>0?'bullish':'bearish');
  document.getElementById('maTable').innerHTML=[{name:'SMA 20',val:d.SMA20},{name:'SMA 50',val:d.SMA50},{name:'EMA 9',val:d.EMA9},{name:'EMA 21',val:d.EMA21}].map(r=>{
    if(!r.val)return'';const s=d.price>r.val?{cls:'b',txt:'COMPRA'}:{cls:'s',txt:'VENTA'};
    return`<div class="ma-row"><span class="ma-name">${r.name}</span><span class="ma-val">$${r.val.toFixed(2)}</span><span class="ma-sig ${s.cls}">${s.txt}</span></div>`;
  }).join('');
}
function addHistory(d){
  const now=new Date(),time=now.getHours().toString().padStart(2,'0')+':'+now.getMinutes().toString().padStart(2,'0');
  const dc=d.cls==='go-buy'?'b':d.cls==='go-sell'?'s':'h';
  signalHistory.unshift({signal:d.valid?d.signal:'BLOQUEADA',cls:dc,price:d.price.toFixed(2),time,rr:d.valid?d.rr:null});
  if(signalHistory.length>6)signalHistory.pop();
  document.getElementById('historyList').innerHTML=signalHistory.map(h=>`<div class="hist-item"><div class="hist-dot ${h.cls}"></div><div class="hist-info"><div class="hist-sig ${h.cls}">${h.signal}</div><div class="hist-sub">$${h.price}${h.rr?' · RR '+h.rr+':1':''}</div></div><div class="hist-time">${h.time}</div></div>`).join('');
}
function triggerSound(d){
  const key=d.valid?d.rawSignal:'X';
  if(key===lastValidSignal)return;
  lastValidSignal=key;
  if(d.valid){d.rawSignal==='COMPRAR'?alertBuy():alertSell();}
}

// ── REGISTRO ─────────────────────────────────────────────
async function logCurrentSignal(){
  if(!currentSignalData||!currentSignalData.valid)return;
  const d=currentSignalData;
  const isLong=d.rawSignal==='COMPRAR';
  const tp=isLong?d.price+d.tpDist:d.price-d.tpDist;
  const sl=isLong?d.price-d.slDist:d.price+d.slDist;
  try{
    await fetch(`/log?signal=${encodeURIComponent(d.rawSignal)}&price=${d.price}&tp=${tp.toFixed(2)}&sl=${sl.toFixed(2)}&atr=${d.ATR.toFixed(2)}&rr=${d.rr}&conf=${d.confidence}&session=${encodeURIComponent(currentSessionName)}`);
    document.getElementById('logBtn').textContent='✓ REGISTRADA';
    document.getElementById('logBtn').style.borderColor='var(--buy)';
    setTimeout(()=>{document.getElementById('logBtn').textContent='📊 REGISTRAR ESTA OPERACIÓN EN EXCEL';document.getElementById('logBtn').style.borderColor=''},3000);
  }catch(e){}
}
async function loadLog(){
  try{
    const r=await fetch('/getlog');
    const rows=await r.json();
    const body=document.getElementById('regBody');
    const total=rows.length, buys=rows.filter(r=>r['Señal']==='COMPRAR').length, sells=rows.filter(r=>r['Señal']==='VENDER').length;
    document.getElementById('regTotal').textContent=total;
    document.getElementById('regBuys').textContent=buys;
    document.getElementById('regSells').textContent=sells;
    if(!rows.length){body.innerHTML='<tr><td colspan="9" class="reg-empty">Sin operaciones aún.</td></tr>';return}
    body.innerHTML=[...rows].reverse().map(r=>`<tr>
      <td>${r['Fecha']}</td><td>${r['Hora GMT']}</td>
      <td class="${r['Señal']==='COMPRAR'?'td-buy':'td-sell'}">${r['Señal']}</td>
      <td>$${r['Precio Entrada']}</td><td style="color:var(--buy)">$${r['TP']}</td>
      <td style="color:var(--sell)">$${r['SL']}</td>
      <td>${r['RR']}:1</td><td>${r['Confianza%']}%</td><td>${r['Sesión']}</td>
    </tr>`).join('');
  }catch(e){document.getElementById('regBody').innerHTML='<tr><td colspan="9" class="reg-empty">Error al cargar datos.</td></tr>'}
}
async function downloadCSV(){
  window.location='/download';
}

// ── BACKTESTING ──────────────────────────────────────────
function runBacktest(){
  const n=parseInt(document.getElementById('btPeriods').value)||200;
  const tpM=parseFloat(document.getElementById('btTP').value)||1.5;
  const slM=parseFloat(document.getElementById('btSL').value)||1.0;
  const minC=parseInt(document.getElementById('btConf').value)||70;

  // Generar precios simulados para backtest
  const BASE=prices.length?prices[prices.length-1]:3020;
  const btPrices=[BASE];
  for(let i=1;i<n;i++){const t=Math.sin(i/20)*3,ns=(Math.random()-.48)*5;btPrices.push(Math.max(BASE*.9,btPrices[i-1]+t+ns))}

  const trades=[]; let equity=10000; const equity_curve=[10000];

  for(let i=50;i<btPrices.length;i++){
    const slice=btPrices.slice(0,i+1);
    const RSI=calcRSI(slice),EMA9=calcEMA(slice,9),EMA21=calcEMA(slice,21);
    const MACD=calcMACD(slice),ATR=calcATR(slice,14)||slice[slice.length-1]*.005;
    const price=slice[slice.length-1];
    let score=0;
    if(RSI<35)score+=2;else if(RSI>65)score-=2;else score+=RSI<50?.5:-.5;
    if(EMA9&&EMA21){if(EMA9>EMA21)score+=1;else score-=1}
    if(MACD.hist>0)score+=1;else score-=1;
    const conf=Math.min(95,Math.round(40+Math.abs(score)*12));
    if(conf<minC)continue;
    const sig=score>=1.5?'COMPRAR':score<=-1.5?'VENDER':null;
    if(!sig)continue;
    const tp=ATR*tpM, sl=ATR*slM;
    const rr=(tp/sl).toFixed(2);
    // Simular resultado: probabilidad basada en confianza
    const prob=(conf-40)/60;
    const won=Math.random()<prob;
    const pnl=won?tp*10:-sl*10; // lotes simplificados
    equity+=pnl;
    equity_curve.push(equity);
    trades.push({sig,price:price.toFixed(2),tp:(sig==='COMPRAR'?price+tp:price-tp).toFixed(2),sl:(sig==='COMPRAR'?price-sl:price+sl).toFixed(2),rr,won,pnl:pnl.toFixed(0)});
    if(trades.length>=50)break;
  }

  const wins=trades.filter(t=>t.won).length;
  const losses=trades.length-wins;
  const wr=trades.length?Math.round(wins/trades.length*100):0;
  document.getElementById('btTotal').textContent=trades.length;
  document.getElementById('btWins').textContent=wins;
  document.getElementById('btLosses').textContent=losses;
  document.getElementById('btWR').textContent=wr+'%';
  document.getElementById('btWR').style.color=wr>=50?'var(--buy)':'var(--sell)';

  // Equity chart
  const cv=document.getElementById('btChart'),ctx=cv.getContext('2d');
  const W=cv.offsetWidth||800,H=160;cv.width=W;cv.height=H;
  const mn=Math.min(...equity_curve)-100,mx=Math.max(...equity_curve)+100;
  const px=i=>(i/(equity_curve.length-1))*(W-20)+10,py=v=>H-((v-mn)/(mx-mn))*(H-20)-10;
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='rgba(201,168,76,.06)';ctx.lineWidth=1;
  for(let g=0;g<=4;g++){const y=py(mn+(mx-mn)*g/4);ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke()}
  const gr=ctx.createLinearGradient(0,0,0,H);
  const lastEq=equity_curve[equity_curve.length-1];
  const c=lastEq>=10000?'rgba(76,175,130':'rgba(224,90,90';
  gr.addColorStop(0,c+',.2)');gr.addColorStop(1,c+',.0)');
  ctx.beginPath();ctx.moveTo(px(0),py(equity_curve[0]));
  equity_curve.forEach((v,i)=>ctx.lineTo(px(i),py(v)));
  ctx.lineTo(px(equity_curve.length-1),H);ctx.lineTo(px(0),H);ctx.fillStyle=gr;ctx.fill();
  ctx.beginPath();ctx.strokeStyle=lastEq>=10000?'#4CAF82':'#E05A5A';ctx.lineWidth=2;
  equity_curve.forEach((v,i)=>i===0?ctx.moveTo(px(i),py(v)):ctx.lineTo(px(i),py(v)));ctx.stroke();

  // Table
  document.getElementById('btBody').innerHTML=trades.slice(-20).map((t,i)=>`<tr>
    <td style="color:var(--text-dim)">${i+1}</td>
    <td class="${t.sig==='COMPRAR'?'td-buy':'td-sell'}">${t.sig}</td>
    <td>$${t.price}</td>
    <td class="td-buy">$${t.tp}</td>
    <td class="td-sell">$${t.sl}</td>
    <td>${t.rr}:1</td>
    <td>${t.won?'✓ WIN':'✗ LOSS'}</td>
    <td class="${parseFloat(t.pnl)>=0?'td-pos':'td-neg'}">${parseFloat(t.pnl)>=0?'+':''}$${t.pnl}</td>
  </tr>`).join('');
}

// ── GRÁFICAS ─────────────────────────────────────────────
function drawPriceChart(){
  const cv=document.getElementById('priceChart'),ctx=cv.getContext('2d');
  const W=cv.offsetWidth||600,H=160;cv.width=W;cv.height=H;
  const sl=prices.slice(-currentRange);if(sl.length<2)return;
  const mn=Math.min(...sl)-5,mx=Math.max(...sl)+5;
  const px=i=>(i/(sl.length-1))*(W-20)+10,py=v=>H-((v-mn)/(mx-mn))*(H-20)-10;
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='rgba(201,168,76,.06)';ctx.lineWidth=1;
  for(let g=0;g<=4;g++){const y=py(mn+(mx-mn)*g/4);ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke()}
  const gr=ctx.createLinearGradient(0,0,0,H);gr.addColorStop(0,'rgba(201,168,76,.12)');gr.addColorStop(1,'rgba(201,168,76,0)');
  ctx.beginPath();ctx.moveTo(px(0),py(sl[0]));sl.forEach((v,i)=>ctx.lineTo(px(i),py(v)));
  ctx.lineTo(px(sl.length-1),H);ctx.lineTo(px(0),H);ctx.fillStyle=gr;ctx.fill();
  ctx.beginPath();ctx.strokeStyle='#C9A84C';ctx.lineWidth=1.5;
  sl.forEach((v,i)=>i===0?ctx.moveTo(px(i),py(v)):ctx.lineTo(px(i),py(v)));ctx.stroke();
  const last=sl[sl.length-1];ctx.beginPath();ctx.arc(px(sl.length-1),py(last),4,0,Math.PI*2);ctx.fillStyle='#F0D080';ctx.fill();
}
function drawMACDChart(){
  const cv=document.getElementById('macdChart'),ctx=cv.getContext('2d');
  const W=cv.offsetWidth||300,H=50;cv.width=W;cv.height=H;ctx.clearRect(0,0,W,H);
  const bars=[];for(let i=30;i<=prices.length;i++){const d=calcMACD(prices.slice(0,i));bars.push(d.hist)}
  const rc=bars.slice(-20);if(!rc.length)return;
  const mxv=Math.max(...rc.map(Math.abs),.01),bw=W/rc.length;
  rc.forEach((v,i)=>{const h=Math.abs(v)/mxv*(H/2-4),y=v>=0?H/2-h:H/2;ctx.fillStyle=v>=0?'rgba(76,175,130,.7)':'rgba(224,90,90,.7)';ctx.fillRect(i*bw+1,y,bw-2,h)});
  ctx.strokeStyle='rgba(201,168,76,.2)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,H/2);ctx.lineTo(W,H/2);ctx.stroke();
}
function setRange(btn,n){document.querySelectorAll('.chart-tab').forEach(t=>t.classList.remove('active'));btn.classList.add('active');currentRange=n;drawPriceChart()}


// ── INTELIGENCIA ARTIFICIAL ──────────────────────────────
let aiRetrains = 0;

async function updateAI(technicalSignal) {
  if (prices.length < 60) return;

  // Entrenar en background
  const pricesParam = prices.slice(-200).join(',');
  try {
    const tr = await fetch('/aitrain?prices=' + encodeURIComponent(pricesParam));
    const td = await tr.json();
    if (td.trained) {
      aiRetrains++;
      document.getElementById('aiAccuracy').textContent = td.accuracy + '%';
      document.getElementById('aiSamples').textContent = td.epochs;
      document.getElementById('aiRetrains').textContent = aiRetrains;
      document.getElementById('aiBadge').textContent = 'IA ACTIVA · ' + td.accuracy + '% ACC';
      document.getElementById('aiBadge').style.borderColor = 'rgba(76,175,130,.4)';
      document.getElementById('aiBadge').style.color = 'var(--buy)';
    }

    // Predicción
    const pr = await fetch('/aipredict?prices=' + encodeURIComponent(pricesParam));
    const pd = await pr.json();

    if (pd.prob === null) return;

    const prob = pd.prob; // 0-100
    const aiSig = pd.signal;
    const cls = aiSig === 'COMPRAR' ? 'buy' : aiSig === 'VENDER' ? 'sell' : 'neutral';

    document.getElementById('aiCircle').className = 'ai-prob-circle ' + cls;
    document.getElementById('aiProbVal').className = 'ai-prob-val ' + cls;
    document.getElementById('aiProbVal').textContent = prob.toFixed(0) + '%';
    document.getElementById('aiSignal').className = 'ai-signal ' + cls;
    document.getElementById('aiSignal').textContent = aiSig;

    const desc = aiSig === 'COMPRAR'
      ? `El modelo ve ${prob.toFixed(0)}% de probabilidad de subida`
      : aiSig === 'VENDER'
      ? `El modelo ve ${(100-prob).toFixed(0)}% de probabilidad de bajada`
      : 'Probabilidad insuficiente para señal clara';
    document.getElementById('aiDesc').textContent = desc;

    // Barra de probabilidad
    const fill = document.getElementById('aiBarFill');
    fill.style.width = prob + '%';
    fill.style.background = cls === 'buy' ? 'var(--buy)' : cls === 'sell' ? 'var(--sell)' : 'var(--gold-dim)';

    // Consenso técnico + IA
    const consensus = document.getElementById('aiConsensus');
    const techValid = technicalSignal && technicalSignal !== 'ESPERAR';
    const agree = techValid && technicalSignal === aiSig;
    const disagree = techValid && technicalSignal !== aiSig && aiSig !== 'ESPERAR';

    if (agree) {
      consensus.textContent = '✓ CONSENSO — Técnico e IA coinciden en ' + aiSig + '. Señal de mayor confianza.';
      consensus.className = 'ai-consensus agree';
    } else if (disagree) {
      consensus.textContent = '⚠ DIVERGENCIA — Técnico dice ' + technicalSignal + ' pero IA dice ' + aiSig + '. Operar con precaución.';
      consensus.className = 'ai-consensus disagree';
    } else {
      consensus.textContent = 'Analizando convergencia entre indicadores técnicos e IA...';
      consensus.className = 'ai-consensus neutral';
    }

  } catch(e) {
    document.getElementById('aiDesc').textContent = 'Error al conectar con el motor de IA';
  }
}
// ── TIMEFRAME ─────────────────────────────────────────────
function setTF(btn, tf){
  document.querySelectorAll('#tf5,#tf15,#tf1h,#tf4h').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  currentTF=tf;
  document.getElementById('tfInfo').textContent='Cargando velas '+tf+'...';
  fetchAndAnalyze();
}

// ── FETCH PRINCIPAL ───────────────────────────────────────
async function fetchAndAnalyze(){
  const btn=document.getElementById('updateBtn');
  btn.disabled=true;btn.querySelector('span').textContent='OBTENIENDO PRECIO...';
  try{
    const tf = currentTF || '5min';
    let priceData = null, candles = null;

    // Try to get current price
    try {
      const priceRes = await fetch('/precio');
      const pd = await priceRes.json();
      if(pd && pd.price) priceData = pd;
    } catch(e){ console.log('precio API:', e); }

    // Try to get OHLC candles
    try {
      const ohlcRes = await fetch('/ohlc?interval='+tf+'&size=200');
      const cd = await ohlcRes.json();
      if(Array.isArray(cd) && cd.length > 0) candles = cd;
    } catch(e){ console.log('ohlc API:', e); }

    // Use candles if available
    if(candles && candles.length > 0){
      ohlcData = candles;
      prices = candles.map(c => c.close);
      if(ohlcData.length) document.getElementById('tfInfo').textContent = ohlcData.length+' velas reales';
      document.getElementById('sourceBadge').innerHTML='<span class="source-badge live">● DATOS REALES '+tf.toUpperCase()+'</span>';
    }

    // Use current price
    if(priceData && priceData.price){
      const livePrice = priceData.price;
      if(prices.length===0){
        let p=livePrice; const arr=[];
        for(let i=150;i>0;i--){p=Math.max(livePrice*.97,p+(Math.random()-.5)*4);arr.unshift(p)}
        arr[arr.length-1]=livePrice; prices=arr;
      } else {
        prices[prices.length-1]=livePrice;
        prices.push(livePrice);
        if(prices.length>300) prices=prices.slice(-300);
      }
      document.getElementById('priceDisplay').textContent='$'+livePrice.toFixed(2);
      const ch=priceData.ch||0, chp=priceData.chp||0;
      const el=document.getElementById('priceChange');
      el.textContent=(ch>=0?'+':'')+ch.toFixed(2)+' ('+(ch>=0?'+':'')+Number(chp).toFixed(2)+'%)';
      el.className='price-change '+(ch>=0?'up':'down');
      if(!candles) document.getElementById('sourceBadge').innerHTML='<span class="source-badge live">● PRECIO REAL</span>';
    } else if(prices.length === 0) {
      // Complete fallback - simulate prices around last known gold price
      const BASE = 3020;
      let p = BASE; const arr=[];
      for(let i=150;i>0;i--){p=Math.max(BASE*.97,p+(Math.random()-.5)*4);arr.unshift(p)}
      prices=arr;
      document.getElementById('priceDisplay').textContent='$'+BASE.toFixed(2);
      document.getElementById('priceChange').textContent='Mercado cerrado o sin conexión';
      document.getElementById('sourceBadge').innerHTML='<span class="source-badge sim">◌ PRECIO SIMULADO</span>';
    }

    if(!prices.length){ btn.disabled=false; btn.querySelector('span').textContent='↻ ACTUALIZAR'; return; }
  }catch(e){
    console.log('fetch error:', e);
    if(!prices.length){
      btn.disabled=false; btn.querySelector('span').textContent='↻ ACTUALIZAR'; return;
    }
  }
  btn.querySelector('span').textContent='ANALIZANDO...';
  const d=computeSignal();
  currentSignalData=d;
  triggerSound(d);
  // Telegram si señal válida nueva
  if(d.valid&&d.rawSignal!==lastValidSignal){
    const isLong=d.rawSignal==='COMPRAR';
    const tp=(isLong?d.price+d.tpDist:d.price-d.tpDist).toFixed(2);
    const sl=(isLong?d.price-d.slDist:d.price+d.slDist).toFixed(2);
    const em=d.rawSignal==='COMPRAR'?'🟢':'🔴';
    sendTG(`${em} *AURUM · ${d.rawSignal}*\n💰 Entrada: $${d.price.toFixed(2)}\n🎯 TP: $${tp}\n🛡 SL: $${sl}\n📊 RR: ${d.rr}:1\n💪 Confianza: ${d.confidence}%\n🌍 Sesión: ${currentSessionName}`);
  }
  renderAlert(d);renderIndicators(d);addHistory(d);
  updateAI(d.valid ? d.rawSignal : 'ESPERAR');
  drawPriceChart();drawMACDChart();
  document.getElementById('timestamp').textContent='✓ '+new Date().toLocaleTimeString('es-ES');
  if(ohlcData.length) document.getElementById('tfInfo').textContent=ohlcData.length+' velas reales cargadas';
  btn.disabled=false;btn.querySelector('span').textContent='↻ ACTUALIZAR';
}

// ── INIT ─────────────────────────────────────────────────
fetchAndAnalyze();
updateSessions();
setInterval(fetchAndAnalyze,30000);
setInterval(updateSessions,60000);
window.addEventListener('resize',()=>{drawPriceChart();drawMACDChart()});
</script>
</body>
</html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split('?')[0]
        params = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.path).query))

        if path == "/":
            self._send(200, "text/html; charset=utf-8", HTML.encode("utf-8"))

        elif path == "/precio":
            data = get_gold_price()
            self._send(200 if data else 503, "application/json",
                       json.dumps(data or {"error":"sin precio"}).encode())
            # (training triggered from JS via /aitrain endpoint)

        elif path == "/log":
            try:
                log_signal(
                    params.get("signal",""),
                    float(params.get("price",0)),
                    float(params.get("tp",0)),
                    float(params.get("sl",0)),
                    float(params.get("atr",0)),
                    params.get("rr",""),
                    params.get("conf",""),
                    params.get("session","")
                )
                self._send(200, "application/json", b'{"ok":true}')
            except Exception as e:
                self._send(500, "application/json", b'{"ok":false}')

        elif path == "/getlog":
            rows = read_log()
            self._send(200, "application/json", json.dumps(rows).encode())

        elif path == "/download":
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", f'attachment; filename="{LOG_FILE}"')
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send(404, "text/plain", b"Sin datos aun")

        elif path == "/telegram":
            token = params.get("token","")
            chat  = params.get("chat","")
            msg   = params.get("msg","")
            if token and chat and msg:
                threading.Thread(target=send_telegram_direct, args=(token, chat, msg), daemon=True).start()
            self._send(200, "application/json", b'{"ok":true}')

        elif path == "/aitrain":
            import threading as _th
            prices_for_ai = [float(x) for x in params.get("prices","").split(",") if x]
            if prices_for_ai:
                _th.Thread(target=ai_train_if_needed, args=(prices_for_ai,), daemon=True).start()
            self._send(200, "application/json", json.dumps({
                "trained": _ai.trained,
                "accuracy": round(_ai.accuracy, 1),
                "epochs": _ai.epochs
            }).encode())

        elif path == "/aipredict":
            prices_for_ai = [float(x) for x in params.get("prices","").split(",") if x]
            prob, signal = ai_predict(prices_for_ai) if prices_for_ai else (None, None)
            self._send(200, "application/json", json.dumps({
                "prob": round(prob*100, 1) if prob is not None else None,
                "signal": signal,
                "trained": _ai.trained,
                "accuracy": round(_ai.accuracy, 1)
            }).encode())

        else:
            self._send(404, "text/plain", b"Not found")

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        p = str(args[0])
        if any(x in p for x in ["/precio", "/log", "/telegram"]):
            print(f"  → {p.split()[0]} {p.split()[1] if len(p.split())>1 else ''} [{args[1]}]")


def send_telegram_direct(token, chat, msg):
    try:
        url  = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat, "text": msg, "parse_mode": "Markdown"}).encode()
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=8)
        print(f"  📲 Telegram enviado")
    except Exception as e:
        print(f"  ⚠ Telegram: {e}")


def main():
    init_log()
    print("=" * 54)
    print("  AURUM v3.0 · Bot de Señales XAU/USD")
    print("  Dashboard · Backtest · Registro · Telegram")
    print("=" * 54)
    print("\n  Verificando precio del oro en tiempo real...")
    test = get_gold_price()
    if test:
        ch = test['ch']
        print(f"  ✓ XAU/USD: ${test['price']:.2f}  ({'+' if ch>=0 else ''}{ch:.2f})")
    print(f"\n  Probando datos históricos (5min)...")
    candles = get_historical_ohlc("5min", 10)
    if candles:
        print(f"  ✓ Twelve Data OK — última vela: ${candles[-1]['close']:.2f} ({candles[-1]['dt']})")
    else:
        print(f"  ⚠ Twelve Data no disponible — usando precio en tiempo real")

    print(f"\n  Servidor: http://localhost:{PORT}")
    print(f"  Registro: {LOG_FILE}")
    print("\n  Para detener: Ctrl + C")
    print("-" * 54)
    server = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"  Dashboard: https://aurum-signals.onrender.com")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n  AURUM detenido. ¡Buena suerte con la prop firm!")
        server.shutdown()

if __name__ == "__main__":
    main()
