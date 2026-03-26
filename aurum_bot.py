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


# ── FILTRO DE NOTICIAS ECONÓMICAS ────────────────────────
import datetime as _dt

# Horarios fijos de noticias de alto impacto (GMT)
# Se actualiza manualmente o via API de calendario económico
HIGH_IMPACT_EVENTS = [
    # Formato: (día_semana 0=lun, hora_gmt, minutos, nombre)
    # Estos son ejemplos de horarios típicos — actualizar semanalmente
    (1, 13, 30, "CPI USA"),       # Martes 13:30 GMT aprox
    (2, 14, 0,  "FOMC Minutes"),  # Miércoles 14:00 GMT aprox  
    (4, 13, 30, "NFP / Jobs"),    # Viernes 13:30 GMT aprox
    (4, 13, 30, "Unemployment"),  # Viernes 13:30 GMT aprox
]

PAUSE_MINUTES_BEFORE = 30  # pausar 30 min antes
PAUSE_MINUTES_AFTER  = 30  # pausar 30 min después

def is_news_time():
    """Retorna (True, nombre_noticia) si estamos cerca de noticia de alto impacto"""
    now = _dt.datetime.utcnow()
    dow = now.weekday()  # 0=lunes
    for (day, hour, minute, name) in HIGH_IMPACT_EVENTS:
        if dow != day:
            continue
        event_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        diff = (now - event_time).total_seconds() / 60  # minutos desde evento
        if -PAUSE_MINUTES_BEFORE <= diff <= PAUSE_MINUTES_AFTER:
            return True, name
    return False, None

# ── MOTOR DE IA v2 — ENSEMBLE (Pure Python, sin librerías) ──
import math, pickle, os, time as _time

class LogisticModel:
    """Regresión logística con gradiente descendente"""
    def __init__(self, n_features=8):
        self.w = [0.0] * n_features
        self.b = 0.0
        self.trained = False

    @staticmethod
    def sigmoid(x):
        x = max(-500, min(500, x))
        return 1 / (1 + math.exp(-x))

    def predict(self, x):
        return self.sigmoid(sum(w*xi for w,xi in zip(self.w,x)) + self.b)

    def train(self, X, y, lr=0.05, epochs=300, l2=0.01):
        n = len(X)
        if n < 10: return
        for _ in range(epochs):
            dw = [0.0]*len(self.w); db = 0.0
            for xi,yi in zip(X,y):
                e = self.predict(xi) - yi
                for j in range(len(self.w)): dw[j] += e*xi[j]
                db += e
            for j in range(len(self.w)):
                self.w[j] -= lr*(dw[j]/n + l2*self.w[j])
            self.b -= lr*db/n
            lr *= 0.997
        self.trained = True


class DecisionTreeModel:
    """Árbol de decisión simple — simula Random Forest con múltiples features"""
    def __init__(self):
        self.thresholds = {}
        self.trained = False

    def train(self, X, y):
        if len(X) < 10: return
        n_features = len(X[0])
        for fi in range(n_features):
            vals = [x[fi] for x in X]
            best_thresh = sum(vals) / len(vals)  # media como umbral
            self.thresholds[fi] = best_thresh
        self.trained = True

    def predict(self, x):
        if not self.trained: return 0.5
        score = 0.0
        weights = [0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1]
        for fi, (thresh, w) in enumerate(zip(self.thresholds.values(), weights)):
            if fi < len(x):
                score += w * (1.0 if x[fi] > thresh else 0.0)
        return score


class AurumAI:
    """
    Ensemble de modelos IA — Pure Python sin dependencias externas.
    Combina Regresión Logística + Árbol de Decisión para señales más robustas.
    Detecta drift de mercado para pausar señales cuando cambia el comportamiento.
    """
    MODEL_FILE = "aurum_model_v2.pkl"

    def __init__(self):
        self.logistic    = LogisticModel(n_features=8)
        self.tree        = DecisionTreeModel()
        self.trained     = False
        self.accuracy    = 0.0
        self.epochs      = 0
        self.drift_score = 0.0   # 0=estable, >0.5=drift detectado
        self.accuracy_by_hour  = {}  # {hora: accuracy}
        self.accuracy_by_dow   = {}  # {dia_semana: accuracy}
        self.recent_errors     = []  # últimos 20 errores para drift
        self.load()

    def save(self):
        try:
            data = {
                "logistic_w": self.logistic.w,
                "logistic_b": self.logistic.b,
                "logistic_trained": self.logistic.trained,
                "tree_thresh": self.tree.thresholds,
                "tree_trained": self.tree.trained,
                "trained": self.trained,
                "accuracy": self.accuracy,
                "epochs": self.epochs,
                "drift_score": self.drift_score,
                "accuracy_by_hour": self.accuracy_by_hour,
                "accuracy_by_dow":  self.accuracy_by_dow,
            }
            with open(AurumAI.MODEL_FILE, "wb") as f:
                pickle.dump(data, f)
            print(f"  💾 Ensemble guardado (acc: {self.accuracy:.1f}%)")
        except Exception as e:
            print(f"  ⚠ Save error: {e}")

    def load(self):
        try:
            if not os.path.exists(AurumAI.MODEL_FILE): return
            with open(AurumAI.MODEL_FILE, "rb") as f:
                d = pickle.load(f)
            self.logistic.w       = d.get("logistic_w", [0.0]*8)
            self.logistic.b       = d.get("logistic_b", 0.0)
            self.logistic.trained = d.get("logistic_trained", False)
            self.tree.thresholds  = d.get("tree_thresh", {})
            self.tree.trained     = d.get("tree_trained", False)
            self.trained          = d.get("trained", False)
            self.accuracy         = d.get("accuracy", 0.0)
            self.epochs           = d.get("epochs", 0)
            self.drift_score      = d.get("drift_score", 0.0)
            self.accuracy_by_hour = d.get("accuracy_by_hour", {})
            self.accuracy_by_dow  = d.get("accuracy_by_dow", {})
            print(f"  ✓ Ensemble cargado (acc: {self.accuracy:.1f}%, {self.epochs} muestras)")
        except Exception as e:
            print(f"  ⚠ Load error: {e}")

    def extract_features(self, prices):
        """8 features normalizadas para el ensemble"""
        if len(prices) < 35: return None
        p = prices

        # RSI normalizado
        n = 14; g = l = 0
        for i in range(len(p)-n, len(p)):
            d = p[i]-p[i-1]
            if d>0: g+=d
            else: l-=d
        ag,al = g/n, l/n
        rsi = (100-100/(1+ag/al))/100 if al>0 else 1.0

        # EMA cross
        def ema(arr, n):
            k=2/(n+1); e=sum(arr[:n])/n
            for v in arr[n:]: e=v*k+e*(1-k)
            return e
        e9,e21 = ema(p,9), ema(p,21)
        ema_cross = max(-1, min(1, (e9-e21)/(e21+1e-9)*100))

        # MACD
        e12,e26 = ema(p,12), ema(p,26)
        macd_norm = max(-1, min(1, (e12-e26)/(p[-1]*0.01+1e-9)))

        # ATR
        atrs = [max(p[i]*1.004-p[i]*0.996,
                    abs(p[i]*1.004-p[i-1]),
                    abs(p[i]*0.996-p[i-1]))
                for i in range(len(p)-14, len(p))]
        atr_norm = max(0, min(2, (sum(atrs)/14)/(p[-1]*0.01+1e-9)))

        # Price vs SMA20
        sma20 = sum(p[-20:])/20
        pvs = max(-2, min(2, (p[-1]-sma20)/(sma20+1e-9)*100))

        # Momentum 5
        mom5 = max(-2, min(2, (p[-1]-p[-6])/(p[-6]+1e-9)*100))

        # Momentum 10
        mom10 = max(-2, min(2, (p[-1]-p[-11])/(p[-11]+1e-9)*100)) if len(p)>11 else 0

        # Volatility (std of last 10 returns)
        rets = [(p[i]-p[i-1])/p[i-1] for i in range(len(p)-10,len(p))]
        mean_r = sum(rets)/len(rets)
        vol = math.sqrt(sum((r-mean_r)**2 for r in rets)/len(rets)) * 100
        vol_norm = max(0, min(2, vol))

        return [rsi, ema_cross, macd_norm, atr_norm, pvs, mom5, mom10, vol_norm]

    def detect_drift(self):
        """Detecta si el comportamiento del mercado ha cambiado"""
        if len(self.recent_errors) < 10: return 0.0
        recent = self.recent_errors[-10:]
        error_rate = sum(recent) / len(recent)
        self.drift_score = error_rate
        return error_rate

    def is_market_drifting(self):
        """True si el mercado ha cambiado y las señales deben pausarse"""
        return self.detect_drift() > 0.65  # más del 65% de errores recientes

    def train(self, prices):
        if len(prices) < 60: return False
        import datetime as _dt
        X, y, hours, dows = [], [], [], []
        now = _dt.datetime.utcnow()

        for i in range(40, len(prices)-5):
            features = self.extract_features(prices[i-35:i])
            if not features: continue
            future_ret = (prices[i+4]-prices[i])/prices[i]
            label = 1 if future_ret > 0.0005 else 0
            # Estimate hour/dow (approximate)
            mins_ago = (len(prices)-i)*5
            t = now - _dt.timedelta(minutes=mins_ago)
            X.append(features); y.append(label)
            hours.append(t.hour); dows.append(t.weekday())

        if len(X) < 20: return False

        # Train both models
        self.logistic.train(X, y)
        self.tree.train(X, y)

        # Ensemble accuracy
        correct = 0
        for xi,yi,h,dow in zip(X,y,hours,dows):
            p_log = self.logistic.predict(xi)
            p_tree = self.tree.predict(xi)
            p_ens = 0.6*p_log + 0.4*p_tree  # weighted ensemble
            pred = 1 if p_ens >= 0.5 else 0
            is_correct = int(pred == yi)
            correct += is_correct
            self.recent_errors.append(1-is_correct)

            # Track by hour
            hk = str(h)
            if hk not in self.accuracy_by_hour:
                self.accuracy_by_hour[hk] = []
            self.accuracy_by_hour[hk].append(is_correct)
            if len(self.accuracy_by_hour[hk]) > 50:
                self.accuracy_by_hour[hk] = self.accuracy_by_hour[hk][-50:]

            # Track by day of week
            dk = str(dow)
            if dk not in self.accuracy_by_dow:
                self.accuracy_by_dow[dk] = []
            self.accuracy_by_dow[dk].append(is_correct)
            if len(self.accuracy_by_dow[dk]) > 50:
                self.accuracy_by_dow[dk] = self.accuracy_by_dow[dk][-50:]

        if len(self.recent_errors) > 50:
            self.recent_errors = self.recent_errors[-50:]

        self.accuracy = correct/len(X)*100
        self.epochs   = len(X)
        self.trained  = True
        self.save()
        return True

    def predict_proba(self, prices):
        """Retorna probabilidad ensemble [0,1]"""
        if not self.trained: return None
        features = self.extract_features(prices[-35:])
        if not features: return None
        p_log  = self.logistic.predict(features)
        p_tree = self.tree.predict(features)
        return 0.6*p_log + 0.4*p_tree

    def get_hour_accuracy(self):
        """Accuracy promedio para la hora actual"""
        import datetime as _dt
        h = str(_dt.datetime.utcnow().hour)
        if h in self.accuracy_by_hour and self.accuracy_by_hour[h]:
            data = self.accuracy_by_hour[h]
            return sum(data)/len(data)*100
        return None

    def get_dow_accuracy(self):
        """Accuracy promedio para el día de la semana actual"""
        import datetime as _dt
        d = str(_dt.datetime.utcnow().weekday())
        if d in self.accuracy_by_dow and self.accuracy_by_dow[d]:
            data = self.accuracy_by_dow[d]
            return sum(data)/len(data)*100
        return None


# Instancia global del modelo
_ai = AurumAI()
_ai_last_train = 0
_ai_last_backup = 0  # para backup cada hora

def ai_train_if_needed(prices):
    global _ai_last_train, _ai_last_backup
    now = _time.time()
    # Re-entrenar cada 5 minutos
    if now - _ai_last_train > 300 and len(prices) >= 60:
        _ai.train(prices)
        _ai_last_train = now
        if _ai.trained:
            drift = _ai.detect_drift()
            h_acc = _ai.get_hour_accuracy()
            d_acc = _ai.get_dow_accuracy()
            print(f"  🤖 Ensemble: {_ai.accuracy:.1f}% acc | Drift: {drift:.2f} | {_ai.epochs} muestras")
            if h_acc: print(f"  🕐 Acc hora actual: {h_acc:.1f}%")
            if d_acc: print(f"  📅 Acc hoy: {d_acc:.1f}%")
            if _ai.is_market_drifting():
                print("  ⚠ DRIFT DETECTADO — señales pausadas temporalmente")
    # Backup cada hora
    if now - _ai_last_backup > 3600 and _ai.trained:
        _ai.save()
        _ai_last_backup = now
        print("  💾 Backup automático del modelo guardado")

def ai_predict(prices):
    """Retorna (prob, señal) del ensemble. None si hay drift."""
    if not _ai.trained or len(prices) < 35:
        return None, None
    if _ai.is_market_drifting():
        return None, "DRIFT"  # señal especial de drift
    prob = _ai.predict_proba(prices)
    if prob is None: return None, None
    if prob >= 0.66:   signal = "COMPRAR"
    elif prob <= 0.34: signal = "VENDER"
    else:              signal = "ESPERAR"
    return prob, signal


import os
PORT = int(os.environ.get("PORT", 8765))
LOG_FILE = "aurum_operaciones.csv"

# ── TELEGRAM ─────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

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


# ── CONFIRMACIÓN MULTI-TIMEFRAME ─────────────────────────
_mtf_cache = {"prices_1h": [], "last_update": 0}

def get_htf_trend():
    """
    Obtiene tendencia del timeframe mayor (1H).
    Retorna: 'up', 'down', o 'neutral'
    """
    import time
    now = time.time()
    # Cache por 5 minutos para no pedir la API cada vez
    if now - _mtf_cache["last_update"] < 300 and _mtf_cache["prices_1h"]:
        prices_1h = _mtf_cache["prices_1h"]
    else:
        candles = get_historical_ohlc("1h", 50)
        if not candles or len(candles) < 21:
            return "neutral"
        prices_1h = [c["close"] for c in candles]
        _mtf_cache["prices_1h"] = prices_1h
        _mtf_cache["last_update"] = now

    # EMA 9 vs EMA 21 en 1H
    def ema(arr, n):
        if len(arr) < n: return None
        k = 2/(n+1)
        e = sum(arr[:n])/n
        for v in arr[n:]: e = v*k + e*(1-k)
        return e

    e9  = ema(prices_1h, 9)
    e21 = ema(prices_1h, 21)
    rsi_1h = 50  # simplified

    if e9 and e21:
        if e9 > e21 * 1.001:
            return "up"
        elif e9 < e21 * 0.999:
            return "down"
    return "neutral"


# ── GESTIÓN DE RIESGO INTELIGENTE ────────────────────────
_consecutive_losses = 0
_consecutive_wins   = 0
_daily_signals      = 0
_last_signal_date   = ""
MAX_CONSECUTIVE_LOSSES = 3
MAX_DAILY_SIGNALS      = 8

def calc_position_size(price, atr, account_pct=0.01):
    """
    Tamaño de posición dinámico basado en volatilidad.
    account_pct = % del capital a arriesgar por operación (default 1%)
    Retorna el multiplicador de lote sugerido.
    """
    if atr <= 0 or price <= 0: return 1.0
    atr_pct = atr / price * 100
    # Reducir tamaño si alta volatilidad
    if atr_pct > 0.8:   size = 0.5   # alta volatilidad -> mitad de lote
    elif atr_pct > 0.5: size = 0.75  # media volatilidad -> 3/4 de lote
    else:               size = 1.0   # baja volatilidad -> lote completo
    return size

def calc_adaptive_levels(price, atr, signal, trend_strength):
    """
    TP y SL adaptativos según tendencia y volatilidad.
    trend_strength: 0-1 (fuerza de la tendencia)
    """
    # Más TP cuando tendencia fuerte
    tp_mult = 1.5 + trend_strength * 0.5  # 1.5x a 2.0x
    sl_mult = 1.0                           # SL fijo en 1x ATR

    is_buy = signal == "COMPRAR"
    tp = price + atr * tp_mult if is_buy else price - atr * tp_mult
    sl = price - atr * sl_mult if is_buy else price + atr * sl_mult
    rr = tp_mult / sl_mult
    return tp, sl, round(rr, 2), round(tp_mult, 2)

def is_lateral_market(prices, n=20):
    """
    Detecta si el mercado está en rango lateral (evitar señales en ruido).
    Retorna True si el mercado está quieto.
    """
    if len(prices) < n: return False
    recent = prices[-n:]
    high, low = max(recent), min(recent)
    range_pct = (high - low) / low * 100
    return range_pct < 0.3  # menos del 0.3% de rango = lateral

def check_risk_limits():
    """
    Verifica si se pueden tomar más señales según límites de riesgo.
    Retorna (bool, razón)
    """
    import datetime as _dt
    global _daily_signals, _last_signal_date
    today = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    if today != _last_signal_date:
        _daily_signals = 0
        _last_signal_date = today
    if _consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
        return False, f"MAX PÉRDIDAS CONSECUTIVAS ({MAX_CONSECUTIVE_LOSSES})"
    if _daily_signals >= MAX_DAILY_SIGNALS:
        return False, f"MAX SEÑALES DIARIAS ({MAX_DAILY_SIGNALS})"
    return True, ""

def register_signal_result(won):
    """Registra resultado de una señal para gestión de riesgo"""
    global _consecutive_losses, _consecutive_wins, _daily_signals
    _daily_signals += 1
    if won:
        _consecutive_losses = 0
        _consecutive_wins += 1
    else:
        _consecutive_losses += 1
        _consecutive_wins = 0

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
    Si falla, retorna lista vacía — sin datos reales no hay señales.
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

    # No synthetic data - return empty to pause signals
    print("  ⚠ Sin velas reales disponibles — señales pausadas")
    return []

# ── HTML ─────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AURUM · XAU/USD</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=IBM+Plex+Mono:wght@300;400;500&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&display=swap" rel="stylesheet">
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=IBM+Plex+Mono:wght@300;400;500&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&display=swap');

:root {
  --bg:     #080600;
  --bg2:    #0C0A02;
  --bg3:    #111005;
  --bg4:    #161408;
  --gold:   #C9A84C;
  --gold2:  #F0D080;
  --gold3:  #6B5520;
  --green:  #00CC88;
  --red:    #CC3344;
  --text:   #E8E0C0;
  --dim:    #3D3010;
  --dim2:   #5A4820;
  --border: rgba(201,168,76,0.1);
  --border2:rgba(201,168,76,0.2);
  --glow:   0 0 30px rgba(201,168,76,0.15);
}

* { margin:0; padding:0; box-sizing:border-box; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'IBM Plex Mono', monospace;
  min-height: 100vh;
  overflow-x: hidden;
}

/* Subtle texture */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse at 20% 50%, rgba(201,168,76,0.03) 0%, transparent 60%),
    radial-gradient(ellipse at 80% 20%, rgba(201,168,76,0.02) 0%, transparent 50%);
  pointer-events: none;
  z-index: 0;
}

@keyframes fadeUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
@keyframes pulse  { 0%,100%{opacity:1} 50%{opacity:.4} }
@keyframes shimmer { 0%{background-position:-200% center} 100%{background-position:200% center} }

/* HEADER */
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 32px;
  border-bottom: 1px solid var(--border);
  background: rgba(8,6,0,0.98);
  position: relative;
  z-index: 10;
}
header::after {
  content: '';
  position: absolute;
  bottom: 0; left: 32px; right: 32px;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--gold), transparent);
  opacity: .3;
}

.logo {
  font-family: 'Playfair Display', serif;
  font-size: 20px;
  font-weight: 700;
  color: var(--gold);
  letter-spacing: 8px;
}
.logo span {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 8px;
  letter-spacing: 4px;
  color: var(--dim2);
  display: block;
  margin-top: -1px;
}

.header-right { display:flex; align-items:center; gap:14px; }
.live-badge { display:flex; align-items:center; gap:7px; font-size:9px; letter-spacing:3px; color:var(--dim2); }
.live-dot { width:6px; height:6px; border-radius:50%; background:var(--green); animation:pulse 2s infinite; }
.sound-toggle { background:transparent; border:1px solid var(--border); color:var(--dim2); font-family:'IBM Plex Mono'; font-size:8px; letter-spacing:2px; padding:5px 10px; cursor:pointer; transition:all .3s; }
.sound-toggle.on { border-color:rgba(0,204,136,.3); color:var(--green); }
.tg-status { font-size:8px; letter-spacing:2px; padding:5px 10px; border:1px solid var(--border); color:var(--dim2); }
.tg-status.connected { border-color:rgba(0,204,136,.3); color:var(--green); }

/* NAV */
nav {
  display: flex;
  border-bottom: 1px solid var(--border);
  padding: 0 32px;
  background: var(--bg);
  position: relative;
  z-index: 10;
}
.nav-tab {
  padding: 14px 20px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 8px;
  letter-spacing: 4px;
  color: var(--dim2);
  cursor: pointer;
  border-bottom: 1px solid transparent;
  transition: all .3s;
  background: transparent;
  border-top: none; border-left: none; border-right: none;
}
.nav-tab:hover { color: var(--gold); }
.nav-tab.active {
  color: var(--gold);
  border-bottom-color: var(--gold);
}

/* LAYOUT */
.main { display:grid; grid-template-columns:1fr 300px; min-height:calc(100vh - 112px); position:relative; z-index:1; }
.left { padding:28px 32px; border-right:1px solid var(--border); }
.right { padding:24px 22px; display:flex; flex-direction:column; gap:18px; overflow-y:auto; background:var(--bg2); }
.page { display:none; }
.page.active { display:block; }

/* SESSIONS */
.sessions-bar { display:flex; gap:1px; margin-bottom:16px; animation:fadeUp .5s ease both; }
.session-block {
  flex:1; padding:10px 6px; text-align:center;
  border:1px solid var(--border); background:var(--bg2);
  transition:all .4s;
}
.session-block.active-session {
  border-color:var(--border2);
  background:rgba(201,168,76,.05);
}
.session-block.overlap {
  border-color:rgba(201,168,76,.4);
  background:rgba(201,168,76,.08);
}
.session-block.closed { opacity:.3; }
.session-name {
  font-family:'Playfair Display',serif;
  font-size:12px; letter-spacing:2px;
  color:var(--dim2); transition:all .4s;
}
.session-block.active-session .session-name { color:var(--gold); }
.session-block.overlap .session-name { color:var(--gold2); }
.session-hours { font-size:7px; letter-spacing:1px; color:var(--dim); margin-top:3px; }
.session-status { font-size:8px; letter-spacing:2px; margin-top:4px; color:var(--dim); }
.session-block.active-session .session-status { color:var(--gold); }
.session-block.overlap .session-status { color:var(--gold2); }

.session-tip {
  font-family:'Cormorant Garamond',serif;
  font-size:12px; font-style:italic;
  color:var(--dim2); padding:8px 14px;
  border-left:2px solid var(--gold3);
  background:rgba(201,168,76,.03);
  margin-bottom:16px; letter-spacing:.5px;
}
.session-tip.hot { border-left-color:var(--gold); color:var(--gold); }
.session-tip.warm { border-left-color:var(--gold2); color:var(--gold2); }

#newsWarning {
  display:none; font-size:9px; letter-spacing:2px;
  padding:8px 14px; margin-bottom:12px;
  border-left:2px solid var(--red); color:var(--red);
  background:rgba(204,51,68,.04);
}

.htf-row {
  display:flex; justify-content:space-between; align-items:center;
  margin-bottom:14px; padding:8px 14px;
  border:1px solid var(--border); background:var(--bg2);
}
.htf-label { font-size:8px; letter-spacing:3px; color:var(--dim2); }

.tf-row { display:flex; gap:6px; align-items:center; margin-bottom:18px; }
.tf-label { font-size:8px; letter-spacing:3px; color:var(--dim2); }

/* PRICE */
.price-section { margin-bottom:24px; animation:fadeUp .5s ease both; }
.price-label { font-size:8px; letter-spacing:4px; color:var(--dim2); margin-bottom:6px; }
.price-main {
  font-family:'Playfair Display',serif;
  font-size:48px; font-weight:700;
  color:var(--gold2); line-height:1;
  letter-spacing:1px;
}
.price-change { font-size:11px; margin-top:6px; letter-spacing:1px; }
.price-change.up { color:var(--green); }
.price-change.down { color:var(--red); }
.source-badge { font-size:8px; letter-spacing:2px; padding:3px 8px; border:1px solid; display:inline-block; margin-top:6px; }
.source-badge.live { color:var(--green); border-color:rgba(0,204,136,.3); }
.source-badge.sim  { color:var(--dim2); border-color:var(--border); }

/* SIGNAL CARD */
.alert-box {
  padding:22px 26px; margin-bottom:20px;
  border:1px solid var(--border); background:var(--bg2);
  position:relative; overflow:hidden;
  animation:fadeUp .5s .1s ease both; transition:all .5s;
}
.alert-box.go-buy  { border-color:rgba(0,204,136,.25); background:rgba(0,204,136,.03); }
.alert-box.go-sell { border-color:rgba(204,51,68,.25);  background:rgba(204,51,68,.03); }
.alert-box::before {
  content:''; position:absolute; top:0; left:0; right:0; height:1px;
  transition:background .5s;
}
.alert-box.go-buy::before  { background:linear-gradient(90deg,transparent,var(--green),transparent); }
.alert-box.go-sell::before { background:linear-gradient(90deg,transparent,var(--red),transparent); }
.alert-box.wait::before    { background:linear-gradient(90deg,transparent,var(--gold3),transparent); }

.alert-tag { font-size:8px; letter-spacing:4px; color:var(--dim2); margin-bottom:12px; }
.alert-signal {
  font-family:'Playfair Display',serif;
  font-size:36px; font-weight:700;
  letter-spacing:2px; line-height:1;
  transition:color .5s;
}
.alert-box.go-buy .alert-signal  { color:var(--green); }
.alert-box.go-sell .alert-signal { color:var(--red); }
.alert-box.wait .alert-signal    { color:var(--dim2); }

.alert-reason {
  font-family:'Cormorant Garamond',serif;
  font-size:13px; font-style:italic;
  color:var(--dim2); margin-top:8px;
}

.validity-row { display:flex; gap:6px; margin-top:14px; flex-wrap:wrap; }
.validity-pill { font-size:7px; letter-spacing:1px; padding:3px 8px; border:1px solid var(--border); color:var(--dim2); }
.validity-pill.ok   { border-color:rgba(0,204,136,.3); color:var(--green); }
.validity-pill.fail { border-color:rgba(204,51,68,.3);  color:var(--red); }

.conf-row { display:flex; align-items:center; gap:10px; margin-top:14px; font-size:8px; letter-spacing:2px; color:var(--dim2); }
.conf-bar { flex:1; height:1px; background:var(--bg3); position:relative; }
.conf-fill { position:absolute; left:0; top:0; height:100%; transition:width .8s, background .5s; }
.alert-box.go-buy  .conf-fill { background:var(--green); }
.alert-box.go-sell .conf-fill { background:var(--red); }
.alert-box.wait    .conf-fill { background:var(--gold3); }

/* LEVELS */
.levels-box { background:var(--bg2); border:1px solid var(--border); padding:18px 22px; margin-bottom:20px; }
.levels-title {
  font-size:8px; letter-spacing:4px; color:var(--dim2);
  margin-bottom:14px; padding-bottom:10px;
  border-bottom:1px solid var(--border);
  display:flex; justify-content:space-between; align-items:center;
}
.level-row { display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid rgba(201,168,76,.04); }
.level-row:last-child { border:none; }
.level-name { font-size:8px; letter-spacing:3px; color:var(--dim2); }
.level-val { font-family:'Playfair Display',serif; font-size:20px; font-weight:700; }
.level-val.entry { color:var(--gold2); }
.level-val.tp    { color:var(--green); }
.level-val.sl    { color:var(--red); }
.level-sub { font-size:8px; color:var(--dim); text-align:right; margin-top:2px; }

.log-btn {
  width:100%; padding:8px; background:transparent;
  border:1px solid rgba(0,204,136,.2); color:var(--green);
  font-family:'IBM Plex Mono'; font-size:8px; letter-spacing:3px;
  cursor:pointer; margin-top:4px; transition:all .3s;
}
.log-btn:hover { background:rgba(0,204,136,.05); }

/* CHART */
.chart-wrap { animation:fadeUp .5s .25s ease both; }
.chart-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }
.chart-title { font-size:8px; letter-spacing:3px; color:var(--dim2); }
.chart-tabs { display:flex; gap:4px; }
.chart-tab {
  padding:4px 10px; font-size:8px; letter-spacing:2px;
  border:1px solid var(--border); background:transparent;
  color:var(--dim2); cursor:pointer; transition:all .2s;
  font-family:'IBM Plex Mono';
}
.chart-tab.active, .chart-tab:hover { border-color:var(--border2); color:var(--gold); }
canvas { width:100%; display:block; }

/* INDICATORS */
.indicators-row { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:20px; animation:fadeUp .5s .35s ease both; }
.indicator-box { background:var(--bg2); border:1px solid var(--border); padding:16px; }
.ind-label { font-size:8px; letter-spacing:3px; color:var(--dim2); margin-bottom:6px; }
.ind-value { font-family:'Playfair Display',serif; font-size:28px; font-weight:700; line-height:1; }
.ind-status { font-size:8px; letter-spacing:2px; margin-top:5px; }
.ind-status.bullish { color:var(--green); }
.ind-status.bearish { color:var(--red); }
.ind-status.neutral { color:var(--gold); }
.rsi-gauge { position:relative; width:100%; height:2px; background:linear-gradient(to right,var(--red) 30%,var(--gold3) 30%,var(--gold3) 70%,var(--green) 70%); margin-top:10px; }
.rsi-needle { position:absolute; top:-5px; width:1px; height:12px; background:var(--gold2); transform:translateX(-50%); transition:left .8s; }
.rsi-labels { display:flex; justify-content:space-between; margin-top:4px; font-size:7px; color:var(--dim); }

/* RIGHT PANEL */
.panel-title { font-size:8px; letter-spacing:4px; color:var(--dim2); margin-bottom:12px; padding-bottom:8px; border-bottom:1px solid var(--border); }
.rules-box { background:var(--bg3); border:1px solid var(--border); padding:14px; }
.rule-row { display:flex; justify-content:space-between; align-items:center; padding:7px 0; border-bottom:1px solid rgba(201,168,76,.04); font-size:9px; }
.rule-row:last-child { border:none; }
.rule-label { color:var(--dim2); letter-spacing:1px; font-size:8px; }
.rule-input { background:transparent; border:none; border-bottom:1px solid var(--gold3); color:var(--gold); font-family:'IBM Plex Mono'; font-size:12px; width:65px; text-align:right; outline:none; padding:2px 4px; }
.rule-input:focus { border-bottom-color:var(--gold); }
.rule-unit { font-size:8px; color:var(--dim2); margin-left:3px; }

.ma-row { display:flex; justify-content:space-between; align-items:center; padding:7px 0; border-bottom:1px solid rgba(201,168,76,.04); }
.ma-name { font-size:8px; letter-spacing:1px; color:var(--dim2); }
.ma-val { font-size:10px; color:var(--text); }
.ma-sig { font-size:8px; letter-spacing:2px; }
.ma-sig.b { color:var(--green); } .ma-sig.s { color:var(--red); }

.history-list { display:flex; flex-direction:column; gap:6px; }
.hist-item { background:var(--bg3); border:1px solid var(--border); padding:9px 12px; display:flex; align-items:center; gap:10px; }
.hist-dot { width:5px; height:5px; border-radius:50%; flex-shrink:0; }
.hist-dot.b { background:var(--green); } .hist-dot.s { background:var(--red); } .hist-dot.h { background:var(--dim); }
.hist-info { flex:1; }
.hist-sig { font-size:9px; letter-spacing:2px; font-family:'IBM Plex Mono'; }
.hist-sig.b { color:var(--green); } .hist-sig.s { color:var(--red); } .hist-sig.h { color:var(--dim2); }
.hist-sub { font-size:8px; color:var(--dim); margin-top:2px; }
.hist-time { font-size:8px; color:var(--dim2); }

.update-btn {
  width:100%; padding:12px; background:transparent;
  border:1px solid var(--border2); color:var(--gold);
  font-family:'IBM Plex Mono'; font-size:9px; letter-spacing:4px;
  cursor:pointer; transition:all .3s; position:relative; overflow:hidden;
}
.update-btn::before { content:''; position:absolute; inset:0; background:var(--gold); transform:scaleX(0); transform-origin:left; transition:transform .3s; }
.update-btn:hover::before { transform:scaleX(1); }
.update-btn:hover { color:var(--bg); }
.update-btn span { position:relative; z-index:1; }
.update-btn:disabled { opacity:.3; pointer-events:none; }

.timestamp { font-size:8px; color:var(--dim2); letter-spacing:2px; text-align:center; }
.disclaimer { font-size:8px; color:var(--dim); line-height:1.7; padding:12px; border:1px solid var(--border); }
.disclaimer strong { color:var(--gold3); }

/* AI PANEL */
.ai-panel { background:var(--bg2); border:1px solid var(--border); padding:18px 22px; margin-bottom:20px; position:relative; }
.ai-panel::before { content:''; position:absolute; top:0;left:0;right:0; height:1px; background:linear-gradient(90deg,transparent,var(--gold3),transparent); }
.ai-title { font-size:8px; letter-spacing:4px; color:var(--dim2); margin-bottom:14px; padding-bottom:10px; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; }
.ai-badge { font-size:7px; letter-spacing:2px; padding:3px 8px; border:1px solid var(--border2); color:var(--gold); }
.ai-prob-wrap { display:flex; align-items:center; gap:14px; margin-bottom:14px; }
.ai-prob-circle { width:68px; height:68px; border-radius:50%; border:1px solid var(--border2); display:flex; flex-direction:column; align-items:center; justify-content:center; flex-shrink:0; transition:all .5s; }
.ai-prob-circle.buy  { border-color:rgba(0,204,136,.3); }
.ai-prob-circle.sell { border-color:rgba(204,51,68,.3); }
.ai-prob-val { font-family:'Playfair Display',serif; font-size:22px; font-weight:700; line-height:1; }
.ai-prob-val.buy  { color:var(--green); } .ai-prob-val.sell { color:var(--red); } .ai-prob-val.neutral { color:var(--gold); }
.ai-prob-label { font-size:7px; letter-spacing:2px; color:var(--dim2); margin-top:2px; }
.ai-signal-wrap { flex:1; }
.ai-signal { font-family:'Playfair Display',serif; font-size:24px; font-weight:700; transition:all .5s; }
.ai-signal.buy { color:var(--green); } .ai-signal.sell { color:var(--red); } .ai-signal.neutral { color:var(--gold); }
.ai-desc { font-family:'Cormorant Garamond',serif; font-size:12px; font-style:italic; color:var(--dim2); margin-top:4px; }
.ai-bar-wrap { margin-top:8px; }
.ai-bar-label { display:flex; justify-content:space-between; font-size:7px; color:var(--dim); margin-bottom:4px; }
.ai-bar { height:1px; background:var(--bg3); position:relative; }
.ai-bar-fill { position:absolute; left:0; top:0; height:100%; transition:width .8s, background .5s; }
.ai-stats-row { display:flex; gap:8px; margin-top:12px; }
.ai-stat { flex:1; text-align:center; padding:8px; background:var(--bg3); border:1px solid var(--border); }
.ai-stat-val { font-family:'Playfair Display',serif; font-size:18px; font-weight:700; color:var(--gold); }
.ai-stat-label { font-size:7px; letter-spacing:2px; color:var(--dim2); margin-top:3px; }
.ai-consensus { margin-top:12px; padding:10px 14px; border:1px solid; font-family:'Cormorant Garamond',serif; font-size:13px; font-style:italic; }
.ai-consensus.agree    { border-color:rgba(0,204,136,.2); color:var(--green); background:rgba(0,204,136,.03); }
.ai-consensus.disagree { border-color:rgba(204,51,68,.2);  color:var(--red);   background:rgba(204,51,68,.03); }
.ai-consensus.neutral  { border-color:var(--border); color:var(--dim2); }
.ai-training { font-size:8px; color:var(--dim); letter-spacing:2px; margin-top:10px; text-align:center; }

/* BACKTESTING */
.bt-page { padding:28px 32px; }
.bt-title { font-family:'Playfair Display',serif; font-size:32px; font-weight:700; color:var(--gold); letter-spacing:2px; }
.bt-subtitle { font-size:8px; letter-spacing:3px; color:var(--dim2); margin-top:4px; }
.bt-controls { display:flex; gap:10px; align-items:flex-end; margin:20px 0; flex-wrap:wrap; }
.bt-input { background:var(--bg2); border:1px solid var(--border); color:var(--gold); font-family:'IBM Plex Mono'; font-size:11px; padding:7px 10px; outline:none; width:100px; }
.bt-input:focus { border-color:var(--border2); }
.bt-label { font-size:8px; letter-spacing:3px; color:var(--dim2); margin-bottom:4px; }
.bt-run { padding:9px 20px; background:transparent; border:1px solid var(--border2); color:var(--gold); font-family:'IBM Plex Mono'; font-size:9px; letter-spacing:3px; cursor:pointer; transition:all .3s; }
.bt-run:hover { background:var(--gold); color:var(--bg); }
.bt-stats { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:20px; }
.bt-stat { background:var(--bg2); border:1px solid var(--border); padding:14px; text-align:center; }
.bt-stat-val { font-family:'Playfair Display',serif; font-size:28px; font-weight:700; }
.bt-stat-val.green { color:var(--green); } .bt-stat-val.red { color:var(--red); } .bt-stat-val.gold { color:var(--gold); }
.bt-stat-label { font-size:7px; letter-spacing:3px; color:var(--dim2); margin-top:4px; }
.bt-chart-wrap { background:var(--bg2); border:1px solid var(--border); padding:16px; margin-bottom:16px; }
.bt-chart-title { font-size:8px; letter-spacing:3px; color:var(--dim2); margin-bottom:12px; }
.bt-table { width:100%; border-collapse:collapse; font-size:9px; }
.bt-table th { font-size:7px; letter-spacing:3px; color:var(--dim2); padding:8px 10px; text-align:left; border-bottom:1px solid var(--border); }
.bt-table td { padding:7px 10px; border-bottom:1px solid rgba(201,168,76,.03); }
.td-buy { color:var(--green); } .td-sell { color:var(--red); } .td-pos { color:var(--green); } .td-neg { color:var(--red); }

/* REGISTRO */
.reg-page { padding:28px 32px; }
.reg-title { font-family:'Playfair Display',serif; font-size:32px; font-weight:700; color:var(--gold); letter-spacing:2px; }
.reg-btn { padding:8px 16px; background:transparent; border:1px solid var(--border); color:var(--dim2); font-family:'IBM Plex Mono'; font-size:8px; letter-spacing:2px; cursor:pointer; transition:all .3s; }
.reg-btn:hover { border-color:var(--border2); color:var(--gold); }
.reg-btn.primary { border-color:var(--border2); color:var(--gold); }
.reg-summary { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:16px 0; }
.reg-stat { background:var(--bg2); border:1px solid var(--border); padding:14px; }
.reg-stat-val { font-family:'Playfair Display',serif; font-size:28px; font-weight:700; }
.reg-stat-val.g { color:var(--green); } .reg-stat-val.r { color:var(--red); } .reg-stat-val.w { color:var(--gold); }
.reg-stat-label { font-size:7px; letter-spacing:3px; color:var(--dim2); margin-top:4px; }
.reg-table-wrap { background:var(--bg2); border:1px solid var(--border); overflow:auto; }
.reg-table { width:100%; border-collapse:collapse; font-size:9px; }
.reg-table th { font-size:7px; letter-spacing:2px; color:var(--dim2); padding:8px 10px; text-align:left; border-bottom:1px solid var(--border); white-space:nowrap; }
.reg-table td { padding:7px 10px; border-bottom:1px solid rgba(201,168,76,.03); white-space:nowrap; }
.reg-empty { padding:30px; text-align:center; color:var(--dim2); font-size:10px; letter-spacing:2px; }

/* CONFIG */
.tg-box { background:var(--bg3); border:1px solid var(--border); padding:16px; }
.tg-field { display:flex; flex-direction:column; gap:4px; margin-bottom:12px; }
.tg-field label { font-size:7px; letter-spacing:3px; color:var(--dim2); }
.tg-field input { background:var(--bg2); border:1px solid var(--border); color:var(--gold); font-family:'IBM Plex Mono'; font-size:10px; padding:7px 10px; outline:none; }
.tg-field input:focus { border-color:var(--border2); }
.tg-save { width:100%; padding:9px; background:transparent; border:1px solid var(--border2); color:var(--gold); font-family:'IBM Plex Mono'; font-size:8px; letter-spacing:3px; cursor:pointer; margin-top:8px; transition:all .3s; }
.tg-save:hover { background:rgba(201,168,76,.08); }
.tg-hint { font-size:8px; color:var(--dim); line-height:1.7; margin-top:10px; padding:10px; border:1px solid var(--border); }

.spinner { width:16px; height:16px; border:1px solid rgba(201,168,76,.2); border-top-color:var(--gold); border-radius:50%; animation:spin .8s linear infinite; display:inline-block; margin-right:8px; vertical-align:middle; }
@keyframes spin { to{ transform:rotate(360deg); } }

.modal-overlay { position:fixed; inset:0; background:rgba(8,6,0,.97); z-index:200; display:none; flex-direction:column; padding:28px; }
.modal-overlay.open { display:flex; }
.modal-title { font-family:'Playfair Display',serif; font-size:18px; font-weight:700; color:var(--gold); }
.modal-close { font-size:20px; color:var(--dim2); cursor:pointer; background:none; border:none; }
.modal-content { flex:1; overflow-y:auto; margin-top:16px; }

@media(max-width:900px){
  .main{grid-template-columns:1fr}
  .right{border-top:1px solid var(--border)}
  .price-main{font-size:36px}
  .bt-stats{grid-template-columns:repeat(2,1fr)}
}

/* ── PREMIUM 4K VISUAL ENHANCEMENTS ─────────────────────

/* Gold shimmer animation for key elements */
@keyframes goldShimmer {
  0%   { background-position: -200% center; }
  100% { background-position:  200% center; }
}
@keyframes subtleGlow {
  0%,100% { box-shadow: 0 0 20px rgba(201,168,76,0.08); }
  50%      { box-shadow: 0 0 40px rgba(201,168,76,0.18); }
}
@keyframes priceFlash {
  0%   { color: var(--gold2); }
  50%  { color: #FFFFFF; }
  100% { color: var(--gold2); }
}
@keyframes borderPulse {
  0%,100% { border-color: rgba(201,168,76,0.1); }
  50%      { border-color: rgba(201,168,76,0.3); }
}

/* Enhanced body with deep texture */
body {
  background: radial-gradient(ellipse at 50% 0%, #0F0C02 0%, #080600 40%, #000000 100%) !important;
}

/* Premium header with glass effect */
header {
  background: linear-gradient(180deg, rgba(15,12,2,0.99) 0%, rgba(8,6,0,0.97) 100%) !important;
  box-shadow: 0 1px 0 rgba(201,168,76,0.08), 0 4px 20px rgba(0,0,0,0.8);
}

/* Logo with shimmer effect */
.logo {
  background: linear-gradient(90deg, #8B6914, #F0D080, #C9A84C, #F0D080, #8B6914);
  background-size: 200% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: goldShimmer 4s linear infinite;
  filter: drop-shadow(0 0 8px rgba(201,168,76,0.3));
}

/* Price with premium styling */
.price-main {
  background: linear-gradient(135deg, #C9A84C 0%, #F0D080 40%, #FFFFFF 60%, #F0D080 80%, #C9A84C 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  filter: drop-shadow(0 2px 8px rgba(201,168,76,0.2));
}

/* Signal card with depth */
.alert-box {
  backdrop-filter: blur(10px);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 4px 24px rgba(0,0,0,0.5);
}
.alert-box.go-sell {
  box-shadow: inset 0 1px 0 rgba(204,51,68,0.1), 0 4px 24px rgba(0,0,0,0.5), 0 0 40px rgba(204,51,68,0.05);
}
.alert-box.go-buy {
  box-shadow: inset 0 1px 0 rgba(0,204,136,0.1), 0 4px 24px rgba(0,0,0,0.5), 0 0 40px rgba(0,204,136,0.05);
}

/* Signal text with premium gradient */
.alert-box.go-sell .alert-signal {
  background: linear-gradient(135deg, #CC3344, #FF6677);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  filter: drop-shadow(0 2px 6px rgba(204,51,68,0.3));
}
.alert-box.go-buy .alert-signal {
  background: linear-gradient(135deg, #00AA66, #00FF99);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  filter: drop-shadow(0 2px 6px rgba(0,204,136,0.3));
}

/* Level values with glow */
.level-val.entry {
  background: linear-gradient(135deg, #C9A84C, #F0D080);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.level-val.tp {
  filter: drop-shadow(0 0 4px rgba(0,204,136,0.4));
}
.level-val.sl {
  filter: drop-shadow(0 0 4px rgba(204,51,68,0.4));
}

/* Premium cards with depth */
.levels-box, .ai-panel, .indicator-box, .rules-box {
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.02), 0 2px 12px rgba(0,0,0,0.4);
  transition: box-shadow 0.4s ease;
}
.levels-box:hover, .ai-panel:hover {
  box-shadow: inset 0 1px 0 rgba(201,168,76,0.05), 0 4px 20px rgba(0,0,0,0.6), 0 0 30px rgba(201,168,76,0.04);
}

/* Session blocks with premium hover */
.session-block.active-session {
  box-shadow: inset 0 0 30px rgba(201,168,76,0.06), 0 0 20px rgba(201,168,76,0.05);
  animation: subtleGlow 3s ease infinite;
}

/* Nav tab with underline animation */
.nav-tab.active {
  text-shadow: 0 0 20px rgba(201,168,76,0.4);
}

/* Update button premium */
.update-btn {
  box-shadow: inset 0 1px 0 rgba(201,168,76,0.1);
}
.update-btn:hover {
  box-shadow: 0 0 20px rgba(201,168,76,0.2);
}

/* AI circle premium */
.ai-prob-circle {
  box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
}
.ai-prob-circle.sell {
  box-shadow: inset 0 0 20px rgba(0,0,0,0.5), 0 0 20px rgba(204,51,68,0.1);
}
.ai-prob-circle.buy {
  box-shadow: inset 0 0 20px rgba(0,0,0,0.5), 0 0 20px rgba(0,204,136,0.1);
}

/* AI value gradient */
.ai-prob-val.sell {
  background: linear-gradient(135deg, #CC3344, #FF6677);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.ai-prob-val.buy {
  background: linear-gradient(135deg, #00AA66, #00FF99);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* Right panel premium background */
.right {
  background: linear-gradient(180deg, #0C0A02 0%, #080600 100%) !important;
}

/* Canvas charts with glow */
canvas {
  filter: drop-shadow(0 0 1px rgba(201,168,76,0.1));
}

/* Scrollbar styling */
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--gold3); border-radius:2px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold); }

/* Selection color */
::selection { background: rgba(201,168,76,0.2); color: var(--gold2); }

/* Divider lines with gradient */
.levels-title, .panel-title, .ai-title {
  position: relative;
}
.levels-title::after, .panel-title::after {
  content: '';
  position: absolute;
  bottom: -1px; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, var(--gold3), transparent);
  opacity: 0.5;
}
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
  <div id="newsWarning" style="display:none;background:rgba(204,51,68,.04);border-left:2px solid var(--red);color:var(--red);padding:8px 14px;font-size:9px;letter-spacing:2px;margin-bottom:8px"></div>
  <div id="riskWarning" style="display:none;background:rgba(201,168,76,.04);border-left:2px solid var(--gold);color:var(--gold);padding:8px 14px;font-size:9px;letter-spacing:2px;margin-bottom:8px"></div>
  <div id="driftWarning" style="display:none;background:rgba(204,51,68,.04);border-left:2px solid var(--red);color:var(--red);padding:8px 14px;font-size:9px;letter-spacing:2px;margin-bottom:8px"></div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
    <div style="font-size:9px;letter-spacing:3px;color:var(--text-dim)">TENDENCIA 1H:</div>
    <div id="htfTrend" style="font-size:11px;letter-spacing:2px;color:var(--gold)">Cargando...</div>
  </div>

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
  // News filter
  const newsBlocked = window._newsActive || false;
  const htfTrend = window._htfTrend || 'neutral';
  const mtfOkBuy  = htfTrend === 'up'   || htfTrend === 'neutral';
  const mtfOkSell = htfTrend === 'down' || htfTrend === 'neutral';
  const mtfOk = rawSignal==='COMPRAR' ? mtfOkBuy : mtfOkSell;

  const checks=[];let valid=true;
  const cok=confidence>=minConf;checks.push({label:`CONF ${confidence}%≥${minConf}%`,ok:cok});if(!cok)valid=false;
  // MTF check
  checks.push({label:`1H ${htfTrend.toUpperCase()}`,ok:mtfOk});if(!mtfOk)valid=false;
  // News check
  if(newsBlocked){checks.push({label:'NOTICIA ACTIVA',ok:false});valid=false;}
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

  // Solo datos reales — sin simulación
  if(prices.length < 50){
    document.getElementById('btBody').innerHTML = '<tr><td colspan="8" style="text-align:center;padding:30px;color:var(--text-dim);letter-spacing:2px">SIN DATOS SUFICIENTES — ESPERA A QUE CARGUEN LAS VELAS REALES</td></tr>';
    document.getElementById('btTotal').textContent='—';
    document.getElementById('btWins').textContent='—';
    document.getElementById('btLosses').textContent='—';
    document.getElementById('btWR').textContent='—';
    return;
  }
  const btPrices = [...prices];

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
    // Sin simulación — solo contar señales reales
    // El resultado se determina comparando con precio futuro real
    const futureIdx = Math.min(i + 5, btPrices.length - 1);
    const futurePrice = btPrices[futureIdx];
    const won = sig==='COMPRAR' ? futurePrice > price + (tp*0.5) : futurePrice < price - (tp*0.5);
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
  const W=Math.max(cv.offsetWidth,600)||800,H=160;cv.width=W;cv.height=H;
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


// ── RISK MANAGEMENT JS ───────────────────────────────────
async function updateRiskStatus() {
  try {
    const [riskRes, aiStatsRes] = await Promise.all([
      fetch('/riskstatus'),
      fetch('/aistats')
    ]);
    const risk    = await riskRes.json();
    const aiStats = await aiStatsRes.json();

    // Risk warning
    const riskEl = document.getElementById('riskWarning');
    if(riskEl) {
      if(!risk.can_trade) {
        riskEl.style.display = 'block';
        riskEl.textContent = '🛡 RIESGO: ' + risk.reason + ' — Señales pausadas';
      } else {
        riskEl.style.display = 'none';
      }
    }

    // Consecutive losses indicator
    const lossEl = document.getElementById('consecutiveLosses');
    if(lossEl) lossEl.textContent = risk.consecutive_losses + '/' + risk.max_losses;

    const signalsEl = document.getElementById('dailySignals');
    if(signalsEl) signalsEl.textContent = risk.daily_signals + '/' + risk.max_daily;

    // AI Drift warning
    const driftEl = document.getElementById('driftWarning');
    if(driftEl) {
      if(aiStats.is_drifting) {
        driftEl.style.display = 'block';
        driftEl.textContent = '⚡ DRIFT DETECTADO (' + (aiStats.drift_score*100).toFixed(0) + '%) — IA pausada, mercado cambiado';
      } else {
        driftEl.style.display = 'none';
      }
    }

    // Hour & DOW accuracy
    const hourAccEl = document.getElementById('hourAccuracy');
    if(hourAccEl && aiStats.hour_accuracy) {
      hourAccEl.textContent = 'Esta hora: ' + aiStats.hour_accuracy + '%';
    }
    const dowAccEl = document.getElementById('dowAccuracy');
    if(dowAccEl && aiStats.dow_accuracy) {
      const days = ['Lun','Mar','Mié','Jue','Vie','Sáb','Dom'];
      const today = days[new Date().getDay()-1] || 'Hoy';
      dowAccEl.textContent = today + ': ' + aiStats.dow_accuracy + '%';
    }

    // Store risk status globally
    window._canTrade = risk.can_trade;
    window._isDrifting = aiStats.is_drifting;

  } catch(e) {
    window._canTrade = true;
    window._isDrifting = false;
  }
}

// ── VOLATILITY & LATERAL MARKET DETECTION ────────────────
function detectLateralMarket(prices, n=20) {
  if(prices.length < n) return false;
  const recent = prices.slice(-n);
  const high = Math.max(...recent);
  const low  = Math.min(...recent);
  const rangePct = (high - low) / low * 100;
  return rangePct < 0.3;
}

function calcPositionSize(atr, price) {
  if(!atr || !price) return 1.0;
  const atrPct = atr / price * 100;
  if(atrPct > 0.8)  return 0.5;
  if(atrPct > 0.5)  return 0.75;
  return 1.0;
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
      // Solo precio real
      prices.push(livePrice);
      if(prices.length>300) prices=prices.slice(-300);
      document.getElementById('priceDisplay').textContent='$'+livePrice.toFixed(2);
      const ch=priceData.ch||0, chp=priceData.chp||0;
      const el=document.getElementById('priceChange');
      el.textContent=(ch>=0?'+':'')+ch.toFixed(2)+' ('+(ch>=0?'+':'')+Number(chp).toFixed(2)+'%)';
      el.className='price-change '+(ch>=0?'up':'down');
      if(!candles) document.getElementById('sourceBadge').innerHTML='<span class="source-badge live">● PRECIO REAL</span>';
    } else if(prices.length === 0) {
      // Sin precio real — mostrar estado de espera
      document.getElementById('priceDisplay').textContent='—';
      document.getElementById('priceChange').textContent='Sin datos — mercado cerrado o sin conexión';
      document.getElementById('sourceBadge').innerHTML='<span class="source-badge sim">◌ SIN DATOS</span>';
      btn.disabled=false; btn.querySelector('span').textContent='↻ ACTUALIZAR'; return;
    }

    if(!prices.length){ btn.disabled=false; btn.querySelector('span').textContent='↻ ACTUALIZAR'; return; }
  }catch(e){
    console.log('fetch error:', e);
    if(!prices.length){
      btn.disabled=false; btn.querySelector('span').textContent='↻ ACTUALIZAR'; return;
    }
  }
  btn.querySelector('span').textContent='ANALIZANDO...';

  // Check news filter and HTF trend
  try {
    const [newsRes, htfRes] = await Promise.all([
      fetch('/newsstatus'),
      fetch('/htftrend')
    ]);
    const newsData = await newsRes.json();
    const htfData  = await htfRes.json();

    // Show news warning if active
    const newsEl = document.getElementById('newsWarning');
    if(newsEl){
      window._newsActive = newsData.active;
    if(newsData.active){
        newsEl.style.display='block';
        newsEl.textContent='⚠ NOTICIA ACTIVA: '+newsData.event+' — Señales pausadas ±'+newsData.pause_before+'min';
      } else {
        newsEl.style.display='none';
      }
    }

    // Store HTF trend for signal validation
    window._htfTrend = htfData.trend;
    const htfEl = document.getElementById('htfTrend');
    if(htfEl){
      const color = htfData.trend==='up'?'var(--buy)':htfData.trend==='down'?'var(--sell)':'var(--gold)';
      const arrow = htfData.trend==='up'?'↑ ALCISTA':htfData.trend==='down'?'↓ BAJISTA':'→ NEUTRAL';
      htfEl.textContent='1H: '+arrow;
      htfEl.style.color=color;
    }
  } catch(e) { window._htfTrend = 'neutral'; }

  const d=computeSignal();
  currentSignalData=d;
  triggerSound(d);
  // Telegram si señal válida nueva
  if(d.valid&&d.rawSignal!==lastValidSignal){
    const isLong=d.rawSignal==='COMPRAR';
    const tp=(isLong?d.price+d.tpDist:d.price-d.tpDist).toFixed(2);
    const sl=(isLong?d.price-d.slDist:d.price+d.slDist).toFixed(2);
    const em=d.rawSignal==='COMPRAR'?'🟢':'🔴';
    // Send with image
    fetch(`/sendimage?signal=${encodeURIComponent(d.rawSignal)}&price=${d.price}&tp=${tp}&sl=${sl}&rr=${d.rr}&conf=${d.confidence}&session=${encodeURIComponent(currentSessionName)}&atr=${d.ATR.toFixed(2)}`);
  }
  renderAlert(d);renderIndicators(d);addHistory(d);
  updateAI(d.valid ? d.rawSignal : 'ESPERAR');
  updateRiskStatus();

  // Lateral market detection
  if(detectLateralMarket(prices)) {
    const tip = document.getElementById('sessionTip');
    if(tip) {
      tip.textContent = '📊 Mercado lateral detectado — movimiento mínimo. Mejor esperar tendencia.';
      tip.className = 'session-tip';
    }
  }
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
            self._send(200, "application/json",
                       json.dumps(data or {"price":0,"ch":0,"chp":0}).encode())

        elif path == "/ohlc":
            interval = params.get("interval", "5min")
            size = int(params.get("size", "150"))
            candles = get_historical_ohlc(interval, size)
            self._send(200, "application/json", json.dumps(candles or []).encode())

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

        elif path == "/newsstatus":
            news_on, news_name = is_news_time()
            self._send(200, "application/json", json.dumps({
                "active": news_on,
                "event": news_name or "",
                "pause_before": PAUSE_MINUTES_BEFORE,
                "pause_after": PAUSE_MINUTES_AFTER
            }).encode())

        elif path == "/htftrend":
            trend = get_htf_trend()
            self._send(200, "application/json", json.dumps({
                "trend": trend
            }).encode())

        elif path == "/sendimage":
            if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                try:
                    signal     = params.get("signal", "")
                    price      = float(params.get("price", 0))
                    tp         = float(params.get("tp", 0))
                    sl         = float(params.get("sl", 0))
                    rr         = params.get("rr", "1.5")
                    confidence = params.get("conf", "70")
                    session    = params.get("session", "")
                    atr        = float(params.get("atr", 0))

                    svg = generate_signal_image(signal, price, tp, sl, rr, confidence, session, atr)
                    em  = "🟢" if signal == "COMPRAR" else "🔴"
                    caption = (f"{em} *AURUM · {signal}*\n"
                               f"💰 Entrada: ${price:.2f}\n"
                               f"🎯 TP: ${tp:.2f}\n"
                               f"🛡 SL: ${sl:.2f}\n"
                               f"📊 RR: {rr}:1\n"
                               f"💪 Confianza: {confidence}%\n"
                               f"🌍 Sesión: {session}")

                    threading.Thread(
                        target=send_telegram_direct,
                        args=(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, caption, svg),
                        daemon=True
                    ).start()
                    self._send(200, "application/json", b'{"ok":true}')
                except Exception as e:
                    print(f"  ⚠ sendimage error: {e}")
                    self._send(500, "application/json", b'{"ok":false}')
            else:
                self._send(200, "application/json", b'{"ok":false,"reason":"no telegram"}')

        elif path == "/aistats":
            hour_acc = _ai.get_hour_accuracy()
            dow_acc  = _ai.get_dow_accuracy()
            self._send(200, "application/json", json.dumps({
                "trained":      _ai.trained,
                "accuracy":     round(_ai.accuracy, 1),
                "epochs":       _ai.epochs,
                "drift_score":  round(_ai.drift_score, 2),
                "is_drifting":  _ai.is_market_drifting(),
                "hour_accuracy": round(hour_acc, 1) if hour_acc else None,
                "dow_accuracy":  round(dow_acc, 1) if dow_acc else None,
                "models":       ["LogisticRegression", "DecisionTree"],
                "ensemble":     "60% LR + 40% DT",
            }).encode())

        elif path == "/riskstatus":
            can_trade, reason = check_risk_limits()
            self._send(200, "application/json", json.dumps({
                "can_trade":          can_trade,
                "reason":             reason,
                "consecutive_losses": _consecutive_losses,
                "consecutive_wins":   _consecutive_wins,
                "daily_signals":      _daily_signals,
                "max_daily":          MAX_DAILY_SIGNALS,
                "max_losses":         MAX_CONSECUTIVE_LOSSES,
            }).encode())

        elif path == "/healthcheck":
            import datetime as _dt
            self._send(200, "application/json", json.dumps({
                "status":    "ok",
                "time_utc":  _dt.datetime.utcnow().isoformat(),
                "ai_ready":  _ai.trained,
                "ai_drift":  _ai.is_market_drifting(),
                "uptime_ok": True,
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



# ── GENERADOR DE IMAGEN PARA TELEGRAM ────────────────────
import base64, urllib.parse

def generate_signal_image(signal, price, tp, sl, rr, confidence, session, atr):
    """
    Genera una imagen SVG de la señal y la convierte a PNG via base64.
    La envía como foto a Telegram.
    """
    is_buy = signal == "COMPRAR"
    color_main  = "#00E5A0" if is_buy else "#FF4466"
    color_bg    = "#0A0A08"
    color_gold  = "#C9A84C"
    arrow       = "▲ COMPRAR" if is_buy else "▼ VENDER"
    emoji       = "🟢" if is_buy else "🔴"

    svg = f'''<svg width="600" height="340" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0A0A08"/>
      <stop offset="100%" stop-color="#111109"/>
    </linearGradient>
    <linearGradient id="glow" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{color_main}" stop-opacity="0.12"/>
      <stop offset="100%" stop-color="{color_main}" stop-opacity="0"/>
    </linearGradient>
  </defs>

  <!-- Background -->
  <rect width="600" height="340" fill="url(#bg)"/>
  <rect width="600" height="180" fill="url(#glow)"/>

  <!-- Top border -->
  <rect x="0" y="0" width="600" height="3" fill="{color_main}"/>

  <!-- Border -->
  <rect x="1" y="1" width="598" height="338" fill="none" stroke="{color_main}" stroke-width="1" stroke-opacity="0.4"/>

  <!-- Corner accents -->
  <line x1="1" y1="1" x2="40" y2="1" stroke="{color_gold}" stroke-width="1.5"/>
  <line x1="1" y1="1" x2="1" y2="40" stroke="{color_gold}" stroke-width="1.5"/>
  <line x1="599" y1="1" x2="560" y2="1" stroke="{color_gold}" stroke-width="1.5"/>
  <line x1="599" y1="1" x2="599" y2="40" stroke="{color_gold}" stroke-width="1.5"/>
  <line x1="1" y1="339" x2="40" y2="339" stroke="{color_gold}" stroke-width="1.5"/>
  <line x1="1" y1="339" x2="1" y2="300" stroke="{color_gold}" stroke-width="1.5"/>
  <line x1="599" y1="339" x2="560" y2="339" stroke="{color_gold}" stroke-width="1.5"/>
  <line x1="599" y1="339" x2="599" y2="300" stroke="{color_gold}" stroke-width="1.5"/>

  <!-- AURUM logo -->
  <text x="28" y="48" font-family="Georgia,serif" font-size="26" font-weight="bold" letter-spacing="8" fill="{color_gold}">AURUM</text>
  <text x="30" y="64" font-family="monospace" font-size="9" letter-spacing="4" fill="#6B6550">SEÑALES XAU/USD · IA</text>

  <!-- Live dot -->
  <circle cx="558" cy="40" r="5" fill="{color_main}"/>
  <text x="568" y="44" font-family="monospace" font-size="9" letter-spacing="2" fill="#6B6550">VIVO</text>

  <!-- Divider -->
  <line x1="28" y1="78" x2="572" y2="78" stroke="{color_gold}" stroke-width="0.5" stroke-opacity="0.3"/>
  <line x1="28" y1="78" x2="120" y2="78" stroke="{color_gold}" stroke-width="1.5" stroke-opacity="0.8"/>

  <!-- Signal label -->
  <text x="28" y="102" font-family="monospace" font-size="9" letter-spacing="4" fill="#6B6550">SEÑAL VALIDADA · IA ACTIVA</text>

  <!-- Big signal -->
  <text x="26" y="168" font-family="Georgia,serif" font-size="64" font-weight="bold" letter-spacing="4" fill="{color_main}">{arrow}</text>

  <!-- Price -->
  <text x="28" y="200" font-family="monospace" font-size="11" letter-spacing="2" fill="#6B6550">PRECIO DE ENTRADA</text>
  <text x="28" y="222" font-family="Georgia,serif" font-size="28" font-weight="bold" fill="#F0D080">${price:.2f}</text>

  <!-- Divider 2 -->
  <line x1="28" y1="238" x2="572" y2="238" stroke="{color_main}" stroke-width="0.5" stroke-opacity="0.3"/>

  <!-- TP -->
  <text x="28" y="262" font-family="monospace" font-size="9" letter-spacing="3" fill="#6B6550">TAKE PROFIT</text>
  <text x="28" y="282" font-family="Georgia,serif" font-size="22" font-weight="bold" fill="{color_main}">${tp:.2f}</text>

  <!-- SL -->
  <text x="220" y="262" font-family="monospace" font-size="9" letter-spacing="3" fill="#6B6550">STOP LOSS</text>
  <text x="220" y="282" font-family="Georgia,serif" font-size="22" font-weight="bold" fill="#FF4466">${sl:.2f}</text>

  <!-- RR -->
  <text x="410" y="262" font-family="monospace" font-size="9" letter-spacing="3" fill="#6B6550">RATIO R/B</text>
  <text x="410" y="282" font-family="Georgia,serif" font-size="22" font-weight="bold" fill="{color_gold}">{rr}:1</text>

  <!-- Bottom bar -->
  <rect x="0" y="300" width="600" height="40" fill="#111109"/>
  <line x1="0" y1="300" x2="600" y2="300" stroke="{color_gold}" stroke-width="0.5" stroke-opacity="0.3"/>

  <!-- Stats bottom -->
  <text x="28" y="318" font-family="monospace" font-size="8" letter-spacing="2" fill="#6B6550">CONFIANZA</text>
  <text x="28" y="332" font-family="Georgia,serif" font-size="13" fill="{color_main}">{confidence}%</text>

  <text x="140" y="318" font-family="monospace" font-size="8" letter-spacing="2" fill="#6B6550">ATR</text>
  <text x="140" y="332" font-family="Georgia,serif" font-size="13" fill="{color_gold}">${atr:.2f}</text>

  <text x="230" y="318" font-family="monospace" font-size="8" letter-spacing="2" fill="#6B6550">SESIÓN</text>
  <text x="230" y="332" font-family="Georgia,serif" font-size="13" fill="{color_gold}">{session}</text>

  <text x="380" y="318" font-family="monospace" font-size="8" letter-spacing="2" fill="#6B6550">BOT</text>
  <text x="380" y="332" font-family="Georgia,serif" font-size="13" fill="#6B6550">aurum-signals.onrender.com</text>
</svg>'''
    return svg.encode('utf-8')


def send_telegram_photo(token, chat_id, svg_bytes, caption):
    """Envía imagen SVG como documento a Telegram con caption"""
    try:
        boundary = "AurumBoundary"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
            f"{chat_id}\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="caption"\r\n\r\n'
            f"{caption}\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="document"; filename="aurum_signal.svg"\r\n'
            f"Content-Type: image/svg+xml\r\n\r\n"
        ).encode() + svg_bytes + f"\r\n--{boundary}--\r\n".encode()

        url = f"https://api.telegram.org/bot{token}/sendDocument"
        req = urllib.request.Request(url, data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            result = json.loads(r.read().decode())
            if result.get("ok"):
                print("  📸 Imagen enviada a Telegram")
            else:
                print(f"  ⚠ Telegram foto: {result.get('description')}")
    except Exception as e:
        print(f"  ⚠ Error enviando imagen: {e}")

def send_telegram_direct(token, chat, msg, image_data=None):
    try:
        # Send image if provided
        if image_data:
            send_telegram_photo(token, chat, image_data, msg)
        else:
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
