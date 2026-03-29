"""
AURUM - Bot de Señales XAU/USD v6.0
Sin librerias externas. Solo Python 3.
Mejoras v6: OHLC real, ML validation split, thread safety, retry APIs, frontend SCALP v2
"""
import http.server, threading, json, urllib.request, urllib.parse
import csv, os, time
from datetime import datetime, timezone
import datetime as _dt

# ── THREAD SAFETY ─────────────────────────────────────────
_data_lock = threading.Lock()  # protege price_history, _scalp_prices_*, _ohlc_candles

# ── RETRY HELPER ──────────────────────────────────────────
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
    e = _cache_store.get(key)
    if e and time.time() - e["ts"] < ttl: return e["val"]
    return None
def set_cache(key, val):
    _cache_store[key] = {"val": val, "ts": time.time()}

# ── HISTORIAL GLOBAL DE PRECIOS (thread-safe) ────────────
price_history = []
def push_price(p):
    with _data_lock:
        price_history.append(p)
        if len(price_history) > 500:
            del price_history[:-500]

# ── FILTRO DE NOTICIAS ────────────────────────────────────
HIGH_IMPACT_EVENTS = [
    (1, 13, 30, "CPI USA"),
    (2, 14,  0, "FOMC Minutes"),
    (4, 13, 30, "NFP / Jobs"),
    (4, 13, 30, "Unemployment"),
]
PAUSE_MINUTES_BEFORE = 30
PAUSE_MINUTES_AFTER  = 30

def is_news_time():
    now = _dt.datetime.utcnow()
    dow = now.weekday()
    for (day, hour, minute, name) in HIGH_IMPACT_EVENTS:
        if dow != day: continue
        event_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        diff = (now - event_time).total_seconds() / 60
        if -PAUSE_MINUTES_BEFORE <= diff <= PAUSE_MINUTES_AFTER:
            return True, name
    return False, None

# ── MOTOR IA ─────────────────────────────────────────────
import math, pickle

class ModelLayer:
    @staticmethod
    def get_signal(accuracy, drift_score, epochs):
        acc_norm   = max(0.0, min(1.0, (accuracy - 55) / 30))
        drift_norm = max(0.0, 1.0 - drift_score)
        epoch_norm = min(1.0, epochs / 500)
        score = (acc_norm * 0.5) + (drift_norm * 0.35) + (epoch_norm * 0.15)
        level = "STRONG" if score > 0.7 else "MODERATE" if score > 0.4 else "WEAK"
        return round(score, 3), level

class MarketLayer:
    @staticmethod
    def get_signal(prices, is_news_active=False):
        h = _dt.datetime.utcnow().hour
        if   8 <= h < 17:  session_factor = 1.0
        elif 7 <= h < 8:   session_factor = 0.7
        elif 17 <= h < 22: session_factor = 0.7
        else:              session_factor = 0.2
        vol_factor = 1.0
        if len(prices) >= 15:
            atr = sum(abs(prices[i]-prices[i-1]) for i in range(len(prices)-14, len(prices))) / 14
            atr_pct = atr / (prices[-1] + 1e-9) * 100
            if   atr_pct < 0.05: vol_factor = 0.3
            elif atr_pct < 0.1:  vol_factor = 0.6
            elif atr_pct < 0.5:  vol_factor = 1.0
            elif atr_pct < 1.0:  vol_factor = 0.8
            else:                vol_factor = 0.4
        news_factor = 0.0 if is_news_active else 1.0
        score = (session_factor * 0.4) + (vol_factor * 0.4) + (news_factor * 0.2)
        level = "IDEAL" if score > 0.7 else "ACCEPTABLE" if score > 0.4 else "HOSTILE"
        return round(score, 3), level, {
            "session": round(session_factor, 2),
            "volatility": round(vol_factor, 2),
            "news": round(news_factor, 2),
        }

class ControlLayer:
    STATES = ["FULL", "CONSERVATIVE", "MINIMAL", "PAUSED"]
    @staticmethod
    def decide(model_signal, model_level, market_signal, market_level):
        combined = (model_signal * 0.6) + (market_signal * 0.4)
        if market_level == "HOSTILE":
            state, lock = "PAUSED", 600
        elif model_level == "WEAK" and market_level != "IDEAL":
            state, lock = "MINIMAL", 1200
        elif model_level == "WEAK" or market_level == "ACCEPTABLE":
            state, lock = "CONSERVATIVE", 1800
        else:
            state, lock = "FULL", 1800
        thresholds = {
            "FULL":         {"buy": 0.63, "sell": 0.37},
            "CONSERVATIVE": {"buy": 0.70, "sell": 0.30},
            "MINIMAL":      {"buy": 0.78, "sell": 0.22},
            "PAUSED":       {"buy": 1.01, "sell": -0.01},
        }
        return {"state": state, "lock_sec": lock,
                "combined": round(combined, 3), "thresholds": thresholds[state]}

_current_control = {
    "state": "FULL",
    "thresholds": {"buy": 0.63, "sell": 0.37},
    "lock_until": 0,
    "combined": 1.0,
}

def update_control_state(ai_instance, prices, is_news=False):
    global _current_control
    if not prices: return _current_control
    m_sig,  m_lvl              = ModelLayer.get_signal(ai_instance.accuracy, ai_instance.drift_score, ai_instance.epochs)
    mk_sig, mk_lvl, mk_factors = MarketLayer.get_signal(prices, is_news)
    decision = ControlLayer.decide(m_sig, m_lvl, mk_sig, mk_lvl)
    now = time.time()
    if decision["state"] != _current_control["state"]:
        if now >= _current_control["lock_until"]:
            print(f"  🎛 CONTROL: {_current_control['state']} → {decision['state']}")
            _current_control = {
                "state":      decision["state"],
                "thresholds": decision["thresholds"],
                "lock_until": now + decision["lock_sec"],
                "combined":   decision["combined"],
            }
        else:
            remaining = int((_current_control["lock_until"] - now) / 60)
            print(f"  🔒 LOCK: {_current_control['state']} | {remaining}min")
    else:
        _current_control["thresholds"] = decision["thresholds"]
        _current_control["combined"]   = decision["combined"]
    return _current_control

class LogisticModel:
    def __init__(self, n_features=11):
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
    def __init__(self):
        self.thresholds = {}
        self.trained = False

    def train(self, X, y):
        if len(X) < 10: return
        for fi in range(len(X[0])):
            vals = [x[fi] for x in X]
            self.thresholds[fi] = sum(vals) / len(vals)
        self.trained = True

    def predict(self, x):
        if not self.trained: return 0.5
        score = 0.0
        weights = [0.15, 0.12, 0.12, 0.08, 0.08, 0.08, 0.08, 0.08, 0.07, 0.07, 0.07]
        for fi, (thresh, w) in enumerate(zip(self.thresholds.values(), weights)):
            if fi < len(x):
                score += w * (1.0 if x[fi] > thresh else 0.0)
        return score

class AurumAI:
    MODEL_FILE   = "/data/aurum_model_v2.pkl"
    MODEL_DIR    = "/data"
    DATASET_FILE = "/data/aurum_dataset.pkl"
    MAX_VERSIONS = 3
    MIN_ACCURACY = 55.0
    _low_acc_streak    = 0
    _high_drift_streak = 0
    CONSERVATIVE_MODE  = False
    ROLLBACK_STREAK    = 3
    DRIFT_STREAK       = 5
    _state_lock_until  = 0
    _current_state     = "OK"
    STATE_LOCK_SECONDS = 1800

    def __init__(self):
        self.logistic         = LogisticModel(n_features=11)
        self.tree             = DecisionTreeModel()
        self.trained          = False
        self.accuracy         = 0.0
        self.epochs           = 0
        self.drift_score      = 0.0
        self.accuracy_by_hour = {}
        self.accuracy_by_dow  = {}
        self.recent_errors    = []
        self.load()

    def save(self):
        data = {
            "logistic_w": self.logistic.w, "logistic_b": self.logistic.b,
            "logistic_trained": self.logistic.trained,
            "tree_thresh": self.tree.thresholds, "tree_trained": self.tree.trained,
            "trained": self.trained, "accuracy": self.accuracy,
            "epochs": self.epochs, "drift_score": self.drift_score,
            "accuracy_by_hour": self.accuracy_by_hour, "accuracy_by_dow": self.accuracy_by_dow,
        }
        for target in [AurumAI.MODEL_FILE, "aurum_model_v2.pkl"]:
            try:
                os.makedirs(os.path.dirname(target) if os.path.dirname(target) else ".", exist_ok=True)
                tmp = target + ".tmp"
                with open(tmp, "wb") as f: pickle.dump(data, f)
                os.replace(tmp, target)
                print(f"  💾 MODEL SAVED: {target} | acc={self.accuracy:.1f}%")
                return
            except Exception as e:
                print(f"  ❌ SAVE ERROR: {target} | {e}")

    def load(self):
        for path in [AurumAI.MODEL_FILE, "aurum_model_v2.pkl"]:
            if not os.path.exists(path):
                print(f"  📂 MODEL CHECK: {path} | exists=False")
                continue
            print(f"  📂 MODEL CHECK: {path} | exists=True")
            try:
                with open(path, "rb") as f: d = pickle.load(f)
                self.logistic.w        = d.get("logistic_w", [0.0]*11)
                # v6: compatibilidad con modelos viejos de 8 features
                if len(self.logistic.w) < 11:
                    self.logistic.w.extend([0.0] * (11 - len(self.logistic.w)))
                self.logistic.b        = d.get("logistic_b", 0.0)
                self.logistic.trained  = d.get("logistic_trained", False)
                self.tree.thresholds   = d.get("tree_thresh", {})
                self.tree.trained      = d.get("tree_trained", False)
                self.trained           = d.get("trained", False)
                self.accuracy          = d.get("accuracy", 0.0)
                self.epochs            = d.get("epochs", 0)
                self.drift_score       = d.get("drift_score", 0.0)
                self.accuracy_by_hour  = d.get("accuracy_by_hour", {})
                self.accuracy_by_dow   = d.get("accuracy_by_dow", {})
                print(f"  ✅ MODEL LOADED: acc={self.accuracy:.1f}% epochs={self.epochs}")
                return
            except Exception as e:
                print(f"  ❌ MODEL LOAD ERROR: {path} | {e}")
        print("  ⚠ MODEL NOT FOUND — starting fresh")

    def save_version(self):
        try:
            import shutil, glob
            if os.path.exists(AurumAI.MODEL_FILE):
                versions = sorted(glob.glob(f"{AurumAI.MODEL_DIR}/aurum_model_v*.bak"))
                ver_num  = len(versions) + 1
                ver_path = f"{AurumAI.MODEL_DIR}/aurum_model_v{ver_num}.bak"
                shutil.copy2(AurumAI.MODEL_FILE, ver_path)
                versions = sorted(glob.glob(f"{AurumAI.MODEL_DIR}/aurum_model_v*.bak"))
                while len(versions) > AurumAI.MAX_VERSIONS:
                    os.remove(versions[0]); versions = versions[1:]
        except Exception as e:
            print(f"  ⚠ Version save error: {e}")

    def rollback(self):
        try:
            import glob, shutil
            versions = sorted(glob.glob(f"{AurumAI.MODEL_DIR}/aurum_model_v*.bak"))
            if not versions: return False
            shutil.copy2(versions[-1], AurumAI.MODEL_FILE)
            self.load()
            return True
        except Exception as e:
            print(f"  ❌ ROLLBACK ERROR: {e}")
            return False

    def save_dataset(self, X, y):
        try:
            os.makedirs(AurumAI.MODEL_DIR, exist_ok=True)
            existing = {"X": [], "y": [], "timestamps": [], "weights": []}
            if os.path.exists(AurumAI.DATASET_FILE):
                with open(AurumAI.DATASET_FILE, "rb") as f:
                    existing = pickle.load(f)
                if "weights" not in existing:
                    existing["weights"] = [1.0] * len(existing["X"])
            now_str = _dt.datetime.utcnow().isoformat()
            existing["X"].extend(X); existing["y"].extend(y)
            existing["timestamps"].extend([now_str] * len(X))
            existing["weights"].extend([1.0] * len(X))
            now = _dt.datetime.utcnow()
            for i in range(len(existing["timestamps"])):
                try:
                    ts = _dt.datetime.fromisoformat(existing["timestamps"][i])
                    hours_old = (now - ts).total_seconds() / 3600
                    existing["weights"][i] = max(0.2, 1.0 - (hours_old / 24) * 0.2)
                except: pass
            if len(existing["X"]) > 2000:
                combined = sorted(zip(existing["weights"], existing["X"], existing["y"], existing["timestamps"]), reverse=True)[:2000]
                existing["weights"]    = [c[0] for c in combined]
                existing["X"]          = [c[1] for c in combined]
                existing["y"]          = [c[2] for c in combined]
                existing["timestamps"] = [c[3] for c in combined]
            tmp = AurumAI.DATASET_FILE + ".tmp"
            with open(tmp, "wb") as f: pickle.dump(existing, f)
            os.replace(tmp, AurumAI.DATASET_FILE)
        except Exception as e:
            print(f"  ⚠ Dataset save error: {e}")

    def _get_lock_duration(self):
        if self.drift_score > 0.8 and self.accuracy < 50:   return 300
        if self.drift_score > 0.65 or self.accuracy < AurumAI.MIN_ACCURACY: return 600
        if self.drift_score > 0.4:  return 1200
        return AurumAI.STATE_LOCK_SECONDS

    def _can_change_state(self, new_state):
        now = time.time()
        if new_state == AurumAI._current_state: return True
        if now < AurumAI._state_lock_until:
            if self.drift_score > 0.8 and self.accuracy < 50:
                if now - (AurumAI._state_lock_until - AurumAI.STATE_LOCK_SECONDS) > 300:
                    return True
            return False
        return True

    def _set_state(self, new_state):
        if new_state != AurumAI._current_state:
            lock_duration = self._get_lock_duration()
            print(f"  🔄 STATE: {AurumAI._current_state} → {new_state} | lock={lock_duration//60}min")
            AurumAI._current_state    = new_state
            AurumAI._state_lock_until = time.time() + lock_duration

    def health_check(self, prev_accuracy=None):
        report = {
            "trained": self.trained, "accuracy": round(self.accuracy, 1),
            "epochs": self.epochs, "drift_score": round(self.drift_score, 2),
            "is_drifting": self.is_market_drifting(),
            "conservative": AurumAI.CONSERVATIVE_MODE,
            "current_state": AurumAI._current_state,
            "status": "OK", "action": None,
        }
        if self.trained and self.accuracy < AurumAI.MIN_ACCURACY:
            AurumAI._low_acc_streak += 1
            if AurumAI._low_acc_streak >= AurumAI.ROLLBACK_STREAK:
                if self._can_change_state("ROLLBACK"):
                    rolled = self.rollback()
                    AurumAI._low_acc_streak = 0
                    report["status"] = "DEGRADED"
                    report["action"] = "ROLLBACK_OK" if rolled else "ROLLBACK_FAILED"
                    self._set_state("ROLLBACK")
            else:
                if self._can_change_state("CONSERVATIVE"):
                    AurumAI.CONSERVATIVE_MODE = True
                    report["status"] = "CONSERVATIVE"
                    report["action"] = "CONSERVATIVE_MODE"
                    self._set_state("CONSERVATIVE")
        else:
            AurumAI._low_acc_streak = max(0, AurumAI._low_acc_streak - 1)
            if self._can_change_state("OK"):
                AurumAI.CONSERVATIVE_MODE = False
                self._set_state("OK")
        if self.is_market_drifting():
            AurumAI._high_drift_streak += 1
            if AurumAI._high_drift_streak >= AurumAI.DRIFT_STREAK:
                if self._can_change_state("DRIFT_CONFIRMED"):
                    report["status"] = "DRIFT_CONFIRMED"; report["action"] = "SIGNALS_PAUSED"
                    AurumAI.CONSERVATIVE_MODE = True; self._set_state("DRIFT_CONFIRMED")
            else:
                if self._can_change_state("DRIFT_WARNING"):
                    AurumAI.CONSERVATIVE_MODE = True; report["status"] = "DRIFT_WARNING"
                    report["action"] = "CONSERVATIVE_MODE"; self._set_state("DRIFT_WARNING")
        else:
            AurumAI._high_drift_streak = max(0, AurumAI._high_drift_streak - 1)
        if prev_accuracy and self.accuracy < prev_accuracy - 25:
            self.rollback(); report["action"] = "ROLLBACK_BIG_DROP"
        elif prev_accuracy and self.accuracy < prev_accuracy - 15:
            AurumAI.CONSERVATIVE_MODE = True; report["status"] = "ACCURACY_DROP"
            report["action"] = "CONSERVATIVE_MODE"
        return report

    def extract_features(self, prices, hour=None, dow=None):
        """Extrae features para ML. v6: +hora, +día, +vol_relativa = 11 features."""
        if len(prices) < 35: return None
        p = prices
        n = 14; g = l = 0
        for i in range(len(p)-n, len(p)):
            d = p[i]-p[i-1]
            if d > 0: g += d
            else: l -= d
        ag, al = g/n, l/n
        rsi = (100-100/(1+ag/al))/100 if al > 0 else 1.0
        def ema(arr, n):
            k = 2/(n+1); e = sum(arr[:n])/n
            for v in arr[n:]: e = v*k+e*(1-k)
            return e
        e9, e21 = ema(p,9), ema(p,21)
        ema_cross = max(-1, min(1, (e9-e21)/(e21+1e-9)*100))
        e12, e26 = ema(p,12), ema(p,26)
        macd_norm = max(-1, min(1, (e12-e26)/(p[-1]*0.01+1e-9)))
        atrs = [max(p[i]*1.004-p[i]*0.996, abs(p[i]*1.004-p[i-1]), abs(p[i]*0.996-p[i-1]))
                for i in range(len(p)-14, len(p))]
        atr_val = sum(atrs)/14
        atr_norm = max(0, min(2, atr_val/(p[-1]*0.01+1e-9)))
        sma20 = sum(p[-20:])/20
        pvs   = max(-2, min(2, (p[-1]-sma20)/(sma20+1e-9)*100))
        mom5  = max(-2, min(2, (p[-1]-p[-6])/(p[-6]+1e-9)*100))
        mom10 = max(-2, min(2, (p[-1]-p[-11])/(p[-11]+1e-9)*100)) if len(p) > 11 else 0
        rets  = [(p[i]-p[i-1])/p[i-1] for i in range(len(p)-10, len(p))]
        mean_r = sum(rets)/len(rets)
        vol = math.sqrt(sum((r-mean_r)**2 for r in rets)/len(rets)) * 100
        # v6: features de contexto temporal
        h = hour if hour is not None else _dt.datetime.utcnow().hour
        d = dow  if dow  is not None else _dt.datetime.utcnow().weekday()
        hour_norm = math.sin(2 * math.pi * h / 24)  # cíclico 0-1
        dow_norm  = d / 6.0  # 0=lunes, 1=domingo
        # v6: volatilidad relativa (ATR actual vs ATR promedio histórico)
        vol_rel = max(0, min(2, vol / 0.15)) if vol > 0 else 0.5  # normalizado ~0.15% es promedio oro
        return [rsi, ema_cross, macd_norm, atr_norm, pvs, mom5, mom10,
                max(0, min(2, vol)), hour_norm, dow_norm, vol_rel]

    def detect_drift(self):
        if len(self.recent_errors) < 10: return 0.0
        self.drift_score = sum(self.recent_errors[-10:]) / 10
        return self.drift_score

    def is_market_drifting(self):
        return self.detect_drift() > 0.65

    def train(self, prices):
        """v6: train/validation split 80/20 para accuracy real."""
        if len(prices) < 60: return False
        now = _dt.datetime.utcnow()
        X, y, hours, dows = [], [], [], []
        for i in range(40, len(prices)-5):
            mins_ago = (len(prices)-i)*5
            t = now - _dt.timedelta(minutes=mins_ago)
            features = self.extract_features(prices[i-35:i], hour=t.hour, dow=t.weekday())
            if not features: continue
            future_ret = (prices[i+4]-prices[i])/prices[i]
            label = 1 if future_ret > 0.0005 else 0
            X.append(features); y.append(label)
            hours.append(t.hour); dows.append(t.weekday())
        if len(X) < 30: return False
        # v6: split 80/20 — entrena en 80%, mide accuracy en 20%
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_val,   y_val   = X[split:], y[split:]
        h_val, d_val     = hours[split:], dows[split:]
        self.logistic.train(X_train, y_train)
        self.tree.train(X_train, y_train)
        # Medir accuracy SOLO en validation set (out-of-sample)
        correct = 0
        for xi,yi,h,dow in zip(X_val,y_val,h_val,d_val):
            p_ens = 0.6*self.logistic.predict(xi) + 0.4*self.tree.predict(xi)
            pred  = 1 if p_ens >= 0.5 else 0
            is_correct = int(pred == yi)
            correct += is_correct
            self.recent_errors.append(1-is_correct)
            hk = str(h)
            if hk not in self.accuracy_by_hour: self.accuracy_by_hour[hk] = []
            self.accuracy_by_hour[hk].append(is_correct)
            if len(self.accuracy_by_hour[hk]) > 50: self.accuracy_by_hour[hk] = self.accuracy_by_hour[hk][-50:]
            dk = str(dow)
            if dk not in self.accuracy_by_dow: self.accuracy_by_dow[dk] = []
            self.accuracy_by_dow[dk].append(is_correct)
            if len(self.accuracy_by_dow[dk]) > 50: self.accuracy_by_dow[dk] = self.accuracy_by_dow[dk][-50:]
        if len(self.recent_errors) > 50: self.recent_errors = self.recent_errors[-50:]
        val_n = len(X_val)
        self.accuracy = correct/val_n*100 if val_n > 0 else 0
        self.epochs   = len(X)
        self.trained  = True
        self.save()
        return True

    def predict_proba(self, prices):
        if not self.trained: return None
        features = self.extract_features(prices[-35:])
        if not features: return None
        return 0.6*self.logistic.predict(features) + 0.4*self.tree.predict(features)

    def get_hour_accuracy(self):
        h = str(_dt.datetime.utcnow().hour)
        if h in self.accuracy_by_hour and self.accuracy_by_hour[h]:
            data = self.accuracy_by_hour[h]
            return sum(data)/len(data)*100
        return None

    def get_dow_accuracy(self):
        d = str(_dt.datetime.utcnow().weekday())
        if d in self.accuracy_by_dow and self.accuracy_by_dow[d]:
            data = self.accuracy_by_dow[d]
            return sum(data)/len(data)*100
        return None

_ai = AurumAI()
_ai_last_train  = 0
_ai_last_backup = 0

def ai_train_if_needed(prices):
    global _ai_last_train, _ai_last_backup
    now = time.time()
    if now - _ai_last_train > 300 and len(prices) >= 60:
        _ai.train(prices)
        _ai_last_train = now
        if _ai.trained:
            drift = _ai.detect_drift()
            h_acc = _ai.get_hour_accuracy()
            d_acc = _ai.get_dow_accuracy()
            print(f"  🤖 Ensemble: {_ai.accuracy:.1f}% | Drift: {drift:.2f} | {_ai.epochs} muestras")
            if h_acc: print(f"  🕐 Acc hora: {h_acc:.1f}%")
            if d_acc: print(f"  📅 Acc hoy: {d_acc:.1f}%")
            if _ai.is_market_drifting():
                print("  ⚠ DRIFT DETECTADO")
    if now - _ai_last_backup > 3600 and _ai.trained:
        _ai.save(); _ai_last_backup = now
        print("  💾 Backup automático guardado")

def ai_predict(prices):
    if not _ai.trained or len(prices) < 35: return None, None
    ctrl_state = _current_control["state"]
    if ctrl_state == "PAUSED": return None, "PAUSED"
    prob = _ai.predict_proba(prices)
    if prob is None: return None, None
    buy_thresh  = _current_control["thresholds"]["buy"]
    sell_thresh = _current_control["thresholds"]["sell"]
    if   prob >= buy_thresh:  signal = "COMPRAR"
    elif prob <= sell_thresh: signal = "VENDER"
    else:                     signal = "ESPERAR"
    return prob, signal

# ── [FIX 3] BACKGROUND WORKER (thread-safe) ──────────────
def background_worker():
    while True:
        time.sleep(300)
        try:
            news_on, _ = is_news_time()
            with _data_lock:
                ph = price_history[-50:] if len(price_history) >= 50 else [0]
            update_control_state(_ai, ph, is_news=news_on)
            with _data_lock:
                ph_full = list(price_history)
            if len(ph_full) >= 60:
                ai_train_if_needed(ph_full)
            # v6: refrescar OHLC candles en background
            get_historical_ohlc("5min", 200)
            get_historical_ohlc("15min", 100)
        except Exception as e:
            print(f"  ⚠ BG worker: {e}")

PORT     = int(os.environ.get("PORT", 8765))
LOG_FILE = "aurum_operaciones.csv"

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
    try:
        os.makedirs("/data", exist_ok=True)
        with open("/data/telegram_config.json", "w") as _f:
            json.dump({"token": TELEGRAM_TOKEN, "chat": TELEGRAM_CHAT_ID}, _f)
        print(f"  ✓ Telegram config guardado en disco")
    except Exception as _e:
        print(f"  ⚠ No se pudo guardar config: {_e}")
else:
    try:
        with open("/data/telegram_config.json") as _f:
            _cfg = json.load(_f)
        TELEGRAM_TOKEN   = _cfg.get("token", "")
        TELEGRAM_CHAT_ID = _cfg.get("chat", "")
        print(f"  ✓ Telegram config cargado desde disco")
    except: pass

def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Fecha","Hora GMT","Señal","Precio Entrada","TP","SL","ATR","RR","Confianza%","Sesión"])

def log_signal(signal, price, tp, sl, atr, rr, confidence, session):
    now = datetime.now(timezone.utc)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M"),
            signal, f"{price:.2f}", f"{tp:.2f}", f"{sl:.2f}", f"{atr:.2f}", f"{rr}", confidence, session])

def read_log():
    if not os.path.exists(LOG_FILE): return []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)][-20:]

TWELVE_API_KEY = os.environ.get("TWELVE_API_KEY", "dd53883de1a84cccaf65bf7f4e7a4756")
_mtf_cache = {"prices_1h": [], "last_update": 0}

def get_htf_trend():
    now = time.time()
    if now - _mtf_cache["last_update"] < 300 and _mtf_cache["prices_1h"]:
        prices_1h = _mtf_cache["prices_1h"]
    else:
        candles = get_historical_ohlc("1h", 50)
        if not candles or len(candles) < 21: return "neutral"
        prices_1h = [c["close"] for c in candles]
        _mtf_cache["prices_1h"]   = prices_1h
        _mtf_cache["last_update"] = now
    def ema(arr, n):
        if len(arr) < n: return None
        k = 2/(n+1); e = sum(arr[:n])/n
        for v in arr[n:]: e = v*k + e*(1-k)
        return e
    e9, e21 = ema(prices_1h, 9), ema(prices_1h, 21)
    if e9 and e21:
        if e9 > e21 * 1.001: return "up"
        elif e9 < e21 * 0.999: return "down"
    return "neutral"

# ── AURUM SCALP v2 — MOTOR DE SEÑALES ───────────────────
import datetime as _sdt

# ── SESIONES ─────────────────────────────────────────────
def is_trading_session():
    """Londres 3-11 UTC + NY 12-17 UTC. No Asia."""
    h = _sdt.datetime.utcnow().hour
    return (3 <= h < 11) or (12 <= h < 17)

def get_session_name():
    h = _sdt.datetime.utcnow().hour
    if  3 <= h < 11: return "Londres 🇬🇧"
    if 12 <= h < 17: return "New York 🇺🇸"
    return "Fuera de sesión 🌙"

# ── SCORING SCALP v2 ─────────────────────────────────────
SCORE_SNIPER = 85   # entrada agresiva
SCORE_NORMAL = 70   # trade normal

# ── DETECCIÓN DE ESTRUCTURA (v6: OHLC real) ──────────────
def _swing_highs_ohlc(candles, n=3):
    """v6: Swing highs usando candle.high real."""
    highs = []
    for i in range(n, len(candles)-n):
        h = candles[i]["h"]
        if all(h >= candles[i-j]["h"] for j in range(1,n+1)) and \
           all(h >= candles[i+j]["h"] for j in range(1,n+1)):
            highs.append((i, h))
    return highs

def _swing_lows_ohlc(candles, n=3):
    """v6: Swing lows usando candle.low real."""
    lows = []
    for i in range(n, len(candles)-n):
        lo = candles[i]["l"]
        if all(lo <= candles[i-j]["l"] for j in range(1,n+1)) and \
           all(lo <= candles[i+j]["l"] for j in range(1,n+1)):
            lows.append((i, lo))
    return lows

# Fallback: funciones originales para cuando no hay OHLC
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

def detect_sweep_and_rejection(prices, bias, candles=None):
    """v6: Sweep + rechazo usando OHLC real si disponible."""
    if candles and len(candles) >= 20:
        return _detect_sweep_ohlc(candles, bias)
    # fallback a close-only
    if len(prices) < 20: return False, 0, 0
    p = prices
    if bias == "bullish":
        lows = _swing_lows(p[:-3], n=2)
        if not lows: return False, 0, 0
        prev_low_idx, prev_low = lows[-1]
        recent_low = min(p[-5:])
        recent_close = p[-1]
        if recent_low < prev_low and recent_close > prev_low:
            candle_range = max(p[-4:]) - min(p[-4:]) + 1e-9
            wick = prev_low - recent_low
            rejection = wick / candle_range
            return True, round(prev_low, 2), round(rejection, 2)
    if bias == "bearish":
        highs = _swing_highs(p[:-3], n=2)
        if not highs: return False, 0, 0
        prev_high_idx, prev_high = highs[-1]
        recent_high = max(p[-5:])
        recent_close = p[-1]
        if recent_high > prev_high and recent_close < prev_high:
            candle_range = max(p[-4:]) - min(p[-4:]) + 1e-9
            wick = recent_high - prev_high
            rejection = wick / candle_range
            return True, round(prev_high, 2), round(rejection, 2)
    return False, 0, 0

def _detect_sweep_ohlc(candles, bias):
    """v6: Sweep detection con OHLC real — wicks reales, no aproximados."""
    c = candles
    if bias == "bullish":
        lows = _swing_lows_ohlc(c[:-3], n=2)
        if not lows: return False, 0, 0
        prev_low_idx, prev_low = lows[-1]
        # ¿Alguna vela reciente hizo wick por debajo del swing low?
        recent_lows = [x["l"] for x in c[-5:]]
        recent_close = c[-1]["c"]
        min_wick = min(recent_lows)
        if min_wick < prev_low and recent_close > prev_low:
            # Rechazo = tamaño del wick / rango total de la vela que hizo el sweep
            sweep_candle = min(range(len(c[-5:])), key=lambda j: c[-5+j]["l"])
            sc = c[-5 + sweep_candle]
            candle_range = sc["h"] - sc["l"] + 1e-9
            wick = prev_low - sc["l"]  # wick inferior real
            rejection = wick / candle_range
            return True, round(prev_low, 2), round(rejection, 2)
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
            wick = sc["h"] - prev_high  # wick superior real
            rejection = wick / candle_range
            return True, round(prev_high, 2), round(rejection, 2)
    return False, 0, 0

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
    """v6: FVG con OHLC real — gap entre high vela1 y low vela3."""
    if candles and len(candles) >= 5:
        return _detect_fvg_ohlc(candles, bias)
    # fallback close-only
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
    """v6: FVG real — bullish: low[vela3] > high[vela1], bearish: high[vela3] < low[vela1]."""
    for i in range(len(candles)-3, len(candles)):
        if i < 2: continue
        c1, c2, c3 = candles[i-2], candles[i-1], candles[i]
        if bias == "bullish":
            # Gap alcista: low de vela 3 > high de vela 1
            if c3["l"] > c1["h"]:
                gap = c3["l"] - c1["h"]
                if gap > c3["c"] * 0.0005:
                    return True, round(c1["h"], 2), round(c3["l"], 2)
        if bias == "bearish":
            # Gap bajista: high de vela 3 < low de vela 1
            if c3["h"] < c1["l"]:
                gap = c1["l"] - c3["h"]
                if gap > c3["c"] * 0.0005:
                    return True, round(c3["h"], 2), round(c1["l"], 2)
    return False, 0, 0

def get_ema_bias(prices):
    if len(prices) < 22: return None
    def ema(arr, n):
        k = 2/(n+1); e = sum(arr[:n])/n
        for v in arr[n:]: e = v*k + e*(1-k)
        return e
    e9  = ema(prices, 9)
    e21 = ema(prices, 21)
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

def detect_confirmation_candle(prices, bias):
    if len(prices) < 4: return False
    last4 = prices[-4:]
    body  = abs(last4[-1] - last4[-2])
    rng   = max(last4) - min(last4) + 1e-9
    momentum = body / rng
    if bias == "bullish" and last4[-1] > last4[-2] and momentum > 0.4:
        return True
    if bias == "bearish" and last4[-1] < last4[-2] and momentum > 0.4:
        return True
    return False

def get_ml_filter(prices):
    if not _ai.trained or len(prices) < 35:
        return "MED", 0.5
    prob = _ai.predict_proba(prices)
    if prob is None: return "MED", 0.5
    if   prob >= 0.7: return "HIGH", round(prob, 2)
    elif prob >= 0.5: return "MED",  round(prob, 2)
    else:             return "LOW",  round(prob, 2)

# ── COOLDOWN ──────────────────────────────────────────────
_last_scalp_signal = {"key": "", "time": 0}

# ── PRECIO HISTORIES CON OHLC REAL (thread-safe) ─────────
# v6: almacenamos candles OHLC completas, no solo close
_scalp_prices_1m  = []
_scalp_prices_5m  = []
_scalp_prices_15m = []
# v6: OHLC candle buffers {open, high, low, close}
_ohlc_candles_5m  = []
_ohlc_candles_15m = []

def update_ict_prices(price, candle=None):
    """Alimenta los historiales de precios para el motor.
    v6: acepta candle OHLC opcional {open, high, low, close}."""
    with _data_lock:
        _scalp_prices_1m.append(price)
        if len(_scalp_prices_1m) > 500:
            del _scalp_prices_1m[:-500]
        if len(_scalp_prices_1m) % 5 == 0:
            _scalp_prices_5m.append(price)
            if len(_scalp_prices_5m) > 300:
                del _scalp_prices_5m[:-300]
        if len(_scalp_prices_1m) % 15 == 0:
            _scalp_prices_15m.append(price)
            if len(_scalp_prices_15m) > 200:
                del _scalp_prices_15m[:-200]

def update_ohlc_candles(candles_list, tf="5m"):
    """v6: carga candles OHLC reales desde Twelve Data."""
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

# backward compat aliases — ahora apuntan al mismo objeto siempre
_ict_prices_5m  = _scalp_prices_5m
_ict_prices_15m = _scalp_prices_15m

def run_ict_engine(current_price, atr):
    """
    AURUM SCALP v6 — Motor principal.
    v6: usa OHLC real para Sweep/FVG cuando disponible.
    Orden fijo: Sweep → BOS → FVG → EMA → ML → Score
    """
    global _last_scalp_signal
    import time as _t

    p15 = _scalp_prices_15m if len(_scalp_prices_15m) >= 15 else _scalp_prices_5m
    p5  = _scalp_prices_5m  if len(_scalp_prices_5m)  >= 20 else _scalp_prices_1m
    p1  = _scalp_prices_1m
    # v6: OHLC candles para detección precisa
    ohlc5 = _ohlc_candles_5m if len(_ohlc_candles_5m) >= 20 else None

    if len(p5) < 20: return None

    if not is_trading_session(): return None
    if detect_chop(p5): return None

    bias = get_ema_bias(p15 if len(p15) >= 22 else p5)
    if not bias: return None

    direction = "COMPRAR" if bias == "bullish" else "VENDER"
    score = 0
    details = {"bias": bias, "direction": direction}

    # v6: pasa OHLC candles a sweep y FVG
    swept, sweep_level, rejection = detect_sweep_and_rejection(p5, bias, candles=ohlc5)
    if not swept: return None
    score += 40
    details["sweep"] = f"Sweep ${sweep_level} | rej={rejection:.1f}"

    bos, bos_type = detect_bos_scalp(p5, bias)
    if not bos: return None
    score += 25
    details["bos"] = bos_type

    # v6: FVG con OHLC real
    fvg, fvg_lo, fvg_hi = detect_fvg_scalp(p5, bias, candles=ohlc5)
    if fvg:
        score += 15
        details["fvg"] = f"FVG ${fvg_lo}–${fvg_hi}"
        if fvg_lo <= current_price <= fvg_hi:
            score += 5
            details["fvg_entry"] = "✓ Precio en FVG"

    if bias == get_ema_bias(p5):
        score += 10
        details["ema"] = "EMA 9/21 alineada"

    ml_label, ml_prob = get_ml_filter(p5)
    if ml_label == "LOW":
        return None
    if ml_label == "HIGH":
        score += 10
        details["ml"] = f"ML HIGH ({ml_prob:.0%})"
    else:
        details["ml"] = f"ML MED ({ml_prob:.0%})"
        if score < SCORE_SNIPER - 10: return None

    details["score"] = score
    if score < SCORE_NORMAL: return None

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

    if len(p1) >= 4:
        confirmed = detect_confirmation_candle(p1, bias)
        details["candle"] = "✅ Confirmación 1M" if confirmed else "⏳ Sin confirmación"
        if not confirmed and score < SCORE_SNIPER:
            return None

    sig_key = f"{direction}_{round(current_price/5)*5}"
    now = _t.time()
    if sig_key == _last_scalp_signal["key"] and now - _last_scalp_signal["time"] < 300:
        return None
    _last_scalp_signal = {"key": sig_key, "time": now}

    is_buy  = direction == "COMPRAR"
    if score >= SCORE_SNIPER:
        tp_mult, sl_mult = 2.5, 1.0
    else:
        tp_mult, sl_mult = 1.5, 1.0

    tp = current_price + atr * tp_mult if is_buy else current_price - atr * tp_mult
    sl = current_price - atr * sl_mult if is_buy else current_price + atr * sl_mult
    rr = round(tp_mult / sl_mult, 1)

    session  = get_session_name()
    mode_tag = "SNIPER 🎯" if score >= SCORE_SNIPER else "NORMAL"

    em = "🟢" if is_buy else "🔴"
    reasons = []
    if details.get("sweep"):   reasons.append(f"✔ Liquidity Sweep")
    if details.get("bos"):     reasons.append(f"✔ {details['bos']}")
    if details.get("fvg"):     reasons.append(f"✔ FVG entry")
    if details.get("ema"):     reasons.append(f"✔ EMA alignment")
    if details.get("ml"):      reasons.append(f"✔ {details['ml']}")

    header = "🥇 *AURUM SCALP — SNIPER*" if score >= SCORE_SNIPER else "⚡ *AURUM SCALP SIGNAL*"
    dirstr = "BUY" if is_buy else "SELL"
    reason_str = "\n".join(reasons)
    msg = (
        header + "\n\n"
        "PAIR: XAUUSD\n"
        "DIRECTION: " + dirstr + "\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "ENTRY:  $" + f"{current_price:.2f}" + "\n"
        "SL:     $" + f"{sl:.2f}" + "\n"
        "TP:     $" + f"{tp:.2f}" + "\n\n"
        "RR:         1:" + str(rr) + "\n"
        "CONFIDENCE: " + str(score) + "%\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "REASON:\n" + reason_str + "\n\n"
        "SESSION: " + session + "\n"
        "MODE: " + mode_tag
    )

    return {
        "direction": direction,
        "score":     score,
        "msg":       msg,
        "tp":        tp,
        "sl":        sl,
        "rr":        rr,
        "details":   details,
        "session":   session,
    }

# ── GESTIÓN DE RIESGO ─────────────────────────────────────
_consecutive_losses = 0; _consecutive_wins = 0
_daily_signals = 0; _last_signal_date = ""
MAX_CONSECUTIVE_LOSSES = 3; MAX_DAILY_SIGNALS = 8

def calc_position_size(price, atr, account_pct=0.01):
    if atr <= 0 or price <= 0: return 1.0
    atr_pct = atr / price * 100
    if atr_pct > 0.8: return 0.5
    elif atr_pct > 0.5: return 0.75
    return 1.0

def calc_adaptive_levels(price, atr, signal, trend_strength):
    tp_mult = 1.5 + trend_strength * 0.5; sl_mult = 1.0
    is_buy  = signal == "COMPRAR"
    tp = price + atr * tp_mult if is_buy else price - atr * tp_mult
    sl = price - atr * sl_mult if is_buy else price + atr * sl_mult
    return tp, sl, round(tp_mult / sl_mult, 2), round(tp_mult, 2)

def is_lateral_market(prices, n=20):
    if len(prices) < n: return False
    recent = prices[-n:]
    return (max(recent) - min(recent)) / (min(recent) + 1e-9) * 100 < 0.3

def check_risk_limits():
    global _daily_signals, _last_signal_date
    today = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    if today != _last_signal_date: _daily_signals = 0; _last_signal_date = today
    if _consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
        return False, f"MAX PÉRDIDAS CONSECUTIVAS ({MAX_CONSECUTIVE_LOSSES})"
    if _daily_signals >= MAX_DAILY_SIGNALS:
        return False, f"MAX SEÑALES DIARIAS ({MAX_DAILY_SIGNALS})"
    return True, ""

def register_signal_result(won):
    global _consecutive_losses, _consecutive_wins, _daily_signals
    _daily_signals += 1
    if won: _consecutive_losses = 0; _consecutive_wins += 1
    else:   _consecutive_losses += 1; _consecutive_wins = 0

def get_gold_price():
    c = cached("precio_live", ttl=8)
    if c: return c
    # v6: usa retry helper
    d = _fetch_with_retry("https://api.gold-api.com/price/XAU", timeout=8, retries=3, backoff=1.0)
    if d and "price" in d:
        result = {"price": float(d["price"]), "ch": float(d.get("ch",0)), "chp": float(d.get("chp",0))}
        set_cache("precio_live", result)
        return result
    return None

def get_historical_ohlc(interval="5min", outputsize=150):
    ttl = 60 if interval in ("1min","5min") else 300
    c = cached(f"ohlc_{interval}", ttl=ttl)
    if c: return c
    # v6: usa retry helper
    url = (f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval={interval}"
           f"&outputsize={outputsize}&apikey={TWELVE_API_KEY}")
    data = _fetch_with_retry(url, timeout=10, retries=3, backoff=1.5)
    if data and "values" in data:
        candles = [{"open": float(v["open"]), "high": float(v["high"]),
                    "low":  float(v["low"]),  "close": float(v["close"]),
                    "dt":   v["datetime"]} for v in reversed(data["values"])]
        set_cache(f"ohlc_{interval}", candles)
        # v6: alimentar buffer OHLC para motor SCALP
        if interval == "5min":
            update_ohlc_candles(candles, tf="5m")
        elif interval == "15min":
            update_ohlc_candles(candles, tf="15m")
        print(f"  ✓ {len(candles)} velas {interval}")
        return candles
    elif data:
        print(f"  ⚠ Twelve Data: {data.get('message','sin datos')}")
    return []

def generate_signal_image(signal, price, tp, sl, rr, confidence, session, atr):
    is_buy     = signal == "COMPRAR"
    color_main = "#00E5A0" if is_buy else "#FF4466"
    color_gold = "#C9A84C"
    arrow      = "▲ COMPRAR" if is_buy else "▼ VENDER"
    svg = f'''<svg width="600" height="340" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0A0A08"/><stop offset="100%" stop-color="#111109"/>
    </linearGradient>
  </defs>
  <rect width="600" height="340" fill="url(#bg)"/>
  <rect x="0" y="0" width="600" height="3" fill="{color_main}"/>
  <rect x="1" y="1" width="598" height="338" fill="none" stroke="{color_main}" stroke-width="1" stroke-opacity="0.4"/>
  <text x="28" y="48" font-family="Georgia,serif" font-size="26" font-weight="bold" letter-spacing="8" fill="{color_gold}">AURUM</text>
  <text x="30" y="64" font-family="monospace" font-size="9" letter-spacing="4" fill="#6B6550">SEÑALES XAU/USD · IA</text>
  <circle cx="558" cy="40" r="5" fill="{color_main}"/>
  <text x="28" y="102" font-family="monospace" font-size="9" letter-spacing="4" fill="#6B6550">SEÑAL VALIDADA · IA ACTIVA</text>
  <text x="26" y="168" font-family="Georgia,serif" font-size="64" font-weight="bold" letter-spacing="4" fill="{color_main}">{arrow}</text>
  <text x="28" y="200" font-family="monospace" font-size="11" letter-spacing="2" fill="#6B6550">PRECIO DE ENTRADA</text>
  <text x="28" y="222" font-family="Georgia,serif" font-size="28" font-weight="bold" fill="#F0D080">${price:.2f}</text>
  <line x1="28" y1="238" x2="572" y2="238" stroke="{color_main}" stroke-width="0.5" stroke-opacity="0.3"/>
  <text x="28" y="262" font-family="monospace" font-size="9" letter-spacing="3" fill="#6B6550">TAKE PROFIT</text>
  <text x="28" y="282" font-family="Georgia,serif" font-size="22" font-weight="bold" fill="{color_main}">${tp:.2f}</text>
  <text x="220" y="262" font-family="monospace" font-size="9" letter-spacing="3" fill="#6B6550">STOP LOSS</text>
  <text x="220" y="282" font-family="Georgia,serif" font-size="22" font-weight="bold" fill="#FF4466">${sl:.2f}</text>
  <text x="410" y="262" font-family="monospace" font-size="9" letter-spacing="3" fill="#6B6550">RATIO R/B</text>
  <text x="410" y="282" font-family="Georgia,serif" font-size="22" font-weight="bold" fill="{color_gold}">{rr}:1</text>
  <rect x="0" y="300" width="600" height="40" fill="#111109"/>
  <text x="28" y="332" font-family="Georgia,serif" font-size="13" fill="{color_main}">{confidence}%</text>
  <text x="140" y="332" font-family="Georgia,serif" font-size="13" fill="{color_gold}">${atr:.2f}</text>
  <text x="230" y="332" font-family="Georgia,serif" font-size="13" fill="{color_gold}">{session}</text>
  <text x="380" y="332" font-family="Georgia,serif" font-size="13" fill="#6B6550">aurum-signals.onrender.com</text>
</svg>'''
    return svg.encode('utf-8')

def send_telegram_photo(token, chat_id, svg_bytes, caption):
    try:
        boundary = "AurumBoundary"
        # [FIX] Usar -- (doble guion ASCII) en vez de – (en-dash) para boundary multipart
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{chat_id}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"document\"; filename=\"aurum_signal.svg\"\r\n"
            f"Content-Type: image/svg+xml\r\n\r\n"
        ).encode() + svg_bytes + f"\r\n--{boundary}--\r\n".encode()
        url = f"https://api.telegram.org/bot{token}/sendDocument"
        req = urllib.request.Request(url, data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            result = json.loads(r.read().decode())
            if result.get("ok"): print("  📸 Imagen enviada a Telegram")
            else: print(f"  ⚠ Telegram foto: {result.get('description')}")
    except Exception as e:
        print(f"  ⚠ Error enviando imagen: {e}")

def send_telegram_direct(token, chat, msg, image_data=None):
    try:
        if image_data:
            send_telegram_photo(token, chat, image_data, msg)
        else:
            url  = f"https://api.telegram.org/bot{token}/sendMessage"
            data = urllib.parse.urlencode({"chat_id": chat, "text": msg, "parse_mode": "Markdown"}).encode()
            urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=8)
            print(f"  📲 Telegram enviado")
    except Exception as e:
        print(f"  ⚠ Telegram: {e}")


HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AURUM · XAU/USD</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=IBM+Plex+Mono:wght@300;400;500&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&display=swap" rel="stylesheet">
<style>
:root{--bg:#080600;--bg2:#0C0A02;--bg3:#111005;--gold:#C9A84C;--gold2:#F0D080;--gold3:#6B5520;--green:#00CC88;--red:#CC3344;--text:#E8E0C0;--dim:#3D3010;--dim2:#5A4820;--border:rgba(201,168,76,0.1);--border2:rgba(201,168,76,0.2)}
*{margin:0;padding:0;box-sizing:border-box}
body{background:radial-gradient(ellipse at 50% 0%,#0F0C02 0%,#080600 40%,#000 100%);color:var(--text);font-family:'IBM Plex Mono',monospace;min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;background:radial-gradient(ellipse at 20% 50%,rgba(201,168,76,.03) 0%,transparent 60%),radial-gradient(ellipse at 80% 20%,rgba(201,168,76,.02) 0%,transparent 50%);pointer-events:none;z-index:0}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
@keyframes goldShimmer{0%{background-position:-200% center}100%{background-position:200% center}}
@keyframes subtleGlow{0%,100%{box-shadow:0 0 20px rgba(201,168,76,.08)}50%{box-shadow:0 0 40px rgba(201,168,76,.18)}}
header{display:flex;align-items:center;justify-content:space-between;padding:18px 32px;border-bottom:1px solid var(--border);background:linear-gradient(180deg,rgba(15,12,2,.99) 0%,rgba(8,6,0,.97) 100%);position:relative;z-index:10;box-shadow:0 4px 20px rgba(0,0,0,.8)}
header::after{content:'';position:absolute;bottom:0;left:32px;right:32px;height:1px;background:linear-gradient(90deg,transparent,var(--gold),transparent);opacity:.3}
.logo{font-family:'Playfair Display',serif;font-size:20px;font-weight:700;letter-spacing:8px;background:linear-gradient(90deg,#8B6914,#F0D080,#C9A84C,#F0D080,#8B6914);background-size:200% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:goldShimmer 4s linear infinite;filter:drop-shadow(0 0 8px rgba(201,168,76,.3))}
.logo span{font-family:'IBM Plex Mono',monospace;font-size:8px;letter-spacing:4px;color:var(--dim2);display:block;margin-top:-1px;-webkit-text-fill-color:var(--dim2)}
.header-right{display:flex;align-items:center;gap:14px}
.live-badge{display:flex;align-items:center;gap:7px;font-size:9px;letter-spacing:3px;color:var(--dim2)}
.live-dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
.sound-toggle{background:transparent;border:1px solid var(--border);color:var(--dim2);font-family:'IBM Plex Mono';font-size:8px;letter-spacing:2px;padding:5px 10px;cursor:pointer;transition:all .3s}
.sound-toggle.on{border-color:rgba(0,204,136,.3);color:var(--green)}
.tg-status{font-size:8px;letter-spacing:2px;padding:5px 10px;border:1px solid var(--border);color:var(--dim2)}
.tg-status.connected{border-color:rgba(0,204,136,.3);color:var(--green)}
nav{display:flex;border-bottom:1px solid var(--border);padding:0 32px;background:var(--bg);position:relative;z-index:10}
.nav-tab{padding:14px 20px;font-family:'IBM Plex Mono';font-size:8px;letter-spacing:4px;color:var(--dim2);cursor:pointer;border-bottom:1px solid transparent;transition:all .3s;background:transparent;border-top:none;border-left:none;border-right:none}
.nav-tab:hover{color:var(--gold)}.nav-tab.active{color:var(--gold);border-bottom-color:var(--gold)}
.main{display:grid;grid-template-columns:1fr 300px;min-height:calc(100vh - 112px);position:relative;z-index:1}
.left{padding:28px 32px;border-right:1px solid var(--border)}
.right{padding:24px 22px;display:flex;flex-direction:column;gap:18px;overflow-y:auto;background:linear-gradient(180deg,#0C0A02 0%,#080600 100%)}
.page{display:none}.page.active{display:block}
.sessions-bar{display:flex;gap:1px;margin-bottom:16px;animation:fadeUp .5s ease both}
.session-block{flex:1;padding:10px 6px;text-align:center;border:1px solid var(--border);background:var(--bg2);transition:all .4s}
.session-block.active-session{border-color:var(--border2);background:rgba(201,168,76,.05);animation:subtleGlow 3s ease infinite}
.session-block.overlap{border-color:rgba(201,168,76,.4);background:rgba(201,168,76,.08)}
.session-block.closed{opacity:.3}
.session-name{font-family:'Playfair Display',serif;font-size:12px;letter-spacing:2px;color:var(--dim2);transition:all .4s}
.session-block.active-session .session-name,.session-block.overlap .session-name{color:var(--gold)}
.session-hours{font-size:7px;letter-spacing:1px;color:var(--dim);margin-top:3px}
.session-status{font-size:8px;letter-spacing:2px;margin-top:4px;color:var(--dim)}
.session-block.active-session .session-status,.session-block.overlap .session-status{color:var(--gold)}
.session-tip{font-family:'Cormorant Garamond',serif;font-size:12px;font-style:italic;color:var(--dim2);padding:8px 14px;border-left:2px solid var(--gold3);background:rgba(201,168,76,.03);margin-bottom:16px;letter-spacing:.5px}
.session-tip.hot{border-left-color:var(--gold);color:var(--gold)}.session-tip.warm{border-left-color:var(--gold2);color:var(--gold2)}
.price-section{margin-bottom:24px;animation:fadeUp .5s ease both}
.price-label{font-size:8px;letter-spacing:4px;color:var(--dim2);margin-bottom:6px}
.price-main{font-family:'Playfair Display',serif;font-size:48px;font-weight:700;line-height:1;letter-spacing:1px;background:linear-gradient(135deg,#C9A84C 0%,#F0D080 40%,#FFF 60%,#F0D080 80%,#C9A84C 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;filter:drop-shadow(0 2px 8px rgba(201,168,76,.2))}
.price-change{font-size:11px;margin-top:6px;letter-spacing:1px}
.price-change.up{color:var(--green)}.price-change.down{color:var(--red)}
.source-badge{font-size:8px;letter-spacing:2px;padding:3px 8px;border:1px solid;display:inline-block;margin-top:6px}
.source-badge.live{color:var(--green);border-color:rgba(0,204,136,.3)}.source-badge.sim{color:var(--dim2);border-color:var(--border)}
.alert-box{padding:22px 26px;margin-bottom:20px;border:1px solid var(--border);background:var(--bg2);position:relative;overflow:hidden;animation:fadeUp .5s .1s ease both;transition:all .5s;backdrop-filter:blur(10px);box-shadow:inset 0 1px 0 rgba(255,255,255,.03),0 4px 24px rgba(0,0,0,.5)}
.alert-box.go-buy{border-color:rgba(0,204,136,.25);background:rgba(0,204,136,.03);box-shadow:inset 0 1px 0 rgba(0,204,136,.1),0 0 40px rgba(0,204,136,.05)}
.alert-box.go-sell{border-color:rgba(204,51,68,.25);background:rgba(204,51,68,.03);box-shadow:inset 0 1px 0 rgba(204,51,68,.1),0 0 40px rgba(204,51,68,.05)}
.alert-box::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;transition:background .5s}
.alert-box.go-buy::before{background:linear-gradient(90deg,transparent,var(--green),transparent)}
.alert-box.go-sell::before{background:linear-gradient(90deg,transparent,var(--red),transparent)}
.alert-box.wait::before{background:linear-gradient(90deg,transparent,var(--gold3),transparent)}
.alert-tag{font-size:8px;letter-spacing:4px;color:var(--dim2);margin-bottom:12px}
.alert-signal{font-family:'Playfair Display',serif;font-size:36px;font-weight:700;letter-spacing:2px;line-height:1;transition:color .5s}
.alert-box.go-buy .alert-signal{background:linear-gradient(135deg,#00AA66,#00FF99);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;filter:drop-shadow(0 2px 6px rgba(0,204,136,.3))}
.alert-box.go-sell .alert-signal{background:linear-gradient(135deg,#CC3344,#FF6677);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;filter:drop-shadow(0 2px 6px rgba(204,51,68,.3))}
.alert-box.wait .alert-signal{color:var(--dim2)}
.alert-reason{font-family:'Cormorant Garamond',serif;font-size:13px;font-style:italic;color:var(--dim2);margin-top:8px}
.validity-row{display:flex;gap:6px;margin-top:14px;flex-wrap:wrap}
.validity-pill{font-size:7px;letter-spacing:1px;padding:3px 8px;border:1px solid var(--border);color:var(--dim2)}
.validity-pill.ok{border-color:rgba(0,204,136,.3);color:var(--green)}.validity-pill.fail{border-color:rgba(204,51,68,.3);color:var(--red)}
.conf-row{display:flex;align-items:center;gap:10px;margin-top:14px;font-size:8px;letter-spacing:2px;color:var(--dim2)}
.conf-bar{flex:1;height:1px;background:var(--bg3);position:relative}
.conf-fill{position:absolute;left:0;top:0;height:100%;transition:width .8s,background .5s}
.alert-box.go-buy .conf-fill{background:var(--green)}.alert-box.go-sell .conf-fill{background:var(--red)}.alert-box.wait .conf-fill{background:var(--gold3)}
.levels-box{background:var(--bg2);border:1px solid var(--border);padding:18px 22px;margin-bottom:20px;box-shadow:inset 0 1px 0 rgba(255,255,255,.02),0 2px 12px rgba(0,0,0,.4);transition:box-shadow .4s}
.levels-box:hover{box-shadow:inset 0 1px 0 rgba(201,168,76,.05),0 4px 20px rgba(0,0,0,.6),0 0 30px rgba(201,168,76,.04)}
.levels-title{font-size:8px;letter-spacing:4px;color:var(--dim2);margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
.level-row{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(201,168,76,.04)}
.level-row:last-child{border:none}
.level-name{font-size:8px;letter-spacing:3px;color:var(--dim2)}
.level-val{font-family:'Playfair Display',serif;font-size:20px;font-weight:700}
.level-val.entry{background:linear-gradient(135deg,#C9A84C,#F0D080);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.level-val.tp{color:var(--green);filter:drop-shadow(0 0 4px rgba(0,204,136,.4))}.level-val.sl{color:var(--red);filter:drop-shadow(0 0 4px rgba(204,51,68,.4))}
.level-sub{font-size:8px;color:var(--dim);text-align:right;margin-top:2px}
.log-btn{width:100%;padding:8px;background:transparent;border:1px solid rgba(0,204,136,.2);color:var(--green);font-family:'IBM Plex Mono';font-size:8px;letter-spacing:3px;cursor:pointer;margin-top:4px;transition:all .3s}
.chart-tab{padding:4px 10px;font-size:8px;letter-spacing:2px;border:1px solid var(--border);background:transparent;color:var(--dim2);cursor:pointer;transition:all .2s;font-family:'IBM Plex Mono'}
.chart-tab.active,.chart-tab:hover{border-color:var(--border2);color:var(--gold)}
canvas{width:100%;display:block;filter:drop-shadow(0 0 1px rgba(201,168,76,.1))}
.indicators-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:20px;animation:fadeUp .5s .35s ease both}
.indicator-box{background:var(--bg2);border:1px solid var(--border);padding:16px;box-shadow:inset 0 1px 0 rgba(255,255,255,.02),0 2px 12px rgba(0,0,0,.4)}
.ind-label{font-size:8px;letter-spacing:3px;color:var(--dim2);margin-bottom:6px}
.ind-value{font-family:'Playfair Display',serif;font-size:28px;font-weight:700;line-height:1}
.ind-status{font-size:8px;letter-spacing:2px;margin-top:5px}
.ind-status.bullish{color:var(--green)}.ind-status.bearish{color:var(--red)}.ind-status.neutral{color:var(--gold)}
.rsi-gauge{position:relative;width:100%;height:2px;background:linear-gradient(to right,var(--red) 30%,var(--gold3) 30%,var(--gold3) 70%,var(--green) 70%);margin-top:10px}
.rsi-needle{position:absolute;top:-5px;width:1px;height:12px;background:var(--gold2);transform:translateX(-50%);transition:left .8s}
.rsi-labels{display:flex;justify-content:space-between;margin-top:4px;font-size:7px;color:var(--dim)}
.ai-panel{background:var(--bg2);border:1px solid var(--border);padding:18px 22px;margin-bottom:20px;position:relative;box-shadow:inset 0 1px 0 rgba(255,255,255,.02),0 2px 12px rgba(0,0,0,.4);transition:box-shadow .4s}
.ai-panel::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--gold3),transparent)}
.ai-panel:hover{box-shadow:inset 0 1px 0 rgba(201,168,76,.05),0 4px 20px rgba(0,0,0,.6),0 0 30px rgba(201,168,76,.04)}
.ai-title{font-size:8px;letter-spacing:4px;color:var(--dim2);margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
.ai-badge{font-size:7px;letter-spacing:2px;padding:3px 8px;border:1px solid var(--border2);color:var(--gold)}
.ai-prob-wrap{display:flex;align-items:center;gap:14px;margin-bottom:14px}
.ai-prob-circle{width:68px;height:68px;border-radius:50%;border:1px solid var(--border2);display:flex;flex-direction:column;align-items:center;justify-content:center;flex-shrink:0;transition:all .5s;box-shadow:inset 0 0 20px rgba(0,0,0,.5)}
.ai-prob-circle.buy{border-color:rgba(0,204,136,.3);box-shadow:inset 0 0 20px rgba(0,0,0,.5),0 0 20px rgba(0,204,136,.1)}
.ai-prob-circle.sell{border-color:rgba(204,51,68,.3);box-shadow:inset 0 0 20px rgba(0,0,0,.5),0 0 20px rgba(204,51,68,.1)}
.ai-prob-val{font-family:'Playfair Display',serif;font-size:22px;font-weight:700;line-height:1}
.ai-prob-val.buy{background:linear-gradient(135deg,#00AA66,#00FF99);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.ai-prob-val.sell{background:linear-gradient(135deg,#CC3344,#FF6677);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.ai-prob-val.neutral{color:var(--gold)}
.ai-prob-label{font-size:7px;letter-spacing:2px;color:var(--dim2);margin-top:2px}
.ai-signal-wrap{flex:1}
.ai-signal{font-family:'Playfair Display',serif;font-size:24px;font-weight:700;transition:all .5s}
.ai-signal.buy{color:var(--green)}.ai-signal.sell{color:var(--red)}.ai-signal.neutral{color:var(--gold)}
.ai-desc{font-family:'Cormorant Garamond',serif;font-size:12px;font-style:italic;color:var(--dim2);margin-top:4px}
.ai-bar-wrap{margin-top:8px}
.ai-bar-label{display:flex;justify-content:space-between;font-size:7px;color:var(--dim);margin-bottom:4px}
.ai-bar{height:1px;background:var(--bg3);position:relative}
.ai-bar-fill{position:absolute;left:0;top:0;height:100%;transition:width .8s,background .5s}
.ai-stats-row{display:flex;gap:8px;margin-top:12px}
.ai-stat{flex:1;text-align:center;padding:8px;background:var(--bg3);border:1px solid var(--border)}
.ai-stat-val{font-family:'Playfair Display',serif;font-size:18px;font-weight:700;color:var(--gold)}
.ai-stat-label{font-size:7px;letter-spacing:2px;color:var(--dim2);margin-top:3px}
.ai-consensus{margin-top:12px;padding:10px 14px;border:1px solid;font-family:'Cormorant Garamond',serif;font-size:13px;font-style:italic}
.ai-consensus.agree{border-color:rgba(0,204,136,.2);color:var(--green);background:rgba(0,204,136,.03)}
.ai-consensus.disagree{border-color:rgba(204,51,68,.2);color:var(--red);background:rgba(204,51,68,.03)}
.ai-consensus.neutral{border-color:var(--border);color:var(--dim2)}
.ai-training{font-size:8px;color:var(--dim);letter-spacing:2px;margin-top:10px;text-align:center}
.panel-title{font-size:8px;letter-spacing:4px;color:var(--dim2);margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--border)}
.rules-box{background:var(--bg3);border:1px solid var(--border);padding:14px;box-shadow:inset 0 1px 0 rgba(255,255,255,.02),0 2px 12px rgba(0,0,0,.4)}
.rule-row{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid rgba(201,168,76,.04);font-size:9px}
.rule-row:last-child{border:none}
.rule-label{color:var(--dim2);letter-spacing:1px;font-size:8px}
.rule-input{background:transparent;border:none;border-bottom:1px solid var(--gold3);color:var(--gold);font-family:'IBM Plex Mono';font-size:12px;width:65px;text-align:right;outline:none;padding:2px 4px}
.rule-input:focus{border-bottom-color:var(--gold)}.rule-unit{font-size:8px;color:var(--dim2);margin-left:3px}
.ma-row{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid rgba(201,168,76,.04)}
.ma-name{font-size:8px;letter-spacing:1px;color:var(--dim2)}.ma-val{font-size:10px;color:var(--text)}
.ma-sig{font-size:8px;letter-spacing:2px}.ma-sig.b{color:var(--green)}.ma-sig.s{color:var(--red)}
.history-list{display:flex;flex-direction:column;gap:6px}
.hist-item{background:var(--bg3);border:1px solid var(--border);padding:9px 12px;display:flex;align-items:center;gap:10px}
.hist-dot{width:5px;height:5px;border-radius:50%;flex-shrink:0}
.hist-dot.b{background:var(--green)}.hist-dot.s{background:var(--red)}.hist-dot.h{background:var(--dim)}
.hist-info{flex:1}.hist-sig{font-size:9px;letter-spacing:2px;font-family:'IBM Plex Mono'}
.hist-sig.b{color:var(--green)}.hist-sig.s{color:var(--red)}.hist-sig.h{color:var(--dim2)}
.hist-sub{font-size:8px;color:var(--dim);margin-top:2px}.hist-time{font-size:8px;color:var(--dim2)}
.update-btn{width:100%;padding:12px;background:transparent;border:1px solid var(--border2);color:var(--gold);font-family:'IBM Plex Mono';font-size:9px;letter-spacing:4px;cursor:pointer;transition:all .3s;position:relative;overflow:hidden;box-shadow:inset 0 1px 0 rgba(201,168,76,.1)}
.update-btn::before{content:'';position:absolute;inset:0;background:var(--gold);transform:scaleX(0);transform-origin:left;transition:transform .3s}
.update-btn:hover::before{transform:scaleX(1)}.update-btn:hover{color:var(--bg)}.update-btn span{position:relative;z-index:1}
.update-btn:disabled{opacity:.3;pointer-events:none}
.timestamp{font-size:8px;color:var(--dim2);letter-spacing:2px;text-align:center}
.disclaimer{font-size:8px;color:var(--dim);line-height:1.7;padding:12px;border:1px solid var(--border)}
.disclaimer strong{color:var(--gold3)}
.bt-page,.reg-page{padding:28px 32px}
.bt-title,.reg-title{font-family:'Playfair Display',serif;font-size:32px;font-weight:700;color:var(--gold);letter-spacing:2px}
.bt-subtitle{font-size:8px;letter-spacing:3px;color:var(--dim2);margin-top:4px}
.bt-controls{display:flex;gap:10px;align-items:flex-end;margin:20px 0;flex-wrap:wrap}
.bt-input{background:var(--bg2);border:1px solid var(--border);color:var(--gold);font-family:'IBM Plex Mono';font-size:11px;padding:7px 10px;outline:none;width:100px}
.bt-input:focus{border-color:var(--border2)}.bt-label{font-size:8px;letter-spacing:3px;color:var(--dim2);margin-bottom:4px}
.bt-run{padding:9px 20px;background:transparent;border:1px solid var(--border2);color:var(--gold);font-family:'IBM Plex Mono';font-size:9px;letter-spacing:3px;cursor:pointer;transition:all .3s}
.bt-run:hover{background:var(--gold);color:var(--bg)}
.bt-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px}
.bt-stat{background:var(--bg2);border:1px solid var(--border);padding:14px;text-align:center}
.bt-stat-val{font-family:'Playfair Display',serif;font-size:28px;font-weight:700}
.bt-stat-val.green{color:var(--green)}.bt-stat-val.red{color:var(--red)}.bt-stat-val.gold{color:var(--gold)}
.bt-stat-label{font-size:7px;letter-spacing:3px;color:var(--dim2);margin-top:4px}
.bt-chart-wrap{background:var(--bg2);border:1px solid var(--border);padding:16px;margin-bottom:16px}
.bt-chart-title{font-size:8px;letter-spacing:3px;color:var(--dim2);margin-bottom:12px}
.bt-table{width:100%;border-collapse:collapse;font-size:9px}
.bt-table th{font-size:7px;letter-spacing:3px;color:var(--dim2);padding:8px 10px;text-align:left;border-bottom:1px solid var(--border)}
.bt-table td{padding:7px 10px;border-bottom:1px solid rgba(201,168,76,.03)}
.td-buy{color:var(--green)}.td-sell{color:var(--red)}.td-pos{color:var(--green)}.td-neg{color:var(--red)}
.reg-btn{padding:8px 16px;background:transparent;border:1px solid var(--border);color:var(--dim2);font-family:'IBM Plex Mono';font-size:8px;letter-spacing:2px;cursor:pointer;transition:all .3s}
.reg-btn:hover,.reg-btn.primary{border-color:var(--border2);color:var(--gold)}
.reg-summary{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:16px 0}
.reg-stat{background:var(--bg2);border:1px solid var(--border);padding:14px}
.reg-stat-val{font-family:'Playfair Display',serif;font-size:28px;font-weight:700}
.reg-stat-val.g{color:var(--green)}.reg-stat-val.r{color:var(--red)}.reg-stat-val.w{color:var(--gold)}
.reg-stat-label{font-size:7px;letter-spacing:3px;color:var(--dim2);margin-top:4px}
.reg-table-wrap{background:var(--bg2);border:1px solid var(--border);overflow:auto}
.reg-table{width:100%;border-collapse:collapse;font-size:9px}
.reg-table th{font-size:7px;letter-spacing:2px;color:var(--dim2);padding:8px 10px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}
.reg-table td{padding:7px 10px;border-bottom:1px solid rgba(201,168,76,.03);white-space:nowrap}
.reg-empty{padding:30px;text-align:center;color:var(--dim2);font-size:10px;letter-spacing:2px}
.tg-box{background:var(--bg3);border:1px solid var(--border);padding:16px}
.tg-field{display:flex;flex-direction:column;gap:4px;margin-bottom:12px}
.tg-field label{font-size:7px;letter-spacing:3px;color:var(--dim2)}
.tg-field input{background:var(--bg2);border:1px solid var(--border);color:var(--gold);font-family:'IBM Plex Mono';font-size:10px;padding:7px 10px;outline:none}
.tg-field input:focus{border-color:var(--border2)}
.tg-save{width:100%;padding:9px;background:transparent;border:1px solid var(--border2);color:var(--gold);font-family:'IBM Plex Mono';font-size:8px;letter-spacing:3px;cursor:pointer;margin-top:8px;transition:all .3s}
.tg-save:hover{background:rgba(201,168,76,.08)}
.tg-hint{font-size:8px;color:var(--dim);line-height:1.7;margin-top:10px;padding:10px;border:1px solid var(--border)}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--gold3);border-radius:2px}::-webkit-scrollbar-thumb:hover{background:var(--gold)}
::selection{background:rgba(201,168,76,.2);color:var(--gold2)}
@media(max-width:900px){.main{grid-template-columns:1fr}.right{border-top:1px solid var(--border)}.price-main{font-size:36px}.bt-stats{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<header>
  <div class="logo">AURUM<span>BOT DE SEÑALES XAU/USD v6.0</span></div>
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
<div class="page active" id="page-dashboard">
<div class="main">
<div class="left">
  <div class="sessions-bar">
    <div class="session-block" id="sessAsia"><div class="session-name">ASIA</div><div class="session-hours">22:00–09:00 GMT</div><div class="session-status" id="sessAsiaStatus">—</div></div>
    <div class="session-block" id="sessLondon"><div class="session-name">LONDRES</div><div class="session-hours">08:00–17:00 GMT</div><div class="session-status" id="sessLondonStatus">—</div></div>
    <div class="session-block" id="sessNY"><div class="session-name">NEW YORK</div><div class="session-hours">13:00–22:00 GMT</div><div class="session-status" id="sessNYStatus">—</div></div>
  </div>
  <div class="session-tip" id="sessionTip">Calculando sesión...</div>
  <div id="newsWarning" style="display:none;background:rgba(204,51,68,.04);border-left:2px solid var(--red);color:var(--red);padding:8px 14px;font-size:9px;letter-spacing:2px;margin-bottom:8px"></div>
  <div id="riskWarning" style="display:none;background:rgba(201,168,76,.04);border-left:2px solid var(--gold);color:var(--gold);padding:8px 14px;font-size:9px;letter-spacing:2px;margin-bottom:8px"></div>
  <div id="driftWarning" style="display:none;background:rgba(204,51,68,.04);border-left:2px solid var(--red);color:var(--red);padding:8px 14px;font-size:9px;letter-spacing:2px;margin-bottom:8px"></div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
    <div style="font-size:9px;letter-spacing:3px;color:var(--dim2)">TENDENCIA 1H:</div>
    <div id="htfTrend" style="font-size:11px;letter-spacing:2px;color:var(--gold)">Cargando...</div>
  </div>
  <div style="display:flex;gap:6px;margin-bottom:16px;align-items:center">
    <span style="font-size:9px;letter-spacing:3px;color:var(--dim2)">TF:</span>
    <button class="chart-tab active" id="tf5" onclick="setTF(this,'5min')">5M</button>
    <button class="chart-tab" id="tf15" onclick="setTF(this,'15min')">15M</button>
    <button class="chart-tab" id="tf1h" onclick="setTF(this,'1h')">1H</button>
    <button class="chart-tab" id="tf4h" onclick="setTF(this,'4h')">4H</button>
    <span style="font-size:9px;color:var(--dim2);margin-left:8px" id="tfInfo">Cargando velas reales...</span>
  </div>
  <div class="price-section">
    <div class="price-label">PRECIO ACTUAL · XAU/USD</div>
    <div class="price-main" id="priceDisplay">—</div>
    <div class="price-change" id="priceChange">Cargando...</div>
    <div id="sourceBadge"></div>
  </div>
  <div class="alert-box wait" id="alertBox">
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
  <div class="ai-panel">
    <div class="ai-title">SEÑAL DE INTELIGENCIA ARTIFICIAL<span class="ai-badge" id="aiBadge">ENTRENANDO...</span></div>
    <div class="ai-prob-wrap">
      <div class="ai-prob-circle neutral" id="aiCircle"><div class="ai-prob-val neutral" id="aiProbVal">—</div><div class="ai-prob-label">PROB.</div></div>
      <div class="ai-signal-wrap">
        <div class="ai-signal neutral" id="aiSignal">CARGANDO</div>
        <div class="ai-desc" id="aiDesc">Entrenando modelo con datos históricos...</div>
        <div class="ai-bar-wrap">
          <div class="ai-bar-label"><span>VENDER</span><span>NEUTRAL</span><span>COMPRAR</span></div>
          <div class="ai-bar"><div class="ai-bar-fill" id="aiBarFill" style="width:50%;background:var(--gold3)"></div></div>
        </div>
      </div>
    </div>
    <div class="ai-stats-row">
      <div class="ai-stat"><div class="ai-stat-val" id="aiAccuracy">—</div><div class="ai-stat-label">ACCURACY</div></div>
      <div class="ai-stat"><div class="ai-stat-val" id="aiSamples">—</div><div class="ai-stat-label">MUESTRAS</div></div>
      <div class="ai-stat"><div class="ai-stat-val" id="aiRetrains">0</div><div class="ai-stat-label">ENTRENAM.</div></div>
    </div>
    <div class="ai-consensus neutral" id="aiConsensus">Esperando señal técnica y de IA...</div>
    <div class="ai-training">El modelo se re-entrena cada 5 minutos con nuevos datos</div>
  </div>
  <div class="levels-box">
    <div class="levels-title">NIVELES · ATR DINÁMICO<span id="rrLabel" style="color:var(--gold);font-size:10px"></span></div>
    <div class="level-row"><span class="level-name">ENTRADA</span><div class="level-val entry" id="lvlEntry">—</div></div>
    <div class="level-row"><span class="level-name">TAKE PROFIT</span><div style="text-align:right"><div class="level-val tp" id="lvlTP">—</div><div class="level-sub" id="lvlTPsub"></div></div></div>
    <div class="level-row"><span class="level-name">STOP LOSS</span><div style="text-align:right"><div class="level-val sl" id="lvlSL">—</div><div class="level-sub" id="lvlSLsub"></div></div></div>
  </div>
  <button class="log-btn" id="logBtn" onclick="logCurrentSignal()" style="display:none">📊 REGISTRAR ESTA OPERACIÓN EN EXCEL</button>
  <div style="margin-bottom:20px">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
      <div style="font-size:8px;letter-spacing:3px;color:var(--dim2)">GRÁFICA EN VIVO · XAU/USD</div>
      <div style="display:flex;gap:4px" id="tvTabs">
        <button class="chart-tab active" onclick="setTV(this,'1')">1M</button>
        <button class="chart-tab" onclick="setTV(this,'5')">5M</button>
        <button class="chart-tab" onclick="setTV(this,'15')">15M</button>
        <button class="chart-tab" onclick="setTV(this,'60')">1H</button>
        <button class="chart-tab" onclick="setTV(this,'240')">4H</button>
      </div>
    </div>
    <div id="tv_chart" style="height:400px;border:1px solid var(--border)"></div>
  </div>
  <div class="indicators-row">
    <div class="indicator-box">
      <div class="ind-label">RSI (14)</div><div class="ind-value" id="rsiValue">—</div>
      <div class="ind-status" id="rsiStatus">—</div>
      <div class="rsi-gauge"><div class="rsi-needle" id="rsiNeedle" style="left:50%"></div></div>
      <div class="rsi-labels"><span>VENTA</span><span>50</span><span>COMPRA</span></div>
    </div>
    <div class="indicator-box">
      <div class="ind-label">MACD</div><div class="ind-value" id="macdValue" style="font-size:24px">—</div>
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
<div class="page" id="page-backtest">
<div class="bt-page">
  <div class="bt-title">BACKTESTING</div><div class="bt-subtitle">SIMULACIÓN DE ESTRATEGIA CON DATOS HISTÓRICOS</div>
  <div class="bt-controls">
    <div><div class="bt-label">PERÍODOS</div><input class="bt-input" id="btPeriods" type="number" value="200" min="50" max="500"></div>
    <div><div class="bt-label">MULT. TP</div><input class="bt-input" id="btTP" type="number" value="1.5" step="0.1" style="width:80px"></div>
    <div><div class="bt-label">MULT. SL</div><input class="bt-input" id="btSL" type="number" value="1.0" step="0.1" style="width:80px"></div>
    <div><div class="bt-label">CONF. MIN %</div><input class="bt-input" id="btConf" type="number" value="70" style="width:80px"></div>
    <button class="bt-run" onclick="runBacktest()">▶ EJECUTAR</button>
  </div>
  <div class="bt-stats">
    <div class="bt-stat"><div class="bt-stat-val gold" id="btTotal">—</div><div class="bt-stat-label">SEÑALES TOTALES</div></div>
    <div class="bt-stat"><div class="bt-stat-val green" id="btWins">—</div><div class="bt-stat-label">GANADORAS</div></div>
    <div class="bt-stat"><div class="bt-stat-val red" id="btLosses">—</div><div class="bt-stat-label">PERDEDORAS</div></div>
    <div class="bt-stat"><div class="bt-stat-val" id="btWR" style="color:var(--gold)">—</div><div class="bt-stat-label">WIN RATE</div></div>
  </div>
  <div class="bt-chart-wrap"><div class="bt-chart-title">CURVA DE EQUITY (SIMULADA)</div><canvas id="btChart" height="160"></canvas></div>
  <div style="overflow:auto"><table class="bt-table">
    <thead><tr><th>#</th><th>SEÑAL</th><th>ENTRADA</th><th>TP</th><th>SL</th><th>RR</th><th>RESULTADO</th><th>P&L</th></tr></thead>
    <tbody id="btBody"></tbody>
  </table></div>
</div>
</div>
<div class="page" id="page-registro">
<div class="reg-page">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
    <div class="reg-title">REGISTRO</div>
    <div style="display:flex;gap:8px">
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
      <tbody id="regBody"><tr><td colspan="9" class="reg-empty">Sin operaciones registradas aún.</td></tr></tbody>
    </table>
  </div>
</div>
</div>
<div class="page" id="page-config">
<div class="bt-page">
  <div class="bt-title">CONFIGURACIÓN</div><div class="bt-subtitle">TELEGRAM · PREFERENCIAS</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:20px">
    <div>
      <div class="panel-title">ALERTAS TELEGRAM</div>
      <div class="tg-box">
        <div class="tg-field"><label>BOT TOKEN</label><input type="text" id="cfgToken" placeholder="123456:ABC-DEF..."></div>
        <div class="tg-field"><label>CHAT ID</label><input type="text" id="cfgChatId" placeholder="-100123456789"></div>
        <button class="tg-save" onclick="saveTelegram()">💾 GUARDAR Y ACTIVAR</button>
        <button class="tg-save" style="margin-top:6px;border-color:rgba(224,90,90,.3);color:var(--red)" onclick="testTelegram()">📲 ENVIAR MENSAJE DE PRUEBA</button>
        <div class="tg-hint">
          <strong style="color:var(--gold3)">Cómo obtener tu token:</strong><br>
          1. Abre Telegram y busca @BotFather<br>
          2. Escribe /newbot y sigue los pasos<br>
          3. Copia el token que te da<br><br>
          <strong style="color:var(--gold3)">Chat ID:</strong><br>
          Ve a: api.telegram.org/bot<b>TOKEN</b>/getUpdates
        </div>
      </div>
    </div>
    <div>
      <div class="panel-title">ESTADO DEL SISTEMA</div>
      <div class="rules-box">
        <div class="rule-row"><span class="rule-label">PRECIO API</span><span style="color:var(--green);font-size:10px">gold-api.com</span></div>
        <div class="rule-row"><span class="rule-label">ACTUALIZACIÓN</span><span style="color:var(--gold);font-size:10px">CADA 30s</span></div>
        <div class="rule-row"><span class="rule-label">CACHE</span><span style="color:var(--gold);font-size:10px">ACTIVO ✓</span></div>
        <div class="rule-row"><span class="rule-label">TELEGRAM</span><span id="cfgTGstatus" style="font-size:10px;color:var(--dim2)">INACTIVO</span></div>
        <div class="rule-row" style="border:none"><span class="rule-label">VERSIÓN</span><span style="color:var(--dim2);font-size:10px">AURUM v6.0</span></div>
      </div>
    </div>
  </div>
</div>
</div>
<script>
let prices=[],signalHistory=[],ohlcData=[],currentTF='5min',currentRange=20;
let soundOn=true,lastValidSignal='',audioCtx=null,currentSignalData=null,currentSessionName='';
let tgToken='',tgChatId='',aiRetrains=0;

function showPage(name){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('page-'+name).classList.add('active');
  event.target.classList.add('active');
  if(name==='registro')loadLog();
}
function getAC(){if(!audioCtx)audioCtx=new(window.AudioContext||window.webkitAudioContext)();return audioCtx}
function playTone(freqs,durs){if(!soundOn)return;try{const ctx=getAC();let t=ctx.currentTime;freqs.forEach((f,i)=>{const o=ctx.createOscillator(),g=ctx.createGain();o.connect(g);g.connect(ctx.destination);o.frequency.value=f;g.gain.setValueAtTime(0,t);g.gain.linearRampToValueAtTime(0.3,t+.02);g.gain.linearRampToValueAtTime(0,t+durs[i]);o.start(t);o.stop(t+durs[i]);t+=durs[i]+.05})}catch(e){}}
function alertBuy(){playTone([440,554,659],[.15,.15,.3])}
function alertSell(){playTone([659,554,440],[.15,.15,.3])}
function toggleSound(){soundOn=!soundOn;const b=document.getElementById('soundBtn');b.textContent=soundOn?'🔔 ON':'🔕 OFF';b.className='sound-toggle '+(soundOn?'on':'off');if(soundOn)try{getAC().resume()}catch(e){}}
function saveTelegram(){
  tgToken=document.getElementById('cfgToken').value.trim();tgChatId=document.getElementById('cfgChatId').value.trim();
  const active=tgToken&&tgChatId;
  document.getElementById('tgStatus').textContent=active?'📲 TELEGRAM ON':'📵 TELEGRAM OFF';
  document.getElementById('tgStatus').className='tg-status '+(active?'connected':'');
  document.getElementById('cfgTGstatus').textContent=active?'ACTIVO':'INACTIVO';
  document.getElementById('cfgTGstatus').style.color=active?'var(--green)':'var(--dim2)';
  alert(active?'✓ Telegram configurado':'Token o Chat ID vacíos');
}
async function testTelegram(){
  if(!tgToken||!tgChatId){alert('Configura el token y chat ID primero');return}
  await fetch('/telegram?token='+encodeURIComponent(tgToken)+'&chat='+encodeURIComponent(tgChatId)+'&msg='+encodeURIComponent('🥇 *AURUM v5.0 TEST*\nConexión exitosa.'));
  alert('Mensaje de prueba enviado');
}
function updateSessions(){
  const now=new Date(),h=now.getUTCHours()+(now.getUTCMinutes()/60);
  const asiaOpen=h<9||h>=22,londonOpen=h>=8&&h<17,nyOpen=h>=13&&h<22,overlap=h>=13&&h<17;
  const set=(id,sid,open,cls)=>{document.getElementById(id).className='session-block '+(open?(overlap&&cls!=='sessAsia'?'overlap':'active-session'):'closed');document.getElementById(sid).textContent=open?'ABIERTA':'CERRADA'};
  set('sessAsia','sessAsiaStatus',asiaOpen,'sessAsia');set('sessLondon','sessLondonStatus',londonOpen,'sessLondon');set('sessNY','sessNYStatus',nyOpen,'sessNY');
  const tip=document.getElementById('sessionTip');
  if(overlap){tip.textContent='⚡ OVERLAP Londres+NY — Máxima volatilidad. Mejores señales del día.';tip.className='session-tip hot';currentSessionName='OVERLAP';}
  else if(londonOpen){tip.textContent='📈 Sesión de Londres activa — Alta liquidez para el oro.';tip.className='session-tip warm';currentSessionName='LONDRES';}
  else if(nyOpen){tip.textContent='🗽 Nueva York activa — Volatilidad alta. Atentos a datos económicos.';tip.className='session-tip warm';currentSessionName='NEW YORK';}
  else if(asiaOpen){tip.textContent='🌙 Sesión Asia — Oro tranquilo. Mejor esperar Londres.';tip.className='session-tip';currentSessionName='ASIA';}
  else{tip.textContent='💤 Mercado cerrado.';tip.className='session-tip';currentSessionName='CERRADO';}
  const hh=String(now.getUTCHours()).padStart(2,'0'),mm=String(now.getUTCMinutes()).padStart(2,'0');
  document.title='AURUM · '+hh+':'+mm+' GMT';
}
function calcSMA(a,n){if(a.length<n)return null;return a.slice(-n).reduce((s,v)=>s+v,0)/n}
function calcEMA(a,n){if(a.length<n)return null;const k=2/(n+1);let e=a.slice(0,n).reduce((s,v)=>s+v,0)/n;for(let i=n;i<a.length;i++)e=a[i]*k+e*(1-k);return e}
function calcRSI(a,n=14){if(a.length<n+1)return 50;let g=0,l=0;for(let i=a.length-n;i<a.length;i++){const d=a[i]-a[i-1];d>0?g+=d:l-=d}const ag=g/n,al=l/n;if(al===0)return 100;return 100-100/(1+ag/al)}
function calcMACD(a){const e12=calcEMA(a,12),e26=calcEMA(a,26);if(!e12||!e26)return{macd:0,signal:0,hist:0};const m=e12-e26,ms=[];for(let i=26;i<=a.length;i++){const x=calcEMA(a.slice(0,i),12),y=calcEMA(a.slice(0,i),26);if(x&&y)ms.push(x-y)}const sig=calcEMA(ms,9)||0;return{macd:m,signal:sig,hist:m-sig}}
function calcATR(a,n=14){if(a.length<n+1)return null;const trs=[];for(let i=a.length-n;i<a.length;i++){const hi=a[i]*1.004,lo=a[i]*.996,pc=a[i-1];trs.push(Math.max(hi-lo,Math.abs(hi-pc),Math.abs(lo-pc)))}return trs.reduce((s,v)=>s+v,0)/n}
function computeSignal(){
  const RSI=calcRSI(prices),SMA20=calcSMA(prices,20),SMA50=calcSMA(prices,50);
  const EMA9=calcEMA(prices,9),EMA21=calcEMA(prices,21),MACD=calcMACD(prices),ATR=calcATR(prices,14);
  const price=prices[prices.length-1];
  let score=0;
  if(RSI<35)score+=2;else if(RSI>65)score-=2;else score+=RSI<50?.5:-.5;
  if(SMA20&&SMA50){if(SMA20>SMA50)score+=1;else score-=1}
  if(EMA9&&EMA21){if(EMA9>EMA21)score+=1;else score-=1}
  if(SMA20)score+=price>SMA20*1.002?.5:price<SMA20*.998?-.5:0;
  if(MACD.hist>0)score+=1;else score-=1;
  const rawSignal=score>=1.5?'COMPRAR':score<=-1.5?'VENDER':'ESPERAR';
  const confidence=Math.min(95,Math.round(40+Math.abs(score)*12));
  const minConf=parseInt(document.getElementById('ruleConf').value)||70;
  const rsiMaxBuy=parseInt(document.getElementById('ruleRSIbuy').value)||60;
  const rsiMinSell=parseInt(document.getElementById('ruleRSIsell').value)||40;
  const tpMult=parseFloat(document.getElementById('ruleTPmult').value)||1.5;
  const slMult=parseFloat(document.getElementById('ruleSLmult').value)||1.0;
  const atrVal=ATR||price*.005;
  const tpDist=atrVal*tpMult,slDist=atrVal*slMult,rr=(tpDist/slDist).toFixed(2);
  if(rawSignal==='ESPERAR')return{signal:'ESPERAR',cls:'wait',confidence,reason:'Sin tendencia clara',checks:[],valid:false,RSI,SMA20,SMA50,EMA9,EMA21,MACD,ATR:atrVal,price,tpDist,slDist,rr};
  const htfTrend=window._htfTrend||'neutral';
  const mtfOk=rawSignal==='COMPRAR'?(htfTrend==='up'||htfTrend==='neutral'):(htfTrend==='down'||htfTrend==='neutral');
  const newsBlocked=window._newsActive||false;
  const checks=[];let valid=true;
  const cok=confidence>=minConf;checks.push({label:'CONF '+confidence+'%≥'+minConf+'%',ok:cok});if(!cok)valid=false;
  checks.push({label:'1H '+htfTrend.toUpperCase(),ok:mtfOk});
  if(newsBlocked){checks.push({label:'NOTICIA ACTIVA',ok:false});valid=false;}
  const rok=rawSignal==='COMPRAR'?RSI<=rsiMaxBuy:RSI>=rsiMinSell;
  checks.push({label:'RSI '+RSI.toFixed(0),ok:rok});
  const ma=rawSignal==='COMPRAR'?(EMA9&&EMA21?EMA9>EMA21:false):(EMA9&&EMA21?EMA9<EMA21:false);
  checks.push({label:'MEDIAS OK',ok:ma});
  const mok=rawSignal==='COMPRAR'?MACD.hist>0:MACD.hist<0;
  checks.push({label:'MACD OK',ok:mok});
  const cls=valid?(rawSignal==='COMPRAR'?'go-buy':'go-sell'):'wait';
  const reason=valid?'✓ Señal confirmada — RR '+rr+':1':'Bloqueada: '+checks.filter(c=>!c.ok).map(c=>c.label).join(', ');
  return{signal:valid?rawSignal:'ESPERAR',rawSignal,cls,confidence,reason,checks,valid,RSI,SMA20,SMA50,EMA9,EMA21,MACD,ATR:atrVal,price,tpDist,slDist,rr};
}
function renderAlert(d){
  document.getElementById('alertBox').className='alert-box '+d.cls;
  document.getElementById('alertSignal').textContent=d.valid?d.signal:(d.rawSignal?d.rawSignal+' ✗':'ESPERAR');
  document.getElementById('alertReason').textContent=d.reason;
  document.getElementById('confPct').textContent=d.confidence+'%';
  document.getElementById('confFill').style.width=d.confidence+'%';
  document.getElementById('validityRow').innerHTML=d.checks.map(c=>'<div class="validity-pill '+(c.ok?'ok':'fail')+'">'+(c.ok?'✓':'✗')+' '+c.label+'</div>').join('');
  const logBtn=document.getElementById('logBtn');
  if(d.valid&&d.rawSignal!=='ESPERAR'){
    const isLong=d.rawSignal==='COMPRAR',tp=isLong?d.price+d.tpDist:d.price-d.tpDist,sl=isLong?d.price-d.slDist:d.price+d.slDist;
    document.getElementById('lvlEntry').textContent='$'+d.price.toFixed(2);
    document.getElementById('lvlTP').textContent='$'+tp.toFixed(2);
    document.getElementById('lvlSL').textContent='$'+sl.toFixed(2);
    document.getElementById('lvlTPsub').textContent=d.tpDist.toFixed(2)+' USD · '+parseFloat(document.getElementById('ruleTPmult').value)+'× ATR';
    document.getElementById('lvlSLsub').textContent=d.slDist.toFixed(2)+' USD · '+parseFloat(document.getElementById('ruleSLmult').value)+'× ATR';
    document.getElementById('rrLabel').textContent='RR '+d.rr+':1';logBtn.style.display='block';
  }else{
    ['lvlEntry','lvlTP','lvlSL'].forEach(id=>document.getElementById(id).textContent='—');
    ['lvlTPsub','lvlSLsub'].forEach(id=>document.getElementById(id).textContent='');
    document.getElementById('rrLabel').textContent='';logBtn.style.display='none';
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
    return'<div class="ma-row"><span class="ma-name">'+r.name+'</span><span class="ma-val">$'+r.val.toFixed(2)+'</span><span class="ma-sig '+s.cls+'">'+s.txt+'</span></div>';
  }).join('');
}
function addHistory(d){
  const now=new Date(),time=now.getHours().toString().padStart(2,'0')+':'+now.getMinutes().toString().padStart(2,'0');
  const dc=d.cls==='go-buy'?'b':d.cls==='go-sell'?'s':'h';
  signalHistory.unshift({signal:d.valid?d.signal:'BLOQUEADA',cls:dc,price:d.price.toFixed(2),time,rr:d.valid?d.rr:null});
  if(signalHistory.length>6)signalHistory.pop();
  document.getElementById('historyList').innerHTML=signalHistory.map(h=>'<div class="hist-item"><div class="hist-dot '+h.cls+'"></div><div class="hist-info"><div class="hist-sig '+h.cls+'">'+h.signal+'</div><div class="hist-sub">$'+h.price+(h.rr?' · RR '+h.rr+':1':'')+'</div></div><div class="hist-time">'+h.time+'</div></div>').join('');
}
function triggerSound(d){
  const key=d.valid?d.rawSignal:'X';
  if(key===lastValidSignal)return;lastValidSignal=key;
  if(d.valid){d.rawSignal==='COMPRAR'?alertBuy():alertSell();}
}
async function logCurrentSignal(){
  if(!currentSignalData||!currentSignalData.valid)return;
  const d=currentSignalData,isLong=d.rawSignal==='COMPRAR';
  const tp=(isLong?d.price+d.tpDist:d.price-d.tpDist).toFixed(2);
  const sl=(isLong?d.price-d.slDist:d.price+d.slDist).toFixed(2);
  try{
    await fetch('/log?signal='+encodeURIComponent(d.rawSignal)+'&price='+d.price+'&tp='+tp+'&sl='+sl+'&atr='+d.ATR.toFixed(2)+'&rr='+d.rr+'&conf='+d.confidence+'&session='+encodeURIComponent(currentSessionName));
    const btn=document.getElementById('logBtn');
    btn.textContent='✓ REGISTRADA';btn.style.borderColor='var(--green)';
    setTimeout(()=>{btn.textContent='📊 REGISTRAR ESTA OPERACIÓN EN EXCEL';btn.style.borderColor=''},3000);
  }catch(e){}
}
async function loadLog(){
  try{
    const rows=await(await fetch('/getlog')).json();
    const total=rows.length,buys=rows.filter(r=>r['Señal']==='COMPRAR').length,sells=rows.filter(r=>r['Señal']==='VENDER').length;
    document.getElementById('regTotal').textContent=total;document.getElementById('regBuys').textContent=buys;document.getElementById('regSells').textContent=sells;
    const body=document.getElementById('regBody');
    if(!rows.length){body.innerHTML='<tr><td colspan="9" class="reg-empty">Sin operaciones aún.</td></tr>';return}
    body.innerHTML=[...rows].reverse().map(r=>'<tr><td>'+r['Fecha']+'</td><td>'+r['Hora GMT']+'</td><td class="'+(r['Señal']==='COMPRAR'?'td-buy':'td-sell')+'">'+r['Señal']+'</td><td>$'+r['Precio Entrada']+'</td><td style="color:var(--green)">$'+r['TP']+'</td><td style="color:var(--red)">$'+r['SL']+'</td><td>'+r['RR']+':1</td><td>'+r['Confianza%']+'%</td><td>'+r['Sesión']+'</td></tr>').join('');
  }catch(e){}
}
function downloadCSV(){window.location='/download'}
function runBacktest(){
  if(prices.length<50){document.getElementById('btBody').innerHTML='<tr><td colspan="8" style="text-align:center;padding:30px;color:var(--dim2);letter-spacing:2px">SIN DATOS SUFICIENTES</td></tr>';return}
  const tpM=parseFloat(document.getElementById('btTP').value)||1.5;
  const slM=parseFloat(document.getElementById('btSL').value)||1.0;
  const minC=parseInt(document.getElementById('btConf').value)||70;
  const trades=[];let equity=10000;const eqCurve=[10000];
  for(let i=50;i<prices.length;i++){
    const sl2=prices.slice(0,i+1),RSI=calcRSI(sl2),EMA9=calcEMA(sl2,9),EMA21=calcEMA(sl2,21);
    const MACD=calcMACD(sl2),ATR=calcATR(sl2,14)||sl2[sl2.length-1]*.005,price=sl2[sl2.length-1];
    let score=0;
    if(RSI<35)score+=2;else if(RSI>65)score-=2;else score+=RSI<50?.5:-.5;
    if(EMA9&&EMA21)score+=EMA9>EMA21?1:-1;score+=MACD.hist>0?1:-1;
    const conf=Math.min(95,Math.round(40+Math.abs(score)*12));if(conf<minC)continue;
    const sig=score>=1.5?'COMPRAR':score<=-1.5?'VENDER':null;if(!sig)continue;
    const tp=ATR*tpM,slv=ATR*slM,rr=(tp/slv).toFixed(2);
    const futIdx=Math.min(i+5,prices.length-1);
    const won=sig==='COMPRAR'?prices[futIdx]>price+(tp*.5):prices[futIdx]<price-(tp*.5);
    const pnl=won?tp*10:-slv*10;equity+=pnl;eqCurve.push(equity);
    trades.push({sig,price:price.toFixed(2),tp:(sig==='COMPRAR'?price+tp:price-tp).toFixed(2),sl:(sig==='COMPRAR'?price-slv:price+slv).toFixed(2),rr,won,pnl:pnl.toFixed(0)});
    if(trades.length>=50)break;
  }
  const wins=trades.filter(t=>t.won).length,wr=trades.length?Math.round(wins/trades.length*100):0;
  document.getElementById('btTotal').textContent=trades.length;document.getElementById('btWins').textContent=wins;
  document.getElementById('btLosses').textContent=trades.length-wins;document.getElementById('btWR').textContent=wr+'%';
  document.getElementById('btWR').style.color=wr>=50?'var(--green)':'var(--red)';
  const cv=document.getElementById('btChart'),ctx=cv.getContext('2d');
  const W=Math.max(cv.offsetWidth,600)||800,H=160;cv.width=W;cv.height=H;
  const mn=Math.min(...eqCurve)-100,mx=Math.max(...eqCurve)+100;
  const px=i=>(i/(eqCurve.length-1||1))*(W-20)+10,py=v=>H-((v-mn)/(mx-mn+.01))*(H-20)-10;
  ctx.clearRect(0,0,W,H);
  const lastEq=eqCurve[eqCurve.length-1],c2=lastEq>=10000?'rgba(76,175,130':'rgba(224,90,90';
  const gr=ctx.createLinearGradient(0,0,0,H);gr.addColorStop(0,c2+',.2)');gr.addColorStop(1,c2+',.0)');
  ctx.beginPath();ctx.moveTo(px(0),py(eqCurve[0]));eqCurve.forEach((v,i)=>ctx.lineTo(px(i),py(v)));ctx.lineTo(px(eqCurve.length-1),H);ctx.lineTo(px(0),H);ctx.fillStyle=gr;ctx.fill();
  ctx.beginPath();ctx.strokeStyle=lastEq>=10000?'#4CAF82':'#E05A5A';ctx.lineWidth=2;eqCurve.forEach((v,i)=>i===0?ctx.moveTo(px(i),py(v)):ctx.lineTo(px(i),py(v)));ctx.stroke();
  document.getElementById('btBody').innerHTML=trades.slice(-20).map((t,i)=>'<tr><td style="color:var(--dim2)">'+(i+1)+'</td><td class="'+(t.sig==='COMPRAR'?'td-buy':'td-sell')+'">'+t.sig+'</td><td>$'+t.price+'</td><td class="td-buy">$'+t.tp+'</td><td class="td-sell">$'+t.sl+'</td><td>'+t.rr+':1</td><td>'+(t.won?'✓ WIN':'✗ LOSS')+'</td><td class="'+(parseFloat(t.pnl)>=0?'td-pos':'td-neg')+'">'+(parseFloat(t.pnl)>=0?'+':'')+'$'+t.pnl+'</td></tr>').join('');
}
function drawPriceChart(){
  const cv=document.getElementById('priceChart');if(!cv)return;
  const ctx=cv.getContext('2d');const W=cv.offsetWidth||600,H=160;cv.width=W;cv.height=H;
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
function setTF(btn,tf){
  document.querySelectorAll('#tf5,#tf15,#tf1h,#tf4h').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');currentTF=tf;document.getElementById('tfInfo').textContent='Cargando velas '+tf+'...';fetchAndAnalyze();
}
async function updateAI(technicalSignal){
  if(prices.length<60)return;
  const pp=prices.slice(-200).join(',');
  try{
    const tr=await fetch('/aitrain?prices='+encodeURIComponent(pp));
    const td=await tr.json();
    if(td.trained){
      aiRetrains++;
      document.getElementById('aiAccuracy').textContent=td.accuracy+'%';
      document.getElementById('aiSamples').textContent=td.epochs;
      document.getElementById('aiRetrains').textContent=aiRetrains;
      document.getElementById('aiBadge').textContent='IA ACTIVA · '+td.accuracy+'% ACC';
      document.getElementById('aiBadge').style.borderColor='rgba(76,175,130,.4)';
      document.getElementById('aiBadge').style.color='var(--green)';
    }
    const pr=await fetch('/aipredict?prices='+encodeURIComponent(pp));
    const pd=await pr.json();
    if(pd.prob===null)return;
    const prob=pd.prob,aiSig=pd.signal,cls=aiSig==='COMPRAR'?'buy':aiSig==='VENDER'?'sell':'neutral';
    document.getElementById('aiCircle').className='ai-prob-circle '+cls;
    document.getElementById('aiProbVal').className='ai-prob-val '+cls;
    document.getElementById('aiProbVal').textContent=prob.toFixed(0)+'%';
    document.getElementById('aiSignal').className='ai-signal '+cls;
    document.getElementById('aiSignal').textContent=aiSig;
    document.getElementById('aiDesc').textContent=aiSig==='COMPRAR'?'El modelo ve '+prob.toFixed(0)+'% de probabilidad de subida':aiSig==='VENDER'?'El modelo ve '+(100-prob).toFixed(0)+'% de probabilidad de bajada':'Probabilidad insuficiente para señal clara';
    const fill=document.getElementById('aiBarFill');fill.style.width=prob+'%';fill.style.background=cls==='buy'?'var(--green)':cls==='sell'?'var(--red)':'var(--gold3)';
    const techValid=technicalSignal&&technicalSignal!=='ESPERAR';
    const agree=techValid&&technicalSignal===aiSig,disagree=techValid&&technicalSignal!==aiSig&&aiSig!=='ESPERAR';
    const consensus=document.getElementById('aiConsensus');
    if(agree){consensus.textContent='✓ CONSENSO — Técnico e IA coinciden en '+aiSig+'. Señal de mayor confianza.';consensus.className='ai-consensus agree'}
    else if(disagree){consensus.textContent='⚠ DIVERGENCIA — Técnico dice '+technicalSignal+' pero IA dice '+aiSig+'. Operar con precaución.';consensus.className='ai-consensus disagree'}
    else{consensus.textContent='Analizando convergencia entre indicadores técnicos e IA...';consensus.className='ai-consensus neutral'}
  }catch(e){document.getElementById('aiDesc').textContent='Error al conectar con el motor de IA'}
}
async function updateRiskStatus(){
  try{
    const [rr,ar]=await Promise.all([fetch('/riskstatus'),fetch('/aistats')]);
    const risk=await rr.json(),aiStats=await ar.json();
    const rEl=document.getElementById('riskWarning');
    if(rEl){rEl.style.display=risk.can_trade?'none':'block';if(!risk.can_trade)rEl.textContent='🛡 RIESGO: '+risk.reason+' — Señales pausadas';}
    const dEl=document.getElementById('driftWarning');
    if(dEl){dEl.style.display=aiStats.is_drifting?'block':'none';if(aiStats.is_drifting)dEl.textContent='⚡ DRIFT DETECTADO ('+Math.round(aiStats.drift_score*100)+'%) — IA pausada, mercado cambiado';}
    window._canTrade=risk.can_trade;window._isDrifting=aiStats.is_drifting;
  }catch(e){window._canTrade=true;window._isDrifting=false;}
}
function detectLateralMarket(prices,n=20){
  if(prices.length<n)return false;
  const r=prices.slice(-n);return(Math.max(...r)-Math.min(...r))/(Math.min(...r)+1e-9)*100<0.3;
}
async function fetchAndAnalyze(){
  const btn=document.getElementById('updateBtn');
  btn.disabled=true;btn.querySelector('span').textContent='OBTENIENDO PRECIO...';
  try{
    let priceData=null,candles=null;
    try{const r=await fetch('/precio');const pd=await r.json();if(pd&&pd.price)priceData=pd;}catch(e){}
    try{const r=await fetch('/ohlc?interval='+currentTF+'&size=200');const cd=await r.json();if(Array.isArray(cd)&&cd.length>0)candles=cd;}catch(e){}
    if(candles&&candles.length>0){
      ohlcData=candles;prices=candles.map(c=>c.close);
      document.getElementById('tfInfo').textContent=ohlcData.length+' velas reales';
      document.getElementById('sourceBadge').innerHTML='<span class="source-badge live">● DATOS REALES '+currentTF.toUpperCase()+'</span>';
    }
    if(priceData&&priceData.price){
      prices.push(priceData.price);if(prices.length>300)prices=prices.slice(-300);
      document.getElementById('priceDisplay').textContent='$'+priceData.price.toFixed(2);
      const ch=priceData.ch||0,chp=priceData.chp||0;
      const el=document.getElementById('priceChange');
      el.textContent=(ch>=0?'+':'')+ch.toFixed(2)+' ('+(ch>=0?'+':'')+Number(chp).toFixed(2)+'%)';
      el.className='price-change '+(ch>=0?'up':'down');
      if(!candles)document.getElementById('sourceBadge').innerHTML='<span class="source-badge live">● PRECIO REAL</span>';
    }else if(!prices.length){
      document.getElementById('priceDisplay').textContent='—';document.getElementById('priceChange').textContent='Sin datos — mercado cerrado o sin conexión';
      document.getElementById('sourceBadge').innerHTML='<span class="source-badge sim">◌ SIN DATOS</span>';
      btn.disabled=false;btn.querySelector('span').textContent='↻ ACTUALIZAR';return;
    }
    if(!prices.length){btn.disabled=false;btn.querySelector('span').textContent='↻ ACTUALIZAR';return;}
  }catch(e){if(!prices.length){btn.disabled=false;btn.querySelector('span').textContent='↻ ACTUALIZAR';return;}}
  btn.querySelector('span').textContent='ANALIZANDO...';
  try{
    const [nR,hR]=await Promise.all([fetch('/newsstatus'),fetch('/htftrend')]);
    const nd=await nR.json(),hd=await hR.json();
    window._newsActive=nd.active;
    const nEl=document.getElementById('newsWarning');
    if(nEl){nEl.style.display=nd.active?'block':'none';if(nd.active)nEl.textContent='⚠ NOTICIA ACTIVA: '+nd.event+' — Señales pausadas ±'+nd.pause_before+'min';}
    window._htfTrend=hd.trend;
    const htfEl=document.getElementById('htfTrend');
    if(htfEl){
      const color=hd.trend==='up'?'var(--green)':hd.trend==='down'?'var(--red)':'var(--gold)';
      const arrow=hd.trend==='up'?'↑ ALCISTA':hd.trend==='down'?'↓ BAJISTA':'→ NEUTRAL';
      htfEl.textContent='1H: '+arrow;htfEl.style.color=color;
    }
  }catch(e){window._htfTrend='neutral';}
  // v6: SCALP v2 como señal principal via /ictsignal
  let scalpSignal=null;
  try{const sr=await fetch('/ictsignal');scalpSignal=await sr.json();if(!scalpSignal||!scalpSignal.direction)scalpSignal=null;}catch(e){scalpSignal=null;}
  // Técnico clásico como fallback/contexto
  const d=computeSignal();
  // v6: si SCALP v2 tiene señal, la usamos como principal
  if(scalpSignal){
    const isBuy=scalpSignal.direction==='COMPRAR';
    const cls=isBuy?'go-buy':'go-sell';
    const det=scalpSignal.details||{};
    const checks=[
      {label:'SWEEP',ok:!!det.sweep},{label:det.bos||'BOS',ok:!!det.bos},
      {label:'FVG',ok:!!det.fvg},{label:'EMA',ok:!!det.ema},
      {label:det.ml||'ML',ok:det.ml&&det.ml.indexOf('HIGH')>=0}
    ];
    document.getElementById('alertBox').className='alert-box '+cls;
    document.getElementById('alertSignal').textContent=scalpSignal.direction;
    document.getElementById('alertReason').textContent='✓ SCALP v2 — Score '+scalpSignal.score+'% | RR '+scalpSignal.rr+':1';
    document.getElementById('confPct').textContent=scalpSignal.score+'%';
    document.getElementById('confFill').style.width=Math.min(scalpSignal.score,100)+'%';
    document.getElementById('validityRow').innerHTML=checks.map(c=>'<div class="validity-pill '+(c.ok?'ok':'fail')+'">'+(c.ok?'✓':'✗')+' '+c.label+'</div>').join('');
    document.getElementById('lvlEntry').textContent='$'+(prices[prices.length-1]||0).toFixed(2);
    document.getElementById('lvlTP').textContent='$'+scalpSignal.tp.toFixed(2);
    document.getElementById('lvlSL').textContent='$'+scalpSignal.sl.toFixed(2);
    document.getElementById('lvlTPsub').textContent=(scalpSignal.score>=85?'SNIPER 2.5':'NORMAL 1.5')+'× ATR';
    document.getElementById('lvlSLsub').textContent='1.0× ATR';
    document.getElementById('rrLabel').textContent='RR '+scalpSignal.rr+':1';
    document.getElementById('logBtn').style.display='block';
    currentSignalData={valid:true,rawSignal:scalpSignal.direction,signal:scalpSignal.direction,cls:cls,confidence:scalpSignal.score,price:prices[prices.length-1],tpDist:Math.abs(scalpSignal.tp-prices[prices.length-1]),slDist:Math.abs(scalpSignal.sl-prices[prices.length-1]),rr:scalpSignal.rr,ATR:d.ATR,checks:checks,reason:'SCALP v2'};
    triggerSound(currentSignalData);
    addHistory(currentSignalData);
    // Telegram via SCALP engine (ya se maneja en backend)
  }else{
    // Fallback: señal técnica clásica
    currentSignalData=d;triggerSound(d);
    renderAlert(d);addHistory(d);
    if(d.valid&&d.rawSignal!==lastValidSignal&&tgToken&&tgChatId){
      const isLong=d.rawSignal==='COMPRAR';
      const tp=(isLong?d.price+d.tpDist:d.price-d.tpDist).toFixed(2);
      const sl=(isLong?d.price-d.slDist:d.price+d.slDist).toFixed(2);
      fetch('/sendimage?signal='+encodeURIComponent(d.rawSignal)+'&price='+d.price+'&tp='+tp+'&sl='+sl+'&rr='+d.rr+'&conf='+d.confidence+'&session='+encodeURIComponent(currentSessionName)+'&atr='+d.ATR.toFixed(2));
    }
  }
  renderIndicators(d);
  updateAI(currentSignalData.valid?currentSignalData.signal:'ESPERAR');
  updateRiskStatus();
  if(detectLateralMarket(prices)){const tip=document.getElementById('sessionTip');if(tip){tip.textContent='📊 Mercado lateral detectado — movimiento mínimo. Mejor esperar tendencia.';tip.className='session-tip';}}
  drawPriceChart();drawMACDChart();
  document.getElementById('timestamp').textContent='✓ '+new Date().toLocaleTimeString('es-ES');
  if(ohlcData.length)document.getElementById('tfInfo').textContent=ohlcData.length+' velas reales cargadas';
  btn.disabled=false;btn.querySelector('span').textContent='↻ ACTUALIZAR';
}
fetchAndAnalyze();
updateSessions();
setInterval(fetchAndAnalyze,30000);
setInterval(updateSessions,60000);
window.addEventListener('resize',()=>{drawPriceChart();drawMACDChart()});
</script>
<script src="https://s3.tradingview.com/tv.js"></script>
<script>
var _tvWidget=null;
function initTVChart(interval){
  if(_tvWidget){try{_tvWidget.remove()}catch(e){}}
  if(typeof TradingView==='undefined')return;
  _tvWidget=new TradingView.widget({
    container_id:"tv_chart",symbol:"FOREXCOM:XAUUSD",interval:interval||'5',
    timezone:"Etc/UTC",theme:"dark",style:"1",locale:"es",
    enable_publishing:false,save_image:false,backgroundColor:"#080600",
    gridColor:"rgba(201,168,76,0.05)",
    studies:["RSI@tv-basicstudies","MACD@tv-basicstudies"],
    overrides:{
      "mainSeriesProperties.candleStyle.upColor":"#00CC88",
      "mainSeriesProperties.candleStyle.downColor":"#CC3344",
      "mainSeriesProperties.candleStyle.wickUpColor":"#00CC88",
      "mainSeriesProperties.candleStyle.wickDownColor":"#CC3344",
      "mainSeriesProperties.candleStyle.borderUpColor":"#00CC88",
      "mainSeriesProperties.candleStyle.borderDownColor":"#CC3344",
      "paneProperties.background":"#080600","paneProperties.backgroundType":"solid",
      "paneProperties.vertGridProperties.color":"rgba(201,168,76,0.03)",
      "paneProperties.horzGridProperties.color":"rgba(201,168,76,0.03)",
      "scalesProperties.textColor":"#5A4820","scalesProperties.backgroundColor":"#080600"
    },
    width:"100%",height:400,withdateranges:true,allow_symbol_change:false
  });
}
function setTV(btn,interval){
  document.querySelectorAll('#tvTabs .chart-tab').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');initTVChart(interval);
}
window.addEventListener('load',()=>setTimeout(()=>initTVChart('5'),800));
</script>
</body>
</html>"""


# ── BACKTEST ENGINE ───────────────────────────────────────

_backtest_result = {"data": None}

def _run_backtest_bg():
    global _backtest_result
    _backtest_result["data"] = None
    print("  Iniciando backtest real XAUUSD 5M...")
    try:
        url = (f"https://api.twelvedata.com/time_series"
               f"?symbol=XAU/USD&interval=5min&outputsize=500&apikey={TWELVE_API_KEY}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
        if "values" not in data:
            _backtest_result["data"] = {"error": f"API: {data.get('message','sin datos')}"}
            return
        candles = [{"c": float(v["close"]), "dt": v["datetime"]}
                   for v in reversed(data["values"])]
        print(f"  {len(candles)} velas descargadas")
    except Exception as e:
        _backtest_result["data"] = {"error": str(e)}
        return

    closes = [c["c"] for c in candles]

    def _atr(p):
        if len(p)<15: return p[-1]*0.003
        return sum(abs(p[i]-p[i-1]) for i in range(len(p)-14,len(p)))/14
    def _ema(a,n):
        if len(a)<n: return None
        k=2/(n+1); e=sum(a[:n])/n
        for v in a[n:]: e=v*k+e*(1-k)
        return e
    def _rsi(p):
        if len(p)<15: return 50
        g=l=0
        for i in range(len(p)-14,len(p)):
            d=p[i]-p[i-1]
            if d>0: g+=d
            else: l-=d
        ag,al=g/14,l/14
        return 100-100/(1+ag/al) if al>0 else 100
    def _bias(p):
        if len(p)<22: return None
        e9=_ema(p,9); e21=_ema(p,21)
        if e9 and e21:
            if e9>e21*1.0005: return "bullish"
            if e9<e21*0.9995: return "bearish"
        return None
    def _chop(p,n=20):
        if len(p)<n: return True
        r=p[-n:]; atr=sum(abs(r[i]-r[i-1]) for i in range(1,len(r)))/(len(r)-1)
        return (max(r)-min(r))<atr*5

    trades = []
    skipped = {"no_session":0,"chop":0,"no_bias":0,"no_sweep":0,"no_bos":0,"low_score":0,"total_bars":0}

    for i in range(50, len(closes)-6):
        skipped["total_bars"] += 1
        w = closes[:i]; price = closes[i]; atr_v = _atr(w)
        dt = candles[i]["dt"]
        try: hour = int(dt[11:13])
        except: hour = 12
        if not ((3<=hour<11) or (12<=hour<17)): skipped["no_session"]+=1; continue
        if _chop(w[-20:]): skipped["chop"]+=1; continue
        w15 = closes[max(0,i-60):i:3]
        bias = _bias(w15 if len(w15)>=22 else w)
        if not bias: skipped["no_bias"]+=1; continue
        # Sweep
        swept = False
        if bias=="bullish":
            lows=[w[j] for j in range(2,len(w)-2) if w[j]<=w[j-1] and w[j]<=w[j+1] and w[j]<=w[j-2] and w[j]<=w[j+2]]
            if lows and min(w[-5:])<lows[-1] and w[-1]>lows[-1]: swept=True
        else:
            highs=[w[j] for j in range(2,len(w)-2) if w[j]>=w[j-1] and w[j]>=w[j+1] and w[j]>=w[j-2] and w[j]>=w[j+2]]
            if highs and max(w[-5:])>highs[-1] and w[-1]<highs[-1]: swept=True
        if not swept: skipped["no_sweep"]+=1; continue
        # BOS
        bos = False; bos_t = "none"
        if bias=="bullish":
            hs=[w[j] for j in range(2,len(w)-2) if w[j]>=w[j-1] and w[j]>=w[j+1]]
            if hs and w[-1]>hs[-1]: bos=True; bos_t="BOS alcista"
            elif len(w)>=10 and w[-1]>w[-5] and w[-5]<w[-10]: bos=True; bos_t="CHOCH"
        else:
            ls=[w[j] for j in range(2,len(w)-2) if w[j]<=w[j-1] and w[j]<=w[j+1]]
            if ls and w[-1]<ls[-1]: bos=True; bos_t="BOS bajista"
            elif len(w)>=10 and w[-1]<w[-5] and w[-5]>w[-10]: bos=True; bos_t="CHOCH"
        if not bos: skipped["no_bos"]+=1; continue
        # Score
        score = 65
        # FVG
        fvg=False
        for k in range(len(w)-4,len(w)-1):
            if k<2: continue
            l1=w[k-2]; h3=w[k]
            if bias=="bullish" and h3>l1 and (h3-l1)>price*0.0008: fvg=True; break
            if bias=="bearish" and h3<l1 and (l1-h3)>price*0.0008: fvg=True; break
        if fvg: score+=15
        if bias==_bias(w): score+=10
        r=_rsi(w)
        if bias=="bullish":
            if r>75: score-=15
            elif r>65: score-=5
            elif r<40: score+=5
        else:
            if r<25: score-=15
            elif r<35: score-=5
            elif r>60: score+=5
        if score<70: skipped["low_score"]+=1; continue
        is_buy=bias=="bullish"
        tp_mult=2.5 if score>=85 else 1.5
        tp=price+atr_v*tp_mult if is_buy else price-atr_v*tp_mult
        sl=price-atr_v if is_buy else price+atr_v
        future=closes[i+1:i+7]
        if not future: continue
        if is_buy: hit_tp=any(p>=tp for p in future); hit_sl=any(p<=sl for p in future)
        else:       hit_tp=any(p<=tp for p in future); hit_sl=any(p>=sl for p in future)
        if hit_tp and hit_sl:
            ti=next((j for j,p in enumerate(future) if (p>=tp if is_buy else p<=tp)),999)
            si=next((j for j,p in enumerate(future) if (p<=sl if is_buy else p>=sl)),999)
            won=ti<si
        elif hit_tp: won=True
        elif hit_sl: won=False
        else: won=False
        sess="Londres" if 3<=hour<11 else "New York"
        trades.append({"dt":dt,"hour":hour,"sess":sess,
            "dir":"BUY" if is_buy else "SELL","score":score,
            "entry":round(price,2),"tp":round(tp,2),"sl":round(sl,2),
            "rr":round(tp_mult,1),"won":won,"pnl_r":round(tp_mult,1) if won else -1.0,
            "bos":bos_t,"rsi":round(r,1)})
        if len(trades)>=300: break

    if not trades:
        _backtest_result["data"] = {"error": "Sin trades — pocas velas o mercado sin estructura"}
        return

    n=len(trades); wins=[t for t in trades if t["won"]]; losses=[t for t in trades if not t["won"]]
    wr=round(len(wins)/n*100,1); total_r=round(sum(t["pnl_r"] for t in trades),2)
    avg_win=round(sum(t["rr"] for t in wins)/len(wins),2) if wins else 0
    gw=sum(t["pnl_r"] for t in wins) if wins else 0
    gl=abs(sum(t["pnl_r"] for t in losses)) if losses else 0.001
    pf=round(gw/gl,2); exp=round(total_r/n,3)
    eq=[0.0]
    for t in trades: eq.append(round(eq[-1]+t["pnl_r"],2))
    peak=eq[0]; max_dd=0
    for e in eq:
        if e>peak: peak=e
        dd=peak-e
        if dd>max_dd: max_dd=dd
    rets=[t["pnl_r"] for t in trades]; mean_r=sum(rets)/len(rets)
    std_r=math.sqrt(sum((r-mean_r)**2 for r in rets)/len(rets)) if len(rets)>1 else 0.01
    sharpe=round(mean_r/std_r*math.sqrt(252),2)
    by_sess={}
    for t in trades:
        s=t["sess"]
        if s not in by_sess: by_sess[s]={"n":0,"w":0,"r":0.0}
        by_sess[s]["n"]+=1; by_sess[s]["w"]+=int(t["won"]); by_sess[s]["r"]+=t["pnl_r"]
    for s in by_sess: by_sess[s]["wr"]=round(by_sess[s]["w"]/by_sess[s]["n"]*100,1); by_sess[s]["r"]=round(by_sess[s]["r"],2)
    snipers=[t for t in trades if t["score"]>=85]; normals=[t for t in trades if 70<=t["score"]<85]
    buys=[t for t in trades if t["dir"]=="BUY"]; sells=[t for t in trades if t["dir"]=="SELL"]
    fr=round((skipped["total_bars"]-n)/max(skipped["total_bars"],1)*100,1)
    mw=ml=cw=cl=0; ct=None
    for t in trades:
        if t["won"]: cw=cw+1 if ct=="w" else 1; cl=0; ct="w"; mw=max(mw,cw)
        else:        cl=cl+1 if ct=="l" else 1; cw=0; ct="l"; ml=max(ml,cl)

    _backtest_result["data"] = {
        "n":n,"wins":len(wins),"losses":len(losses),"wr":wr,"total_r":total_r,
        "avg_win":avg_win,"pf":pf,"expectancy":exp,"max_dd":round(max_dd,2),"sharpe":sharpe,
        "by_sess":by_sess,"eq":eq,"trades":trades[-60:],
        "snipers":len(snipers),"normals":len(normals),
        "sn_wr":round(len([t for t in snipers if t["won"]])/max(len(snipers),1)*100,1),
        "no_wr":round(len([t for t in normals if t["won"]])/max(len(normals),1)*100,1),
        "buy_wr":round(len([t for t in buys if t["won"]])/max(len(buys),1)*100,1),
        "sell_wr":round(len([t for t in sells if t["won"]])/max(len(sells),1)*100,1),
        "buys":len(buys),"sells":len(sells),
        "max_win_streak":mw,"max_loss_streak":ml,
        "filter_rate":fr,"skipped":skipped,
        "candles":len(candles),
        "data_from":candles[0]["dt"][:16],"data_to":candles[-1]["dt"][:16],
        "generated_at":_dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
    }
    print(f"  Backtest OK: {n} trades | WR={wr}% | PF={pf} | R={total_r}")


def _build_backtest_html():
    d = _backtest_result.get("data")
    if d is None:
        return """<!DOCTYPE html><html><body style="background:#080600;color:#C9A84C;font-family:monospace;padding:40px">
<h2>Backtest corriendo...</h2><p>Refresca en 30 segundos.</p>
<script>setTimeout(()=>location.reload(),5000)</script></body></html>"""
    if "error" in d:
        return f"""<!DOCTYPE html><html><body style="background:#080600;color:#CC3344;font-family:monospace;padding:40px">
<h2>Error</h2><p>{d['error']}</p>
<p style="color:#5A4820;margin-top:20px">Ve a /backtest para reintentar</p></body></html>"""
    eq=d["eq"]; W,H=900,200
    mn=min(eq)-0.5; mx=max(eq)+0.5
    def px(i): return (i/max(len(eq)-1,1))*W
    def py(v): return H-((v-mn)/(mx-mn+0.01))*(H-30)-10
    pts=" ".join(f"{px(i):.1f},{py(v):.1f}" for i,v in enumerate(eq))
    ec="#00CC88" if eq[-1]>=0 else "#CC3344"
    fp=f"0,{H} "+pts+f" {W},{H}"
    svg=f"""<svg width="100%" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
<stop offset="0%" stop-color="{ec}" stop-opacity="0.25"/>
<stop offset="100%" stop-color="{ec}" stop-opacity="0.02"/></linearGradient></defs>
<rect width="{W}" height="{H}" fill="#0A0800"/>
<polygon points="{fp}" fill="url(#g)"/>
<line x1="0" y1="{py(0):.1f}" x2="{W}" y2="{py(0):.1f}" stroke="rgba(201,168,76,0.4)" stroke-width="1" stroke-dasharray="6,3"/>
<polyline points="{pts}" fill="none" stroke="{ec}" stroke-width="2.5"/>
<circle cx="{px(len(eq)-1):.1f}" cy="{py(eq[-1]):.1f}" r="4" fill="{ec}"/></svg>"""
    sess_rows="".join(f"<tr><td>{s}</td><td>{sv['n']}</td><td style='color:{'#00CC88' if sv['wr']>=55 else '#CC3344'}'>{sv['wr']}%</td><td style='color:{'#00CC88' if sv['r']>=0 else '#CC3344'}'>{'+'if sv['r']>=0 else''}{sv['r']}R</td></tr>" for s,sv in d["by_sess"].items())
    trade_rows=""
    for t in reversed(d["trades"]):
        c="#00CC88" if t["won"] else "#CC3344"
        dc="#00CC88" if t["dir"]=="BUY" else "#CC3344"
        sc="#F0D080" if t["score"]>=85 else "#5A4820"
        pl=f"+{t['pnl_r']}R" if t["pnl_r"]>=0 else f"{t['pnl_r']}R"
        trade_rows+=f"<tr><td style='color:#5A4820'>{t['dt'][5:16]}</td><td style='color:{dc}'>{t['dir']}</td><td>${t['entry']}</td><td style='color:#00CC88'>${t['tp']}</td><td style='color:#CC3344'>${t['sl']}</td><td>{t['rr']}:1</td><td style='color:{sc}'>{t['score']}</td><td style='color:{c}'>{'WIN' if t['won'] else 'LOSS'}</td><td style='color:{c}'>{pl}</td><td style='color:#5A4820'>{t['sess']}</td></tr>"
    pfc="#00CC88" if d["pf"]>=1.5 else "#CC3344"
    wrc="#00CC88" if d["wr"]>=55 else "#CC3344"
    rc="#00CC88" if d["total_r"]>=0 else "#CC3344"
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AURUM SCALP v2 - Backtest</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=IBM+Plex+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root{{--bg:#080600;--bg2:#0C0A02;--gold:#C9A84C;--green:#00CC88;--red:#CC3344;--dim:#5A4820;--border:rgba(201,168,76,0.1)}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:radial-gradient(ellipse at 50% 0%,#0F0C02 0%,#080600 50%,#000 100%);color:#E8E0C0;font-family:"IBM Plex Mono",monospace;padding:40px 32px}}
h1{{font-family:"Playfair Display",serif;font-size:28px;font-weight:700;letter-spacing:6px;background:linear-gradient(135deg,#8B6914,#F0D080,#C9A84C);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.sub{{font-size:9px;letter-spacing:3px;color:var(--dim);margin:6px 0 24px}}
.g4{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
.card{{background:var(--bg2);border:1px solid var(--border);padding:18px;position:relative}}
.card::before{{content:"";position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--gold),transparent);opacity:.25}}
.cl{{font-size:8px;letter-spacing:3px;color:var(--dim);margin-bottom:8px}}
.cv{{font-family:"Playfair Display",serif;font-size:30px;font-weight:700;line-height:1}}
.cs{{font-size:8px;color:#3D3010;margin-top:4px}}
.sec{{font-size:8px;letter-spacing:4px;color:var(--dim);margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--border)}}
.panel{{background:var(--bg2);border:1px solid var(--border);padding:16px;margin-bottom:20px}}
table{{width:100%;border-collapse:collapse;font-size:9px}}
th{{font-size:7px;letter-spacing:2px;color:var(--dim);padding:8px 10px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}}
td{{padding:6px 10px;border-bottom:1px solid rgba(201,168,76,0.04);white-space:nowrap}}
::-webkit-scrollbar{{width:4px}}::-webkit-scrollbar-track{{background:var(--bg)}}
::-webkit-scrollbar-thumb{{background:var(--dim);border-radius:2px}}
@media(max-width:700px){{.g4{{grid-template-columns:1fr 1fr}}.g2{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>AURUM SCALP v2</h1>
<div class="sub">BACKTEST REAL · XAUUSD 5M · {d["data_from"]} &rarr; {d["data_to"]} · {d["candles"]} velas · {d["generated_at"]} UTC</div>
<div class="g4">
  <div class="card"><div class="cl">WIN RATE</div><div class="cv" style="color:{wrc}">{d["wr"]}%</div><div class="cs">{d["wins"]}W / {d["losses"]}L</div></div>
  <div class="card"><div class="cl">PROFIT FACTOR</div><div class="cv" style="color:{pfc}">{d["pf"]}</div><div class="cs">objetivo &ge; 1.5</div></div>
  <div class="card"><div class="cl">TOTAL R</div><div class="cv" style="color:{rc}">{'+'if d['total_r']>=0 else''}{d["total_r"]}R</div><div class="cs">{d["n"]} trades</div></div>
  <div class="card"><div class="cl">EXPECTANCY</div><div class="cv" style="color:{'#00CC88' if d['expectancy']>=0.3 else '#CC3344'}">{'+'if d['expectancy']>=0 else''}{d["expectancy"]}R</div><div class="cs">por trade</div></div>
</div>
<div class="g4">
  <div class="card"><div class="cl">MAX DRAWDOWN</div><div class="cv" style="color:{'#00CC88' if d['max_dd']<=3 else '#CC3344'}">{d["max_dd"]}R</div><div class="cs">consecutivo</div></div>
  <div class="card"><div class="cl">SHARPE</div><div class="cv" style="color:{'#00CC88' if d['sharpe']>=1 else '#CC3344'}">{d["sharpe"]}</div><div class="cs">objetivo &ge; 1.0</div></div>
  <div class="card"><div class="cl">SNIPER vs NORMAL</div><div class="cv" style="font-size:18px"><span style="color:var(--gold)">{d["snipers"]}</span> <span style="color:#3D3010">/</span> <span style="color:var(--dim)">{d["normals"]}</span></div><div class="cs">WR {d["sn_wr"]}% / {d["no_wr"]}%</div></div>
  <div class="card"><div class="cl">BUY / SELL</div><div class="cv" style="font-size:18px"><span style="color:var(--green)">{d["buys"]}</span> <span style="color:#3D3010">/</span> <span style="color:var(--red)">{d["sells"]}</span></div><div class="cs">WR {d["buy_wr"]}% / {d["sell_wr"]}%</div></div>
</div>
<div class="panel">
  <div class="sec">CURVA DE EQUITY (en R)</div>
  {svg}
  <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:8px;color:var(--dim)"><span>0.0R</span><span>{'+'if eq[-1]>=0 else''}{eq[-1]}R</span></div>
</div>
<div class="g2">
  <div><div class="sec">POR SESION</div><div class="panel"><table><thead><tr><th>SESION</th><th>TRADES</th><th>WIN RATE</th><th>TOTAL R</th></tr></thead><tbody>{sess_rows}</tbody></table></div></div>
  <div><div class="sec">SCORE TIERS</div><div class="panel" style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
    <div style="text-align:center;padding:14px;border:1px solid var(--border)"><div style="font-family:'Playfair Display',serif;font-size:26px;color:var(--gold)">{d["snipers"]}</div><div style="font-size:7px;letter-spacing:2px;color:var(--dim);margin-top:4px">SNIPER &ge;85</div><div style="font-size:11px;margin-top:6px;color:{'#00CC88' if d['sn_wr']>=55 else '#CC3344'}">WR {d["sn_wr"]}%</div></div>
    <div style="text-align:center;padding:14px;border:1px solid var(--border)"><div style="font-family:'Playfair Display',serif;font-size:26px;color:var(--dim)">{d["normals"]}</div><div style="font-size:7px;letter-spacing:2px;color:var(--dim);margin-top:4px">NORMAL 70-84</div><div style="font-size:11px;margin-top:6px;color:{'#00CC88' if d['no_wr']>=55 else '#CC3344'}">WR {d["no_wr"]}%</div></div>
  </div></div>
</div>
<div><div class="sec">ULTIMOS 60 TRADES</div>
<div style="overflow-x:auto" class="panel"><table>
<thead><tr><th>DATETIME</th><th>DIR</th><th>ENTRY</th><th>TP</th><th>SL</th><th>RR</th><th>SCORE</th><th>RESULTADO</th><th>P&L</th><th>SESION</th></tr></thead>
<tbody>{trade_rows}</tbody></table></div></div>
<div style="margin-top:32px;font-size:8px;color:#3D3010;line-height:2;padding:14px;border:1px solid var(--border)">
Backtest real XAUUSD 5M · Twelve Data · Motor SCALP v2 · Sweep+BOS obligatorio · Sin slippage ni spread · Resultados pasados no garantizan resultados futuros.
</div></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path   = self.path.split('?')[0]
        params = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.path).query))

        if path == "/":
            self._send(200, "text/html; charset=utf-8", HTML.encode("utf-8"))

        elif path == "/precio":
            c = cached("precio", ttl=5)
            if not c:
                c = get_gold_price()
                if c: set_cache("precio", c)
            if c and c.get("price"):
                push_price(c["price"])
                update_ict_prices(c["price"])
            self._send(200, "application/json",
                       json.dumps(c or {"price":0,"ch":0,"chp":0}).encode())

        elif path == "/ohlc":
            interval = params.get("interval", "5min")
            size     = int(params.get("size", "150"))
            candles  = get_historical_ohlc(interval, size)
            if candles:
                for c2 in candles:
                    push_price(c2["close"])
                    update_ict_prices(c2["close"])
            self._send(200, "application/json", json.dumps(candles or []).encode())

        elif path == "/ictsignal":
            atr_val = 0.0
            if len(_ict_prices_5m) >= 15:
                atr_val = sum(abs(_ict_prices_5m[i]-_ict_prices_5m[i-1])
                              for i in range(len(_ict_prices_5m)-14, len(_ict_prices_5m))) / 14
            cp = _ict_prices_5m[-1] if _ict_prices_5m else 0
            result = run_ict_engine(cp, atr_val) if cp else None
            if result and result["score"] >= SCORE_SNIPER and TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                threading.Thread(target=send_telegram_direct,
                    args=(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, result["msg"]), daemon=True).start()
            self._send(200, "application/json", json.dumps(result or {}).encode())

        elif path == "/log":
            try:
                log_signal(params.get("signal",""), float(params.get("price",0)),
                           float(params.get("tp",0)), float(params.get("sl",0)),
                           float(params.get("atr",0)), params.get("rr",""),
                           params.get("conf",""), params.get("session",""))
                self._send(200, "application/json", b'{"ok":true}')
            except:
                self._send(500, "application/json", b'{"ok":false}')

        elif path == "/getlog":
            self._send(200, "application/json", json.dumps(read_log()).encode())

        elif path == "/download":
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "rb") as f: data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", f'attachment; filename="{LOG_FILE}"')
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send(404, "text/plain", b"Sin datos aun")

        elif path == "/telegram":
            token = params.get("token",""); chat = params.get("chat",""); msg = params.get("msg","")
            if token and chat and msg:
                threading.Thread(target=send_telegram_direct, args=(token, chat, msg), daemon=True).start()
            self._send(200, "application/json", b'{"ok":true}')

        elif path == "/aitrain":
            p_list = [float(x) for x in params.get("prices","").split(",") if x]
            if p_list:
                threading.Thread(target=ai_train_if_needed, args=(p_list,), daemon=True).start()
            self._send(200, "application/json", json.dumps({
                "trained": _ai.trained, "accuracy": round(_ai.accuracy,1), "epochs": _ai.epochs}).encode())

        elif path == "/aipredict":
            p_list = [float(x) for x in params.get("prices","").split(",") if x]
            prob, signal = ai_predict(p_list) if p_list else (None, None)
            self._send(200, "application/json", json.dumps({
                "prob": round(prob*100,1) if prob is not None else None,
                "signal": signal, "trained": _ai.trained,
                "accuracy": round(_ai.accuracy,1)}).encode())

        elif path == "/newsstatus":
            c = cached("newsstatus", ttl=30)
            if not c:
                news_on, news_name = is_news_time()
                c = {"active": news_on, "event": news_name or "",
                     "pause_before": PAUSE_MINUTES_BEFORE, "pause_after": PAUSE_MINUTES_AFTER}
                set_cache("newsstatus", c)
            self._send(200, "application/json", json.dumps(c).encode())

        elif path == "/htftrend":
            c = cached("htftrend_r", ttl=60)
            if not c:
                c = {"trend": get_htf_trend()}
                set_cache("htftrend_r", c)
            self._send(200, "application/json", json.dumps(c).encode())

        elif path == "/sendimage":
            if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                try:
                    sig  = params.get("signal",""); price = float(params.get("price",0))
                    tp   = float(params.get("tp",0)); sl = float(params.get("sl",0))
                    rr   = params.get("rr","1.5"); conf = params.get("conf","70")
                    sess = params.get("session",""); atr = float(params.get("atr",0))
                    svg  = generate_signal_image(sig, price, tp, sl, rr, conf, sess, atr)
                    em   = "🟢" if sig=="COMPRAR" else "🔴"
                    caption = (f"{em} *AURUM · {sig}*\n💰 Entrada: ${price:.2f}\n"
                               f"🎯 TP: ${tp:.2f}\n🛡 SL: ${sl:.2f}\n"
                               f"📊 RR: {rr}:1\n💪 Confianza: {conf}%\n🌍 Sesión: {sess}")
                    threading.Thread(target=send_telegram_direct,
                        args=(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, caption, svg), daemon=True).start()
                    self._send(200, "application/json", b'{"ok":true}')
                except Exception as e:
                    print(f"  ⚠ sendimage error: {e}")
                    self._send(500, "application/json", b'{"ok":false}')
            else:
                self._send(200, "application/json", b'{"ok":false,"reason":"no telegram"}')

        elif path == "/aistats":
            hour_acc = _ai.get_hour_accuracy(); dow_acc = _ai.get_dow_accuracy()
            self._send(200, "application/json", json.dumps({
                "trained": _ai.trained, "accuracy": round(_ai.accuracy,1),
                "epochs": _ai.epochs, "drift_score": round(_ai.drift_score,2),
                "is_drifting": _ai.is_market_drifting(),
                "hour_accuracy": round(hour_acc,1) if hour_acc else None,
                "dow_accuracy":  round(dow_acc,1)  if dow_acc  else None,
                "models": ["LogisticRegression","DecisionTree"], "ensemble": "60% LR + 40% DT"
            }).encode())

        elif path == "/riskstatus":
            can_trade, reason = check_risk_limits()
            self._send(200, "application/json", json.dumps({
                "can_trade": can_trade, "reason": reason,
                "consecutive_losses": _consecutive_losses,
                "consecutive_wins":   _consecutive_wins,
                "daily_signals":      _daily_signals,
                "max_daily":          MAX_DAILY_SIGNALS,
                "max_losses":         MAX_CONSECUTIVE_LOSSES,
            }).encode())

        elif path == "/healthcheck":
            self._send(200, "application/json", json.dumps({
                "status": "ok", "time_utc": _dt.datetime.utcnow().isoformat(),
                "ai_ready": _ai.trained, "ai_drift": _ai.is_market_drifting(), "uptime_ok": True,
            }).encode())

        # [FIX] /scalpscore — corregidas funciones inexistentes
        elif path == "/scalpscore":
            atr_v = 0
            if len(_ict_prices_5m) >= 14:
                atr_v = sum(abs(_ict_prices_5m[i]-_ict_prices_5m[i-1])
                            for i in range(len(_ict_prices_5m)-14, len(_ict_prices_5m))) / 14
            # [FIX] get_bias_15m → get_ema_bias (función que sí existe)
            bias = get_ema_bias(_ict_prices_15m if len(_ict_prices_15m)>=12 else _ict_prices_5m)
            self._send(200, "application/json", json.dumps({
                "session_active":   is_trading_session(),
                "session_name":     get_session_name(),
                # [FIX] is_trending_market → not detect_chop (función que sí existe)
                "trending":         not detect_chop(_ict_prices_5m) if _ict_prices_5m else False,
                "bias_15m":         bias or "NEUTRAL",
                "prices_5m":        len(_ict_prices_5m),
                "prices_15m":       len(_ict_prices_15m),
                "atr":              round(atr_v, 2),
                # [FIX] SCORE_SETUP → SCORE_NORMAL (constante que sí existe)
                "score_thresholds": {"sniper": SCORE_SNIPER, "normal": SCORE_NORMAL},
            }).encode())

        elif path == "/modelhealth":
            health = _ai.health_check()
            import glob
            versions     = glob.glob(f"{AurumAI.MODEL_DIR}/aurum_model_v*.bak")
            dataset_size = 0
            if os.path.exists(AurumAI.DATASET_FILE):
                try:
                    with open(AurumAI.DATASET_FILE, "rb") as f: ds = pickle.load(f)
                    dataset_size = len(ds.get("X", []))
                except: pass
            lock_remaining = max(0, int((AurumAI._state_lock_until - time.time()) / 60))
            self._send(200, "application/json", json.dumps({
                **health,
                "model_versions":    len(versions),
                "dataset_samples":   dataset_size,
                "model_file":        AurumAI.MODEL_FILE,
                "model_exists":      os.path.exists(AurumAI.MODEL_FILE),
                "current_state":     AurumAI._current_state,
                "state_lock_min":    lock_remaining,
                "conservative_mode": AurumAI.CONSERVATIVE_MODE,
            }).encode())

        elif path == "/controlstate":
            ph = price_history[-50:] if len(price_history) >= 50 else (price_history if price_history else [0])
            news_on, _ = is_news_time()
            m_sig,  m_lvl              = ModelLayer.get_signal(_ai.accuracy, _ai.drift_score, _ai.epochs)
            mk_sig, mk_lvl, mk_factors = MarketLayer.get_signal(ph, news_on)
            lock_remaining = max(0, int((_current_control["lock_until"] - time.time()) / 60))
            self._send(200, "application/json", json.dumps({
                "control_state":  _current_control["state"],
                "thresholds":     _current_control["thresholds"],
                "combined_score": _current_control["combined"],
                "lock_remaining": lock_remaining,
                "model_layer":    {"signal": m_sig, "level": m_lvl},
                "market_layer":   {"signal": mk_sig, "level": mk_lvl, "factors": mk_factors},
                "news_active":    news_on,
            }).encode())

        elif path == "/backtest":
            threading.Thread(target=_run_backtest_bg, daemon=True).start()
            self._send(200, "application/json", json.dumps({"ok": True, "msg": "Backtest iniciado - listo en ~30s en /backtest_result"}).encode())

        elif path == "/backtest_result":
            result = _backtest_result.get("data")
            if result is None:
                self._send(200, "application/json", json.dumps({"status": "running"}).encode())
            else:
                self._send(200, "application/json", json.dumps(result).encode())

        elif path == "/backtest_report":
            report_html = _build_backtest_html()
            self._send(200, "text/html; charset=utf-8", report_html.encode("utf-8"))

        else:
            self._send(404, "text/plain", b"Not found")

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "keep-alive")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        p = str(args[0])
        if any(x in p for x in ["/precio", "/log", "/telegram"]):
            print(f"  → {p.split()[0]} {p.split()[1] if len(p.split())>1 else ''} [{args[1]}]")


def main():
    init_log()
    print("=" * 54)
    print("  AURUM v6.0 · Bot de Señales XAU/USD")
    print("  OHLC real ✓ | ML validation ✓ | Thread-safe ✓ | SCALP v2 ✓")
    print("=" * 54)

    print("\n  Verificando precio del oro...")
    test = get_gold_price()
    if test:
        ch = test['ch']
        print(f"  ✓ XAU/USD: ${test['price']:.2f}  ({'+' if ch>=0 else ''}{ch:.2f})")

    print("\n  Probando datos históricos (5min)...")
    candles = get_historical_ohlc("5min", 50)
    if candles:
        print(f"  ✓ Twelve Data OK — {len(candles)} velas | última: ${candles[-1]['close']:.2f}")
        for c in candles:
            push_price(c["close"])
            update_ict_prices(c["close"])
    else:
        print("  ⚠ Twelve Data no disponible")

    if len(price_history) >= 60:
        print("\n  Pre-entrenando IA...")
        _ai.train(price_history)
        if _ai.trained:
            print(f"  ✓ IA lista: {_ai.accuracy:.1f}% acc | {_ai.epochs} muestras")

    if price_history:
        news_on, _ = is_news_time()
        ph = price_history[-50:] if len(price_history) >= 50 else price_history
        update_control_state(_ai, ph, is_news=news_on)

    threading.Thread(target=background_worker, daemon=True).start()
    print("  ✓ Background worker activo (Control Layer + IA cada 5 min)")

    print(f"\n  Servidor: http://localhost:{PORT}")
    print(f"  Dashboard: https://aurum-signals.onrender.com")
    print(f"  Registro: {LOG_FILE}")
    print("\n  Para detener: Ctrl + C")
    print("-" * 54)

    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n  AURUM detenido. ¡Buena suerte con la prop firm!")
        server.shutdown()


if __name__ == "__main__":
    main()
