"""aurum_models.py — ML models, AI engine, performance tracking."""
import math, time, os, pickle
import datetime as _dt

from aurum_state import (
    _data_lock, _control_lock,
    _ohlc_candles_5m, _ohlc_candles_15m, price_history,
    _current_control, _ai_timers,
    _signal_history, _perf_counters,
    _performance_monitor, PERF_WINDOW, PERF_MIN_N,
    MAX_CONSECUTIVE_LOSSES, MAX_DAILY_SIGNALS,
)


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


def update_control_state(ai_instance, prices, is_news=False):
    if not prices: return dict(_current_control)
    m_sig,  m_lvl              = ModelLayer.get_signal(ai_instance.accuracy, ai_instance.drift_score, ai_instance.epochs)
    mk_sig, mk_lvl, mk_factors = MarketLayer.get_signal(prices, is_news)
    decision = ControlLayer.decide(m_sig, m_lvl, mk_sig, mk_lvl)
    now = time.time()
    with _control_lock:
        if decision["state"] != _current_control["state"]:
            if now >= _current_control["lock_until"]:
                print(f"  🎛 CONTROL: {_current_control['state']} → {decision['state']}")
                _current_control.clear()
                _current_control.update({
                    "state":      decision["state"],
                    "thresholds": decision["thresholds"],
                    "lock_until": now + decision["lock_sec"],
                    "combined":   decision["combined"],
                })
            else:
                remaining = int((_current_control["lock_until"] - now) / 60)
                print(f"  🔒 LOCK: {_current_control['state']} | {remaining}min")
        else:
            _current_control["thresholds"] = decision["thresholds"]
            _current_control["combined"]   = decision["combined"]
        return dict(_current_control)


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
    class _Node:
        __slots__ = ("feature", "threshold", "left", "right", "value")
        def __init__(self):
            self.feature = self.threshold = self.left = self.right = self.value = None

    def __init__(self, max_depth=6, min_samples=10):
        self.max_depth   = max_depth
        self.min_samples = min_samples
        self.root        = None
        self.trained     = False

    @staticmethod
    def _gini(y):
        if not y: return 0.0
        p = sum(y) / len(y)
        return 1.0 - p * p - (1.0 - p) * (1.0 - p)

    def _best_split(self, X, y):
        best_gain = 1e-9; best_feat = None; best_thresh = None
        n = len(y); pg = self._gini(y)
        for fi in range(len(X[0])):
            vals = sorted(set(x[fi] for x in X))
            if len(vals) < 2: continue
            thresholds = [(vals[i] + vals[i+1]) * 0.5 for i in range(len(vals)-1)]
            if len(thresholds) > 20:
                step = max(1, len(thresholds) // 20)
                thresholds = thresholds[::step]
            for t in thresholds:
                ly = [y[i] for i in range(n) if X[i][fi] <= t]
                ry = [y[i] for i in range(n) if X[i][fi] >  t]
                if len(ly) < self.min_samples or len(ry) < self.min_samples: continue
                gain = pg - (len(ly)/n * self._gini(ly) + len(ry)/n * self._gini(ry))
                if gain > best_gain:
                    best_gain = gain; best_feat = fi; best_thresh = t
        return best_feat, best_thresh

    def _build(self, X, y, depth):
        node = self._Node()
        if depth >= self.max_depth or len(y) < self.min_samples * 2:
            node.value = sum(y) / len(y) if y else 0.5
            return node
        feat, thresh = self._best_split(X, y)
        if feat is None:
            node.value = sum(y) / len(y)
            return node
        node.feature = feat; node.threshold = thresh
        lm = [i for i in range(len(y)) if X[i][feat] <= thresh]
        rm = [i for i in range(len(y)) if X[i][feat] >  thresh]
        node.left  = self._build([X[i] for i in lm], [y[i] for i in lm], depth + 1)
        node.right = self._build([X[i] for i in rm], [y[i] for i in rm], depth + 1)
        return node

    def _predict_one(self, node, x):
        if node.value is not None: return node.value
        branch = node.left if x[node.feature] <= node.threshold else node.right
        return self._predict_one(branch, x)

    def train(self, X, y):
        if len(X) < self.min_samples * 2: return
        self.root    = self._build(X, y, 0)
        self.trained = True

    def predict(self, x):
        if not self.trained or self.root is None: return 0.5
        return self._predict_one(self.root, x)


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
            "tree_root": self.tree.root, "tree_trained": self.tree.trained,
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
                if len(self.logistic.w) < 11:
                    self.logistic.w.extend([0.0] * (11 - len(self.logistic.w)))
                self.logistic.b        = d.get("logistic_b", 0.0)
                self.logistic.trained  = d.get("logistic_trained", False)
                self.tree.root         = d.get("tree_root", None)
                self.tree.trained      = d.get("tree_trained", False) and self.tree.root is not None
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
        atrs    = [abs(p[i] - p[i-1]) for i in range(len(p)-14, len(p))]
        atr_val = sum(atrs) / 14
        atr_norm = max(0, min(2, atr_val/(p[-1]*0.01+1e-9)))
        sma20 = sum(p[-20:])/20
        pvs   = max(-2, min(2, (p[-1]-sma20)/(sma20+1e-9)*100))
        mom5  = max(-2, min(2, (p[-1]-p[-6])/(p[-6]+1e-9)*100))
        mom10 = max(-2, min(2, (p[-1]-p[-11])/(p[-11]+1e-9)*100)) if len(p) > 11 else 0
        rets  = [(p[i]-p[i-1])/(p[i-1] + 1e-9) for i in range(len(p)-10, len(p))]
        mean_r = sum(rets)/len(rets)
        vol = math.sqrt(sum((r-mean_r)**2 for r in rets)/len(rets)) * 100
        h = hour if hour is not None else _dt.datetime.utcnow().hour
        d = dow  if dow  is not None else _dt.datetime.utcnow().weekday()
        hour_norm = math.sin(2 * math.pi * h / 24)
        dow_norm  = d / 6.0
        vol_rel = max(0, min(2, vol / 0.15)) if vol > 0 else 0.5
        return [rsi, ema_cross, macd_norm, atr_norm, pvs, mom5, mom10,
                max(0, min(2, vol)), hour_norm, dow_norm, vol_rel]

    def detect_drift(self):
        if len(self.recent_errors) < 20: return 0.0
        window  = self.recent_errors[-30:]
        n       = len(window)
        weights = [math.exp(0.1 * i) for i in range(n)]
        w_sum   = sum(weights)
        self.drift_score = sum(w * e for w, e in zip(weights, window)) / w_sum
        return self.drift_score

    def is_market_drifting(self):
        return self.detect_drift() > 0.65

    @staticmethod
    def _label_with_sl_tp(prices, i, lookahead=10, tp_pct=0.0015, sl_pct=0.001):
        entry = prices[i]
        tp = entry * (1.0 + tp_pct)
        sl = entry * (1.0 - sl_pct)
        for j in range(i + 1, min(i + lookahead + 1, len(prices))):
            if prices[j] >= tp: return 1
            if prices[j] <= sl: return 0
        return 1 if prices[min(i + lookahead, len(prices) - 1)] > entry else 0

    @staticmethod
    def _balance_classes(X, y):
        pos = [i for i, v in enumerate(y) if v == 1]
        neg = [i for i, v in enumerate(y) if v == 0]
        if not pos or not neg: return X, y
        minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
        diff = len(majority) - len(minority)
        if diff <= 0: return X, y
        extra = [minority[i % len(minority)] for i in range(diff)]
        all_idx = list(range(len(y))) + extra
        return [X[i] for i in all_idx], [y[i] for i in all_idx]

    def train(self, prices):
        if len(prices) < 60: return False
        now = _dt.datetime.utcnow()
        X, y, hours, dows = [], [], [], []
        for i in range(40, len(prices) - 12):
            mins_ago = (len(prices) - i) * 5
            t = now - _dt.timedelta(minutes=mins_ago)
            features = self.extract_features(prices[i-35:i], hour=t.hour, dow=t.weekday())
            if not features: continue
            label = self._label_with_sl_tp(prices, i, lookahead=10, tp_pct=0.0015, sl_pct=0.001)
            X.append(features); y.append(label)
            hours.append(t.hour); dows.append(t.weekday())
        if len(X) < 30: return False
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_val,   y_val   = X[split:], y[split:]
        h_val, d_val     = hours[split:], dows[split:]
        X_train_b, y_train_b = self._balance_classes(X_train, y_train)
        self.logistic.train(X_train_b, y_train_b)
        self.tree.train(X_train_b, y_train_b)
        self.save_dataset(X_train, y_train)
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


# Singleton AI instance
_ai = AurumAI()


def ai_train_if_needed(prices):
    now = time.time()
    if now - _ai_timers["last_train"] > 300:
        with _data_lock:
            ohlc5_closes  = [c["c"] for c in _ohlc_candles_5m]
            ohlc15_closes = [c["c"] for c in _ohlc_candles_15m]
        combined = list(ohlc5_closes)
        if len(ohlc15_closes) > 0:
            combined = ohlc15_closes + combined
        if len(prices) > len(combined):
            combined = list(prices)
        if len(combined) < 60:
            return
        _ai.train(combined)
        _ai_timers["last_train"] = now
        if _ai.trained:
            drift = _ai.detect_drift()
            h_acc = _ai.get_hour_accuracy()
            d_acc = _ai.get_dow_accuracy()
            print(f"  🤖 Ensemble: {_ai.accuracy:.1f}% | Drift: {drift:.2f} | {_ai.epochs} muestras | dataset={len(combined)}")
            if h_acc: print(f"  🕐 Acc hora: {h_acc:.1f}%")
            if d_acc: print(f"  📅 Acc hoy: {d_acc:.1f}%")
            if _ai.is_market_drifting():
                print("  ⚠ DRIFT DETECTADO")
    if now - _ai_timers["last_backup"] > 3600 and _ai.trained:
        _ai.save(); _ai_timers["last_backup"] = now
        print("  💾 Backup automático guardado")


def ai_predict(prices):
    if not _ai.trained or len(prices) < 35: return None, None
    with _control_lock:
        ctrl_state  = _current_control["state"]
        buy_thresh  = _current_control["thresholds"]["buy"]
        sell_thresh = _current_control["thresholds"]["sell"]
    if ctrl_state == "PAUSED": return None, "PAUSED"
    prob = _ai.predict_proba(prices)
    if prob is None: return None, None
    if   prob >= buy_thresh:  signal = "COMPRAR"
    elif prob <= sell_thresh: signal = "VENDER"
    else:                     signal = "ESPERAR"
    return prob, signal


def update_performance_metrics():
    recent = _signal_history[-PERF_WINDOW:] if _signal_history else []
    n = len(recent)
    if n < PERF_MIN_N:
        _performance_monitor["status"] = "CALIBRATING"
        _performance_monitor["last_update"] = time.time()
        return
    wins = [s for s in recent if s["won"]]
    losses = [s for s in recent if not s["won"]]
    wr = len(wins) / n * 100
    gains = sum(s["pnl_r"] for s in wins)
    losses_sum = abs(sum(s["pnl_r"] for s in losses))
    pf = gains / (losses_sum + 1e-9)
    returns = [s["pnl_r"] for s in recent]
    mean_r = sum(returns) / n
    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / n) if n > 1 else 0.01
    sharpe = (mean_r / (std_r + 1e-9)) * math.sqrt(252 * 5)
    _performance_monitor["rolling_wr"]     = round(wr, 1)
    _performance_monitor["rolling_pf"]     = round(pf, 2)
    _performance_monitor["rolling_sharpe"] = round(sharpe, 2)
    _performance_monitor["last_update"]    = time.time()
    b_wr     = _performance_monitor["baseline_wr"]
    b_pf     = _performance_monitor["baseline_pf"]
    b_sharpe = _performance_monitor["baseline_sharpe"]
    if wr < b_wr * 0.8 or sharpe < b_sharpe * 0.5:
        _performance_monitor["status"] = "DEGRADED"
        last10 = recent[-10:] if len(recent) >= 10 else recent
        last10_wr = sum(1 for s in last10 if s["won"]) / len(last10) * 100
        if last10_wr < 30 and n >= 30:
            _performance_monitor["status"]          = "SHUTDOWN"
            _performance_monitor["shutdown_until"]  = time.time() + 3600
            _performance_monitor["shutdown_reason"] = f"WR={last10_wr:.0f}% últimos 10 trades"
            print(f"  🛑 AUTO-SHUTDOWN 1h: {_performance_monitor['shutdown_reason']}")
    elif wr < b_wr * 0.9 or pf < b_pf * 0.7:
        _performance_monitor["status"] = "WARNING"
    else:
        _performance_monitor["status"] = "OK"


def is_performance_ok():
    if _performance_monitor["status"] == "SHUTDOWN":
        if time.time() < _performance_monitor["shutdown_until"]:
            remaining_min = int((_performance_monitor["shutdown_until"] - time.time()) / 60)
            return False, f"🛑 AUTO-PAUSA ({remaining_min}min): {_performance_monitor['shutdown_reason']}"
    return True, ""


def register_signal_result(won, direction="", pnl_r=0.0, score=0):
    _perf_counters["daily_signals"] += 1
    if won:
        _perf_counters["cons_losses"] = 0
        _perf_counters["cons_wins"] += 1
    else:
        _perf_counters["cons_losses"] += 1
        _perf_counters["cons_wins"] = 0
    _signal_history.append({
        "ts": time.time(), "direction": direction,
        "won": won,
        "pnl_r": pnl_r if pnl_r else (1.5 if won else -1.0),
        "score": score,
    })
    if len(_signal_history) > 200:
        del _signal_history[:-200]
    update_performance_metrics()


def check_risk_limits():
    today = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    if today != _perf_counters["last_signal_date"]:
        _perf_counters["daily_signals"] = 0
        _perf_counters["last_signal_date"] = today
    perf_ok, perf_reason = is_performance_ok()
    if not perf_ok:
        return False, perf_reason
    if _perf_counters["cons_losses"] >= MAX_CONSECUTIVE_LOSSES:
        return False, f"MAX PÉRDIDAS CONSECUTIVAS ({MAX_CONSECUTIVE_LOSSES})"
    if _perf_counters["daily_signals"] >= MAX_DAILY_SIGNALS:
        return False, f"MAX SEÑALES DIARIAS ({MAX_DAILY_SIGNALS})"
    return True, ""


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
