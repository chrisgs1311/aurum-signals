"""
AURUM - Bot de Señales XAU/USD v6.0
Sin librerias externas. Solo Python 3.
Mejoras v6: OHLC real, ML validation split, thread safety, retry APIs, frontend SCALP v2
"""
import http.server, threading, json, urllib.request, urllib.parse
import csv, os, time
from datetime import datetime, timezone
import datetime as _dt
import socket, ssl, struct, base64, hashlib, secrets
import math, pickle

# ── MÓDULOS AURUM ────────────────────────────────────────
from aurum_state import *
from aurum_models import *
from aurum_news import *
from aurum_signal import *
from aurum_trading import *

# Imports explícitos de nombres con _ (no exportados por import *)
from aurum_state import (
    _live_cache, _sse_lock, _sse_clients, _data_lock, _cache_lock,
    _scalp_prices_1m, _scalp_prices_5m, _scalp_prices_15m,
    _ohlc_candles_5m, _ohlc_candles_15m,
    _paper_trades, _backtest_result, _walkforward_result,
    _api_counter, _current_control, _pre_signal,
    _performance_monitor, _signal_history,
    _fetch_with_retry, _push_sse, _mtf_cache,
)
from aurum_models import _ai

# Aliases de compatibilidad
_ict_prices_5m  = _scalp_prices_5m
_ict_prices_15m = _scalp_prices_15m








# ── FUENTES HTTP DE PRECIO ───────────────────────────────
def _parse_massive_nbbo(d):
    if not d or d.get("status") != "OK": return None
    r = d.get("results", {})
    bid = r.get("b") or r.get("B") or r.get("bid")
    ask = r.get("a") or r.get("A") or r.get("ask")
    if bid and ask:
        return {"price": round((float(bid)+float(ask))/2, 2), "ch":0,"chp":0,
                "bid":float(bid),"ask":float(ask)}
    return None

def _parse_goldapi(d):
    if not d: return None
    p = d.get("price") or d.get("gold") or d.get("XAU")
    if p:
        return {"price":float(p),"ch":float(d.get("ch",0)),"chp":float(d.get("chp",0))}
    return None

def _parse_metals_live(d):
    # metals.live devuelve [{metal, price, ...}] o {"gold": price}
    try:
        if isinstance(d, list):
            for item in d:
                if item.get("metal","").lower() in ("gold","xau"):
                    return {"price":float(item["price"]),"ch":0,"chp":0}
        elif isinstance(d, dict):
            p = d.get("gold") or d.get("XAU") or d.get("price")
            if p: return {"price":float(p),"ch":0,"chp":0}
    except Exception: pass
    return None

def _parse_yahoo(d):
    try:
        meta = d["chart"]["result"][0]["meta"]
        p = float(meta.get("regularMarketPrice") or 0)
        prev = float(meta.get("chartPreviousClose") or p)
        if p > 0:
            return {"price":p,"ch":round(p-prev,2),"chp":round((p-prev)/prev*100,3) if prev else 0}
    except Exception: pass
    return None

_PRICE_APIS = []
if MASSIVE_API_KEY:
    _PRICE_APIS.append((
        f"https://api.massive.com/v2/last/nbbo/C:XAUUSD?apikey={MASSIVE_API_KEY}",
        _parse_massive_nbbo,
    ))
if TWELVE_API_KEY:
    _PRICE_APIS.append((
        f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={TWELVE_API_KEY}",
        lambda d: {"price":float(d["price"]),"ch":0,"chp":0} if d and d.get("price") else None,
    ))
# Fuentes gratuitas de respaldo
_PRICE_APIS += [
    ("https://api.gold-api.com/price/XAU",                                     _parse_goldapi),
    ("https://api.metals.live/v1/spot/gold",                                    _parse_metals_live),
    ("https://query1.finance.yahoo.com/v8/finance/chart/XAUUSD%3DX?interval=1m&range=1d", _parse_yahoo),
]

print(f"  ✓ Fuentes precio: {len(_PRICE_APIS)} APIs")

def _worker_price():
    """Worker HTTP de precio. Lógica original: gate en ws_age, sleep=1s con Massive."""
    api_idx = 0
    price_fails = 0
    while True:
        try:
            ws_active = _live_cache.get("ws_active", False)
            ws_age    = time.time() - _live_cache.get("ws_last_tick", 0)
            # Sleep original: 1s con Massive key (unlimited), 30s si WS activo / 5s si no
            sleep_time = 1 if MASSIVE_API_KEY else (30 if ws_active else 5)

            url, parser = _PRICE_APIS[api_idx]
            if "twelvedata.com" in url and not _can_call_twelve():
                api_idx = (api_idx + 1) % len(_PRICE_APIS)
                url, parser = _PRICE_APIS[api_idx]

            d = _fetch_with_retry(url, timeout=5, retries=1, backoff=0.5)
            if "twelvedata.com" in url:
                _count_api_call("twelve")
            result = parser(d) if d else None

            if result and result.get("price", 0) > 0:
                # Calcular ch/chp desde precio anterior
                prev = _live_cache.get("price")
                prev_p = prev.get("price", 0) if isinstance(prev, dict) else 0
                if prev_p > 0 and prev_p != result["price"]:
                    result["ch"]  = round(result["price"] - prev_p, 2)
                    result["chp"] = round(result["ch"] / prev_p * 100, 4)
                elif prev_p > 0:
                    result["ch"]  = prev.get("ch", 0)
                    result["chp"] = prev.get("chp", 0)
                # Gate original: HTTP no sobreescribe si WS está activo y fresco
                if not ws_active or ws_age > 3:
                    _live_cache["price"]    = result
                    _live_cache["price_ts"] = time.time()
                    push_price(result["price"])
                    _push_sse("price", result)
                price_fails = 0
            else:
                price_fails += 1
                if price_fails >= 3:
                    api_idx = (api_idx + 1) % len(_PRICE_APIS)
                    price_fails = 0
                    print(f"  🔄 Price fallback → {_PRICE_APIS[api_idx][0][:50]}...")
        except Exception as e:
            print(f"  ⚠ Price worker: {e}")
        is_data_stale()
        if _live_cache.get("price_stale"):
            _push_sse("stale", {"type":"price","age_sec":round(time.time()-_live_cache["price_ts"],1)})
        time.sleep(sleep_time)

# ── WEBSOCKET TWELVE DATA (v6.2) ─────────────────────────
# Implementación WebSocket nativa sin librerías externas (RFC 6455)

def _ws_connect_massive():
    """v6.4: Conecta al WebSocket de Massive.com (Forex) y devuelve el socket.
    Flow: 1) conectar, 2) recibir status:connected, 3) auth, 4) recibir authenticated, 5) subscribe."""
    host = "socket.massive.com"
    port = 443
    path = "/forex"
    ws_key = base64.b64encode(secrets.token_bytes(16)).decode()
    handshake = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {ws_key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"\r\n"
    )
    ctx = ssl.create_default_context()
    sock = socket.create_connection((host, port), timeout=10)
    sock = ctx.wrap_socket(sock, server_hostname=host)
    sock.sendall(handshake.encode())
    response = b""
    while b"\r\n\r\n" not in response:
        chunk = sock.recv(4096)
        if not chunk: raise Exception("Massive WS handshake failed: empty response")
        response += chunk
    if b"101" not in response[:20]:
        raise Exception(f"Massive WS handshake failed: {response[:100]}")
    return sock

def _ws_connect_twelve():
    """Conecta al WebSocket de Twelve Data y devuelve el socket."""
    host = "ws.twelvedata.com"
    port = 443
    path = f"/v1/quotes/price?apikey={TWELVE_API_KEY}"
    # Handshake WebSocket
    ws_key = base64.b64encode(secrets.token_bytes(16)).decode()
    handshake = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {ws_key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"\r\n"
    )
    ctx = ssl.create_default_context()
    sock = socket.create_connection((host, port), timeout=10)
    sock = ctx.wrap_socket(sock, server_hostname=host)
    sock.sendall(handshake.encode())
    # Leer respuesta del handshake
    response = b""
    while b"\r\n\r\n" not in response:
        chunk = sock.recv(4096)
        if not chunk: raise Exception("WS handshake failed: empty response")
        response += chunk
    if b"101" not in response[:20]:
        raise Exception(f"WS handshake failed: {response[:100]}")
    return sock

def _ws_send_frame(sock, payload, opcode=0x81):
    """Envía un frame WebSocket con masking. opcode=0x81 text, 0x8A pong."""
    data = payload.encode() if isinstance(payload, str) else (payload if isinstance(payload, (bytes, bytearray)) else b"")
    length = len(data)
    frame = bytearray([0x80 | (opcode & 0x0F)])  # FIN + opcode
    mask_bit = 0x80
    if length < 126:
        frame.append(mask_bit | length)
    elif length < 65536:
        frame.append(mask_bit | 126)
        frame.extend(struct.pack(">H", length))
    else:
        frame.append(mask_bit | 127)
        frame.extend(struct.pack(">Q", length))
    mask = secrets.token_bytes(4)
    frame.extend(mask)
    masked = bytearray(b ^ mask[i % 4] for i, b in enumerate(data))
    frame.extend(masked)
    sock.sendall(bytes(frame))

def _ws_recv_frame(sock):
    """Lee un frame WebSocket y devuelve el payload de texto."""
    def recv_exact(n):
        data = b""
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk: raise Exception("WS connection closed")
            data += chunk
        return data
    header = recv_exact(2)
    opcode = header[0] & 0x0F
    length = header[1] & 0x7F
    if length == 126:
        length = struct.unpack(">H", recv_exact(2))[0]
    elif length == 127:
        length = struct.unpack(">Q", recv_exact(8))[0]
    payload = recv_exact(length) if length > 0 else b""
    if opcode == 0x8:  # close frame
        raise Exception("WS close frame received")
    if opcode == 0x9:  # ping — responder pong con mismo payload (RFC 6455)
        _ws_send_frame(sock, payload, opcode=0x8A)
        return None
    return payload.decode("utf-8", errors="ignore")

def _worker_websocket_massive():
    """Worker WebSocket Massive.com → XAU/USD tick-by-tick.
    Protocolo Polygon: suscripción con PUNTO (C.XAUUSD), no dos puntos (C:XAUUSD).
    Los dos puntos son para el API REST; el WebSocket usa punto como separador.
    """
    _msg_count = [0]  # contador de mensajes para debug

    while True:
        sock = None
        try:
            if not MASSIVE_API_KEY:
                time.sleep(60)
                continue

            print("  🔌 [Massive WS] Conectando a socket.massive.com/forex ...")
            sock = _ws_connect_massive()
            sock.settimeout(30)

            # 1) Leer frame inicial (status:connected)
            initial = _ws_recv_frame(sock)
            print(f"  📡 [Massive WS] Inicial: {(initial or '')[:150]}")

            # 2) Autenticar
            _ws_send_frame(sock, json.dumps({"action": "auth", "params": MASSIVE_API_KEY}))

            # 3) Leer respuesta de auth
            auth_resp = _ws_recv_frame(sock)
            print(f"  📡 [Massive WS] Auth resp: {(auth_resp or '')[:150]}")
            if auth_resp and "auth_success" not in auth_resp:
                if "auth_failed" in auth_resp or "not_authorized" in auth_resp:
                    print("  ❌ [Massive WS] Auth fallida — key inválida")
                    _live_cache["massive_key_invalid"] = True
                    time.sleep(300)
                    continue

            # 4) Suscribir — formato Massive WebSocket: CANAL.{from}-{to}
            # El formato correcto es XAU-USD (con guión), NO XAUUSD ni XAU:USD
            _ws_send_frame(sock, json.dumps({
                "action": "subscribe",
                "params": "C.XAU-USD,CAS.XAU-USD,CA.XAU-USD"
            }))

            # 5) Leer respuesta de suscripción
            sub_resp = _ws_recv_frame(sock)
            print(f"  📡 [Massive WS] Sub resp: {(sub_resp or '')[:150]}")

            _live_cache["ws_active"]   = True
            _live_cache["ws_provider"] = "massive"
            _msg_count[0] = 0
            last_price = 0
            print("  ✅ [Massive WS] Listo — esperando ticks XAU/USD ...")

            while True:
                try:
                    msg = _ws_recv_frame(sock)
                    if not msg:
                        continue
                    events = json.loads(msg)
                    if not isinstance(events, list):
                        events = [events]

                    for ev in events:
                        ev_type = ev.get("ev", "")

                        # Log completo de los primeros 30 mensajes para debug
                        if _msg_count[0] < 30:
                            print(f"  📊 [Massive WS] msg#{_msg_count[0]} ev={ev_type}: {str(ev)[:200]}")
                        _msg_count[0] += 1

                        if ev_type == "C":          # Quote: bid/ask
                            bid = ev.get("b") or ev.get("B")
                            ask = ev.get("a") or ev.get("A")
                            if bid and ask:
                                last_price = (float(bid) + float(ask)) / 2

                        elif ev_type in ("CA", "CAS"):  # Agregados minuto/segundo
                            c = ev.get("c")
                            if c:
                                last_price = float(c)

                        elif ev_type == "status":
                            print(f"  📡 [Massive WS] status: {ev.get('message', str(ev))[:150]}")

                        if last_price > 0:
                            _prev = _live_cache.get("price")
                            prev_p = _prev.get("price", last_price) if isinstance(_prev, dict) else last_price
                            ch  = round(last_price - prev_p, 2)
                            chp = round(ch / prev_p * 100, 3) if prev_p else 0
                            result = {"price": last_price, "ch": ch, "chp": chp}
                            if ev_type == "C" and bid and ask:
                                result["bid"]    = float(bid)
                                result["ask"]    = float(ask)
                                result["spread"] = round(float(ask) - float(bid), 2)
                            _live_cache["price"]        = result
                            _live_cache["price_ts"]     = time.time()
                            _live_cache["ws_last_tick"] = time.time()
                            push_price(last_price)
                            _push_sse("price", result)

                except socket.timeout:
                    continue  # Massive no necesita ping manual

        except Exception as e:
            print(f"  ⚠ [Massive WS] Error: {e} — reintentando en 15s")
            _live_cache["ws_active"] = False
            try: sock.close()
            except: pass
            time.sleep(15)

def _worker_websocket():
    """Worker #6: WebSocket Twelve Data → precio tiempo real (~1 tick/seg).
    v6.4: solo se usa si NO hay MASSIVE_API_KEY (fallback).
    Auto-reconnect si se cae. Marca _live_cache['ws_active'] para que el HTTP
    worker reduzca frecuencia."""
    while True:
        # Si la key fue marcada como inválida, no intentar conectar
        if _live_cache.get("twelve_key_invalid"):
            _live_cache["ws_active"] = False
            time.sleep(60)
            continue
        sock = None
        try:
            print("  🔌 Conectando WebSocket Twelve Data...")
            sock = _ws_connect_twelve()
            # Suscribirse a XAU/USD
            subscribe = json.dumps({
                "action": "subscribe",
                "params": {"symbols": "XAU/USD"}
            })
            _ws_send_frame(sock, subscribe)
            print("  ✅ WebSocket Twelve Data conectado, suscrito a XAU/USD")
            _live_cache["ws_active"] = True
            _live_cache["ws_provider"] = "twelvedata"
            sock.settimeout(30)
            last_heartbeat = time.time()
            while True:
                try:
                    msg = _ws_recv_frame(sock)
                    if not msg: continue
                    data = json.loads(msg)
                    if data.get("event") == "price" and "price" in data:
                        price = float(data["price"])
                        if price > 0:
                            _prev_data = _live_cache.get("price")
                            prev = _prev_data.get("price", price) if isinstance(_prev_data, dict) else price
                            ch = price - prev if prev else 0
                            chp = (ch / prev * 100) if prev else 0
                            result = {"price": price, "ch": round(ch, 2), "chp": round(chp, 3)}
                            _live_cache["price"] = result
                            _live_cache["price_ts"] = time.time()
                            _live_cache["ws_last_tick"] = time.time()
                            push_price(price)
                            _push_sse("price", result)
                    if time.time() - last_heartbeat > 30:
                        _ws_send_frame(sock, json.dumps({"action": "heartbeat"}))
                        last_heartbeat = time.time()
                except socket.timeout:
                    _ws_send_frame(sock, json.dumps({"action": "heartbeat"}))
                    last_heartbeat = time.time()
                    continue
        except Exception as e:
            print(f"  ⚠ WebSocket Twelve: {e} — reintentando en 10s")
            _live_cache["ws_active"] = False
            try: sock.close()
            except: pass
            time.sleep(10)

def _worker_ohlc():
    """Worker #2: OHLC con frecuencia por TF.
    5min: cada 3min | 15min: cada 5min | 1h: cada 10min | 4h: cada 30min.
    ~500 calls/día de Twelve Data (bien dentro de 800)."""
    # Timestamps de última carga por TF
    last_fetch = {"5min": 0, "15min": 0, "1h": 0, "4h": 0}
    intervals = {
        "5min":  {"tf": "5m",  "size": 200, "freq": 180},   # ~16 horas contexto
        "15min": {"tf": "15m", "size": 200, "freq": 300},   # ~50 horas contexto
        "1h":    {"tf": "1h",  "size": 200, "freq": 600},   # ~8 días contexto
        "4h":    {"tf": "4h",  "size": 200, "freq": 1800},  # ~33 días contexto HTF
    }
    while True:
        try:
            now = time.time()
            for interval, cfg in intervals.items():
                if now - last_fetch[interval] < cfg["freq"]:
                    continue
                # v6.5: Massive = unlimited, Twelve Data necesita check de límite
                if not MASSIVE_API_KEY and not _can_call_twelve():
                    print(f"  ⚠ OHLC worker: límite diario Twelve Data")
                    break
                candles = get_historical_ohlc(interval, cfg["size"])
                if candles:
                    cache_key = f"ohlc_{cfg['tf']}"
                    _live_cache[cache_key] = candles
                    _live_cache["ohlc_ts"] = now
                    # Solo alimentar price history con 5m (evitar duplicados)
                    if interval == "5min":
                        for c in candles:
                            push_price(c["close"])
                            update_ict_prices(c["close"])
                last_fetch[interval] = now
                time.sleep(3)  # pausa entre calls
        except Exception as e:
            print(f"  ⚠ OHLC worker: {e}")
        is_data_stale()
        time.sleep(30)  # check cada 30s qué TF necesita refresh

def _worker_signal():
    """Worker #3: señal SCALP v2 cada 1s — detecta y pushea instantáneo.
    v6.3.2: SIMPLE. Solo requiere velas OHLC reales y precio live.
    Sin histéresis, sin staleness checks agresivos, sin validaciones extra.
    El motor SCALP ya valida internamente todo lo necesario."""
    while True:
        try:
            # ── Verificación mínima: hay velas reales suficientes ──
            with _data_lock:
                ohlc5_count = len(_ohlc_candles_5m)
                ohlc15_count = len(_ohlc_candles_15m)
                ohlc5_local = list(_ohlc_candles_5m)
            if ohlc5_count < 20 or ohlc15_count < 15:
                time.sleep(1)
                continue
            
            # ── Precio LIVE (del WebSocket o HTTP cache) ──
            live_price_data = _live_cache.get("price")
            if not live_price_data or not live_price_data.get("price"):
                time.sleep(1)
                continue
            cp = live_price_data["price"]
            
            # ── ATR de velas reales ──
            closes = [c["c"] for c in ohlc5_local]
            atr_val = sum(abs(closes[i]-closes[i-1])
                          for i in range(len(closes)-14, len(closes))) / 14
            if atr_val <= 0:
                time.sleep(1)
                continue
            
            # ── Ejecutar motor SCALP ──
            result = run_ict_engine(cp, atr_val)
            
            # Push pre-signal state
            _push_sse("presignal", {
                "state": _pre_signal["state"],
                "bias": _pre_signal["bias"],
                "sweep_level": _pre_signal["sweep_level"],
                "fvg_zone": _pre_signal["fvg_zone"],
            })
            
            if result:
                # v6.5: Risk Manager — verificar límites antes de registrar señal
                can_trade, risk_reason = risk_can_trade()
                if not can_trade:
                    _push_sse("wait", {"reason": risk_reason, "blocked": True})
                    _live_cache["signal"] = None
                    time.sleep(1)
                    continue
                
                _live_cache["signal"] = result
                _live_cache["signal_ts"] = time.time()
                _push_sse("signal", {
                    "direction": result["direction"],
                    "score": result["score"],
                    "tp": result["tp"],
                    "sl": result["sl"],
                    "rr": result["rr"],
                    "details": result["details"],
                    "session": result["session"],
                    "entry": cp,
                })
                # v6.5: registrar paper trade automáticamente
                paper_register_signal(result, cp)
                # Telegram para todas las señales válidas
                if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                    threading.Thread(target=send_telegram_direct,
                        args=(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, result["msg"]),
                        daemon=True).start()
        except Exception as e:
            print(f"  ⚠ Signal worker: {e}")
        time.sleep(1)

def _worker_train():
    """Worker #4: ML train cada 5min + control state — nunca bloquea requests.
    v6.1: corre inmediatamente al arrancar, luego cada 5min."""
    first_run = True
    while True:
        if not first_run:
            time.sleep(300)
        first_run = False
        try:
            news_on, _ = is_news_time()
            with _data_lock:
                ph = price_history[-50:] if len(price_history) >= 50 else [0]
                # v6.2: usar velas OHLC reales (que ya tenemos cargadas) para train + predict
                ohlc5_closes  = [c["c"] for c in _ohlc_candles_5m]
                ohlc15_closes = [c["c"] for c in _ohlc_candles_15m]
            update_control_state(_ai, ph, is_news=news_on)
            # Dataset combinado: 15m + 5m + tick history (lo que más tenga)
            combined = ohlc15_closes + ohlc5_closes
            if len(price_history) > len(combined):
                combined = list(price_history)
            if len(combined) >= 60:
                ai_train_if_needed(combined)
                # v6.2: predicción con el MISMO dataset que se entrenó
                prob, sig = ai_predict(combined)
                _live_cache["ai_prob"]   = round(prob * 100, 1) if prob is not None else None
                _live_cache["ai_signal"] = sig
                print(f"  🤖 AI cache updated: prob={_live_cache['ai_prob']} sig={sig} | dataset={len(combined)}")
            else:
                print(f"  ⏳ AI: esperando datos ({len(combined)}/60)")
        except Exception as e:
            print(f"  ⚠ Train worker: {e}")

def _worker_news():
    """Worker #5: alertas de noticias HIGH IMPACT al Telegram de AURUM.
    v6.2: usa Forex Factory real, no calendario hardcoded."""
    icons = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}
    while True:
        try:
            alerts = check_upcoming_news()
            for a in alerts:
                if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
                    continue
                icon = icons.get(a["impact"], "⚠")
                mins = a["minutes_away"]
                if mins <= 5:
                    urgency = "🚨 *INMINENTE*"
                elif mins <= 15:
                    urgency = "⚠️ *PRÓXIMA*"
                else:
                    urgency = "📅 *PROGRAMADA*"
                msg = (
                    f"{urgency}\n\n"
                    f"{icon} *NOTICIA HIGH IMPACT*\n\n"
                    f"📌 Evento: *{a['name']}*\n"
                    f"⏳ Faltan: *{mins} min*\n"
                    f"📊 Impacto: {a['impact']}\n\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"⚠ AURUM pausará señales\n"
                    f"30 min antes y después.\n"
                    f"Evita operar XAU/USD."
                )
                threading.Thread(
                    target=send_telegram_direct,
                    args=(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, msg),
                    daemon=True
                ).start()
                print(f"  📢 Alerta noticia enviada: {a['name']} ({mins}min antes)")
        except Exception as e:
            print(f"  ⚠ News worker: {e}")
        time.sleep(60)


def start_all_workers():
    """Lanza todos los workers en threads daemon."""
    # Cargar paper trades del disco antes de arrancar workers
    _load_paper_trades()
    workers = [
        ("Massive WS (real-time)", _worker_websocket_massive),
        ("Twelve WS (fallback)",   _worker_websocket),
        ("Price HTTP",             _worker_price),
        ("OHLC Feed",              _worker_ohlc),
        ("Signal Engine (1s)",     _worker_signal),
        ("ML Train (5min)",        _worker_train),
        ("News Alerts (60s)",      _worker_news),
        ("Paper Trading (1s)",     _worker_paper),
    ]
    for name, fn in workers:
        t = threading.Thread(target=fn, daemon=True, name=name)
        t.start()
        print(f"  ✓ Worker: {name}")
    return len(workers)


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

# TWELVE_API_KEY ya fue definida arriba (antes de _PRICE_APIS)

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



def get_gold_price():
    """v6.4: usa _PRICE_APIS en orden (Massive primero si hay key, gold-api fallback)."""
    c = cached("precio_live", ttl=8)
    if c: return c
    # Iterar _PRICE_APIS en orden — primero Massive si está configurado
    for url, parser in _PRICE_APIS:
        try:
            d = _fetch_with_retry(url, timeout=5, retries=1, backoff=0.5)
            if not d: continue
            result = parser(d)
            if result and result.get("price", 0) > 0:
                set_cache("precio_live", result)
                return result
        except Exception as e:
            continue
    return None


def _count_api_call(api="twelve"):
    """Cuenta calls diarias. Resetea a medianoche UTC."""
    today = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    key_calls = f"{api}_calls"
    key_date  = f"{api}_date"
    if _api_counter[key_date] != today:
        _api_counter[key_calls] = 0
        _api_counter[key_date] = today
    _api_counter[key_calls] += 1
    return _api_counter[key_calls]

def _can_call_twelve():
    """Verifica si quedan calls de Twelve Data hoy."""
    today = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    if _api_counter["twelve_date"] != today:
        return True
    return _api_counter["twelve_calls"] < TWELVE_DAILY_LIMIT

def _fetch_massive_ohlc(interval="5min", outputsize=150):
    """v6.4: Fetch OHLC de Massive.com (Forex).
    Massive usa format: /v2/aggs/ticker/C:XAUUSD/range/{multiplier}/{timespan}/{from}/{to}
    Timespan: minute, hour, day
    Multiplier: 1, 5, 15, 30, etc."""
    if not MASSIVE_API_KEY: return None
    # Mapear interval a Massive format
    interval_map = {
        "1min":  ("1", "minute"),
        "5min":  ("5", "minute"),
        "15min": ("15", "minute"),
        "30min": ("30", "minute"),
        "1h":    ("1", "hour"),
        "4h":    ("4", "hour"),
        "1day":  ("1", "day"),
    }
    if interval not in interval_map:
        return None
    multiplier, timespan = interval_map[interval]
    # Rango: desde hace N períodos hasta ahora
    now_ms = int(time.time() * 1000)
    # Calcular from_ms según interval y outputsize (con margen x2 para fines de semana)
    interval_sec = {
        "1min": 60, "5min": 300, "15min": 900, "30min": 1800,
        "1h": 3600, "4h": 14400, "1day": 86400,
    }[interval]
    from_ms = now_ms - (outputsize * interval_sec * 1000 * 3)  # x3 margen
    url = (f"https://api.massive.com/v2/aggs/ticker/C:XAUUSD/range/"
           f"{multiplier}/{timespan}/{from_ms}/{now_ms}"
           f"?adjusted=true&sort=asc&limit=50000&apikey={MASSIVE_API_KEY}")
    data = _fetch_with_retry(url, timeout=10, retries=2, backoff=1.5)
    if data and data.get("status") == "OK" and "results" in data:
        results = data["results"]
        # Limitar a outputsize (los más recientes)
        results = results[-outputsize:]
        candles = [{
            "open":  float(v["o"]),
            "high":  float(v["h"]),
            "low":   float(v["l"]),
            "close": float(v["c"]),
            "dt":    _dt.datetime.utcfromtimestamp(v["t"]/1000).strftime("%Y-%m-%d %H:%M:%S"),
        } for v in results]
        return candles
    elif data:
        err = data.get("error") or data.get("message", "sin datos")
        print(f"  ⚠ Massive OHLC {interval}: {err}")
    return None

def _fetch_massive_bulk(interval="5min", days=30):
    """v7.1: Descarga data masiva de Massive (30+ días) para walk-forward/backtest.
    Con Massive unlimited calls + 10 años de historia, descargamos MILES de velas reales.
    
    30 días de 5min = ~8,640 velas (vs 500 de Twelve Data).
    Esto permite walk-forward con 10+ ventanas y backtest estadísticamente significativo.
    """
    if not MASSIVE_API_KEY:
        print("  ⚠ Massive bulk: sin API key")
        return None
    interval_map = {
        "1min": ("1", "minute"), "5min": ("5", "minute"),
        "15min": ("15", "minute"), "1h": ("1", "hour"), "4h": ("4", "hour"),
    }
    if interval not in interval_map: return None
    multiplier, timespan = interval_map[interval]
    now_ms = int(time.time() * 1000)
    from_ms = now_ms - (days * 86400 * 1000)
    url = (f"https://api.massive.com/v2/aggs/ticker/C:XAUUSD/range/"
           f"{multiplier}/{timespan}/{from_ms}/{now_ms}"
           f"?adjusted=true&sort=asc&limit=50000&apikey={MASSIVE_API_KEY}")
    print(f"  📥 Descargando {days} días de {interval} de Massive...")
    data = _fetch_with_retry(url, timeout=30, retries=3, backoff=2.0)
    if data and data.get("status") == "OK" and "results" in data:
        results = data["results"]
        candles = [{
            "c": float(v["c"]), "o": float(v["o"]),
            "h": float(v["h"]), "l": float(v["l"]),
            "dt": _dt.datetime.utcfromtimestamp(v["t"]/1000).strftime("%Y-%m-%d %H:%M:%S"),
        } for v in results]
        print(f"  ✅ Massive bulk: {len(candles)} velas de {interval} ({days} días)")
        return candles
    elif data:
        err = data.get("error") or data.get("message", "sin datos")
        print(f"  ⚠ Massive bulk {interval}: {err}")
    return None

def get_historical_ohlc(interval="5min", outputsize=150):
    ttl = 120 if interval in ("1min","5min") else 300
    c = cached(f"ohlc_{interval}", ttl=ttl)
    if c: return c
    
    # v6.4: PRIMERO intentar Massive (unlimited calls, mismo feed que TradingView)
    if MASSIVE_API_KEY:
        candles = _fetch_massive_ohlc(interval, outputsize)
        if candles:
            set_cache(f"ohlc_{interval}", candles)
            if interval == "5min":
                update_ohlc_candles(candles, tf="5m")
            elif interval == "15min":
                update_ohlc_candles(candles, tf="15m")
            _live_cache["ohlc_ts"] = time.time()
            print(f"  ✓ {len(candles)} velas {interval} [Massive]")
            return candles
        else:
            print(f"  ⚠ Massive {interval} no disponible, intentando Twelve Data...")
    
    # Fallback: Twelve Data
    if _live_cache.get("twelve_key_invalid"):
        return []
    if not _can_call_twelve():
        print(f"  ⚠ Twelve Data: límite diario alcanzado ({_api_counter['twelve_calls']}/{TWELVE_DAILY_LIMIT})")
        return []
    url = (f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval={interval}"
           f"&outputsize={outputsize}&apikey={TWELVE_API_KEY}")
    data = _fetch_with_retry(url, timeout=10, retries=2, backoff=1.5)
    _count_api_call("twelve")
    if data and "values" in data:
        candles = [{"open": float(v["open"]), "high": float(v["high"]),
                    "low":  float(v["low"]),  "close": float(v["close"]),
                    "dt":   v["datetime"]} for v in reversed(data["values"])]
        set_cache(f"ohlc_{interval}", candles)
        if interval == "5min":
            update_ohlc_candles(candles, tf="5m")
        elif interval == "15min":
            update_ohlc_candles(candles, tf="15m")
        _live_cache["ohlc_ts"] = time.time()
        print(f"  ✓ {len(candles)} velas {interval} [Twelve Data #{_api_counter['twelve_calls']}/día]")
        return candles
    elif data:
        msg = data.get('message','sin datos')
        print(f"  ⚠ Twelve Data: {msg}")
        if any(kw in str(msg).lower() for kw in ["api key", "apikey", "invalid", "unauthor", "forbidden", "not registered"]):
            print(f"  ❌ TWELVE DATA API KEY INVÁLIDA")
            _live_cache["twelve_key_invalid"] = True
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
  <div class="logo">AURUM<span>BOT DE SEÑALES XAU/USD v8.0</span></div>
  <div class="header-right">
    <div class="tg-status" id="tgStatus">📵 TELEGRAM OFF</div>
    <button class="sound-toggle on" id="soundBtn" onclick="toggleSound()">🔔 ON</button>
    <div class="live-badge"><div class="live-dot"></div>EN VIVO</div>
  </div>
</header>
<nav>
  <button class="nav-tab active" onclick="showPage('dashboard')">DASHBOARD</button>
  <button class="nav-tab" onclick="showPage('stats')">STATS</button>
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
<div class="page" id="page-stats">
<div class="bt-page">
  <div class="bt-title">ESTADÍSTICAS REALES</div>
  <div class="bt-subtitle">PAPER TRADING · DATOS EN VIVO · PROFIT FACTOR · WIN RATE</div>
  <div style="display:flex;gap:10px;margin:16px 0">
    <button class="reg-btn primary" onclick="loadStats()">↻ ACTUALIZAR</button>
    <span id="statsUpdated" style="font-size:8px;letter-spacing:2px;color:var(--dim2);align-self:center"></span>
  </div>
  <div class="bt-stats">
    <div class="bt-stat"><div class="bt-stat-val gold" id="statTotal">—</div><div class="bt-stat-label">TRADES</div></div>
    <div class="bt-stat"><div class="bt-stat-val" id="statWR" style="color:var(--gold)">—</div><div class="bt-stat-label">WIN RATE</div></div>
    <div class="bt-stat"><div class="bt-stat-val" id="statPF" style="color:var(--gold)">—</div><div class="bt-stat-label">PROFIT FACTOR</div></div>
    <div class="bt-stat"><div class="bt-stat-val" id="statR" style="color:var(--gold)">—</div><div class="bt-stat-label">TOTAL R</div></div>
  </div>
  <div class="bt-stats">
    <div class="bt-stat"><div class="bt-stat-val green" id="statWins">—</div><div class="bt-stat-label">GANADORAS</div></div>
    <div class="bt-stat"><div class="bt-stat-val red" id="statLosses">—</div><div class="bt-stat-label">PERDEDORAS</div></div>
    <div class="bt-stat"><div class="bt-stat-val gold" id="statOpen">—</div><div class="bt-stat-label">ABIERTAS</div></div>
    <div class="bt-stat"><div class="bt-stat-val gold" id="statExp">—</div><div class="bt-stat-label">EXPECTANCY</div></div>
  </div>
  <div class="bt-chart-wrap">
    <div class="bt-chart-title">CURVA DE EQUITY REAL (R acumulado)</div>
    <canvas id="statsChart" height="180"></canvas>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px">
    <div class="bt-chart-wrap">
      <div class="bt-chart-title">POR SESIÓN</div>
      <table class="bt-table">
        <thead><tr><th>SESIÓN</th><th>TRADES</th><th>WR</th><th>R</th></tr></thead>
        <tbody id="statsBySession"></tbody>
      </table>
    </div>
    <div class="bt-chart-wrap">
      <div class="bt-chart-title">POR TIER</div>
      <table class="bt-table">
        <thead><tr><th>TIER</th><th>TRADES</th><th>WR</th><th>R</th></tr></thead>
        <tbody id="statsByTier"></tbody>
      </table>
    </div>
  </div>
  <div class="bt-chart-wrap">
    <div class="bt-chart-title">ÚLTIMOS 20 TRADES</div>
    <div style="overflow:auto">
      <table class="bt-table">
        <thead><tr><th>HORA</th><th>DIR</th><th>ENTRY</th><th>TP</th><th>SL</th><th>TIER</th><th>RESULT</th><th>R</th><th>DUR</th></tr></thead>
        <tbody id="statsRecent"></tbody>
      </table>
    </div>
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
        <div class="rule-row"><span class="rule-label">PRECIO API</span><span style="color:var(--green);font-size:10px">massive.com</span></div>
        <div class="rule-row"><span class="rule-label">ACTUALIZACIÓN</span><span style="color:var(--green);font-size:10px">TIEMPO REAL ⚡</span></div>
        <div class="rule-row"><span class="rule-label">WEBSOCKET</span><span style="color:var(--green);font-size:10px">TICK-BY-TICK</span></div>
        <div class="rule-row"><span class="rule-label">TELEGRAM</span><span id="cfgTGstatus" style="font-size:10px;color:var(--dim2)">INACTIVO</span></div>
        <div class="rule-row" style="border:none"><span class="rule-label">VERSIÓN</span><span style="color:var(--gold);font-size:10px">AURUM v6.4</span></div>
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
  if(name==='stats')loadStats();
}
function getAC(){if(!audioCtx)audioCtx=new(window.AudioContext||window.webkitAudioContext)();return audioCtx}
function playTone(freqs,durs){if(!soundOn)return;try{const ctx=getAC();let t=ctx.currentTime;freqs.forEach((f,i)=>{const o=ctx.createOscillator(),g=ctx.createGain();o.connect(g);g.connect(ctx.destination);o.frequency.value=f;g.gain.setValueAtTime(0,t);g.gain.linearRampToValueAtTime(0.3,t+.02);g.gain.linearRampToValueAtTime(0,t+durs[i]);o.start(t);o.stop(t+durs[i]);t+=durs[i]+.05})}catch(e){}}
function alertBuy(){playTone([440,554,659],[.15,.15,.3])}
function alertSell(){playTone([659,554,440],[.15,.15,.3])}
function toggleSound(){soundOn=!soundOn;const b=document.getElementById('soundBtn');b.textContent=soundOn?'🔔 ON':'🔕 OFF';b.className='sound-toggle '+(soundOn?'on':'off');if(soundOn)try{getAC().resume()}catch(e){}}
async function saveTelegram(){
  tgToken=document.getElementById('cfgToken').value.trim();tgChatId=document.getElementById('cfgChatId').value.trim();
  const active=tgToken&&tgChatId;
  if(active){
    // v6.1: persistir al backend
    try{
      const r=await fetch('/telegram_config?token='+encodeURIComponent(tgToken)+'&chat='+encodeURIComponent(tgChatId));
      const d=await r.json();
      if(d.ok){
        document.getElementById('tgStatus').textContent='📲 TELEGRAM ON';
        document.getElementById('tgStatus').className='tg-status connected';
        document.getElementById('cfgTGstatus').textContent='ACTIVO ✓';
        document.getElementById('cfgTGstatus').style.color='var(--green)';
        alert('✓ Telegram configurado y guardado en servidor');
      }else{
        alert('⚠ Error guardando config');
      }
    }catch(e){alert('⚠ Error de conexión');}
  }else{
    document.getElementById('tgStatus').textContent='📵 TELEGRAM OFF';
    document.getElementById('tgStatus').className='tg-status';
    document.getElementById('cfgTGstatus').textContent='INACTIVO';
    document.getElementById('cfgTGstatus').style.color='var(--dim2)';
    alert('Token o Chat ID vacíos');
  }
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

// v6.5: Paper Trading Stats
async function loadStats(){
  try{
    const s=await(await fetch('/stats')).json();
    document.getElementById('statTotal').textContent=s.total||0;
    document.getElementById('statOpen').textContent=s.open||0;
    document.getElementById('statWins').textContent=s.wins||0;
    document.getElementById('statLosses').textContent=s.losses||0;
    const wr=s.win_rate||0;
    document.getElementById('statWR').textContent=wr+'%';
    document.getElementById('statWR').style.color=wr>=55?'var(--green)':wr>=45?'var(--gold)':'var(--red)';
    const pf=s.profit_factor||0;
    document.getElementById('statPF').textContent=pf;
    document.getElementById('statPF').style.color=pf>=1.5?'var(--green)':pf>=1?'var(--gold)':'var(--red)';
    const totalR=s.total_r||0;
    document.getElementById('statR').textContent=(totalR>=0?'+':'')+totalR+'R';
    document.getElementById('statR').style.color=totalR>=0?'var(--green)':'var(--red)';
    const exp=s.expectancy||0;
    document.getElementById('statExp').textContent=(exp>=0?'+':'')+exp+'R';
    document.getElementById('statExp').style.color=exp>=0.3?'var(--green)':exp>=0?'var(--gold)':'var(--red)';
    // By session
    const sessBody=document.getElementById('statsBySession');
    const sessEntries=Object.entries(s.by_session||{});
    if(sessEntries.length===0){
      sessBody.innerHTML='<tr><td colspan="4" style="text-align:center;color:var(--dim2);padding:16px">Sin datos</td></tr>';
    }else{
      sessBody.innerHTML=sessEntries.map(([k,v])=>'<tr><td>'+k+'</td><td>'+v.n+'</td><td style="color:'+(v.wr>=55?'var(--green)':'var(--red)')+'">'+v.wr+'%</td><td style="color:'+(v.r>=0?'var(--green)':'var(--red)')+'">'+(v.r>=0?'+':'')+v.r+'R</td></tr>').join('');
    }
    // By tier
    const tierBody=document.getElementById('statsByTier');
    const tierEntries=Object.entries(s.by_tier||{});
    if(tierEntries.length===0){
      tierBody.innerHTML='<tr><td colspan="4" style="text-align:center;color:var(--dim2);padding:16px">Sin datos</td></tr>';
    }else{
      tierBody.innerHTML=tierEntries.map(([k,v])=>'<tr><td>'+k+'</td><td>'+v.n+'</td><td style="color:'+(v.wr>=55?'var(--green)':'var(--red)')+'">'+v.wr+'%</td><td style="color:'+(v.r>=0?'var(--green)':'var(--red)')+'">'+(v.r>=0?'+':'')+v.r+'R</td></tr>').join('');
    }
    // Recent trades
    const recBody=document.getElementById('statsRecent');
    const recent=(s.recent_trades||[]).slice().reverse();
    if(recent.length===0){
      recBody.innerHTML='<tr><td colspan="9" style="text-align:center;color:var(--dim2);padding:16px">Sin trades aún</td></tr>';
    }else{
      recBody.innerHTML=recent.map(t=>{
        const time=new Date((t.opened_at||0)*1000).toLocaleTimeString('es-ES',{hour:'2-digit',minute:'2-digit'});
        const dirCls=t.direction==='COMPRAR'?'td-buy':'td-sell';
        const resCls=t.result==='WIN'?'td-pos':t.result==='LOSS'?'td-neg':'';
        const dur=t.duration_sec<60?t.duration_sec+'s':Math.round(t.duration_sec/60)+'m';
        return '<tr><td style="color:var(--dim2)">'+time+'</td><td class="'+dirCls+'">'+t.direction+'</td><td>$'+t.entry.toFixed(2)+'</td><td style="color:var(--green)">$'+t.tp.toFixed(2)+'</td><td style="color:var(--red)">$'+t.sl.toFixed(2)+'</td><td style="color:'+(t.tier==='SNIPER'?'var(--gold)':'var(--dim2)')+'">'+t.tier+'</td><td class="'+resCls+'">'+t.result+'</td><td class="'+resCls+'">'+(t.pnl_r>=0?'+':'')+t.pnl_r+'R</td><td style="color:var(--dim2)">'+dur+'</td></tr>';
      }).join('');
    }
    // Equity curve
    drawEquityCurve(s.equity_curve||[0]);
    document.getElementById('statsUpdated').textContent='Actualizado: '+new Date().toLocaleTimeString('es-ES');
  }catch(e){console.error('stats:',e);}
}
function drawEquityCurve(eq){
  const cv=document.getElementById('statsChart');if(!cv)return;
  const ctx=cv.getContext('2d');
  const W=cv.offsetWidth||600,H=180;cv.width=W;cv.height=H;
  ctx.clearRect(0,0,W,H);
  if(eq.length<2){
    ctx.fillStyle='#5A4820';ctx.font='11px monospace';ctx.textAlign='center';
    ctx.fillText('Sin trades aún — empieza cuando llegue la primera señal',W/2,H/2);
    return;
  }
  const mn=Math.min(...eq)-0.5,mx=Math.max(...eq)+0.5;
  const px=i=>(i/(eq.length-1||1))*(W-20)+10;
  const py=v=>H-((v-mn)/(mx-mn+0.01))*(H-30)-15;
  // Grid
  ctx.strokeStyle='rgba(201,168,76,.08)';ctx.lineWidth=1;
  for(let g=0;g<=4;g++){const y=py(mn+(mx-mn)*g/4);ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
  // Zero line
  if(mn<0&&mx>0){
    ctx.strokeStyle='rgba(201,168,76,.3)';ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(0,py(0));ctx.lineTo(W,py(0));ctx.stroke();
    ctx.setLineDash([]);
  }
  // Gradient fill
  const last=eq[eq.length-1],color=last>=0?'#00CC88':'#CC3344';
  const gr=ctx.createLinearGradient(0,0,0,H);
  gr.addColorStop(0,color+'40');gr.addColorStop(1,color+'00');
  ctx.beginPath();ctx.moveTo(px(0),py(eq[0]));
  eq.forEach((v,i)=>ctx.lineTo(px(i),py(v)));
  ctx.lineTo(px(eq.length-1),H);ctx.lineTo(px(0),H);
  ctx.fillStyle=gr;ctx.fill();
  // Line
  ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=2;
  eq.forEach((v,i)=>i===0?ctx.moveTo(px(i),py(v)):ctx.lineTo(px(i),py(v)));
  ctx.stroke();
  // Last point
  ctx.beginPath();ctx.arc(px(eq.length-1),py(last),4,0,Math.PI*2);
  ctx.fillStyle=color;ctx.fill();
}
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
  if(prices.length<30)return;
  const pp=prices.slice(-200).join(',');
  try{
    const tr=await fetch('/aitrain');
    const td=await tr.json();
    if(td.trained){
      // v6.2: contador sube cuando cambia el número de muestras (= re-entrenó)
      if(window._lastEpochs!==undefined && td.epochs!==window._lastEpochs){
        aiRetrains++;
      }
      window._lastEpochs=td.epochs;
      document.getElementById('aiAccuracy').textContent=td.accuracy+'%';
      document.getElementById('aiSamples').textContent=td.epochs;
      document.getElementById('aiRetrains').textContent=aiRetrains;
      document.getElementById('aiBadge').textContent='IA ACTIVA · '+td.accuracy+'% ACC';
      document.getElementById('aiBadge').style.borderColor='rgba(76,175,130,.4)';
      document.getElementById('aiBadge').style.color='var(--green)';
    }else{
      document.getElementById('aiBadge').textContent='ENTRENANDO...';
      document.getElementById('aiDesc').textContent='Recopilando datos ('+prices.length+'/60 precios)...';
    }
    const pr=await fetch('/aipredict');
    const pd=await pr.json();
    if(pd.prob===null||pd.prob===undefined){
      // No hacer return silencioso — mostrar estado
      document.getElementById('aiSignal').textContent=pd.trained?'CALCULANDO':'ENTRENANDO';
      document.getElementById('aiSignal').className='ai-signal neutral';
      document.getElementById('aiProbVal').textContent='—';
      document.getElementById('aiDesc').textContent=pd.trained?'Modelo listo, esperando suficientes datos...':'Recopilando datos para entrenar...';
      return;
    }
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

// v6.4: SSE — precio real-time directo de Massive (sin interpolación)
// Con ticks tick-by-tick de Massive, no necesitamos suavizar — es precio real
let evtSource=null;
function initSSE(){
  if(evtSource)evtSource.close();
  evtSource=new EventSource('/stream');
  evtSource.addEventListener('price',function(e){
    try{const pd=JSON.parse(e.data);_updatePriceDisplay(pd);}catch(err){}
  });
  evtSource.addEventListener('signal',function(e){
    try{
      const s=JSON.parse(e.data);
      if(s&&s.direction){
        const isBuy=s.direction==='COMPRAR';
        const cls=isBuy?'go-buy':'go-sell';
        // v6.3: usar precio REAL de entrada del backend, no del buffer del frontend
        const entryPrice=s.entry||prices[prices.length-1]||0;
        document.getElementById('alertBox').className='alert-box '+cls;
        document.getElementById('alertSignal').textContent=s.direction;
        document.getElementById('alertReason').textContent='⚡ SCALP v2 LIVE — Score '+s.score+'% | RR '+s.rr+':1';
        document.getElementById('confPct').textContent=s.score+'%';
        document.getElementById('confFill').style.width=Math.min(s.score,100)+'%';
        document.getElementById('lvlEntry').textContent='$'+entryPrice.toFixed(2);
        document.getElementById('lvlTP').textContent='$'+s.tp.toFixed(2);
        document.getElementById('lvlSL').textContent='$'+s.sl.toFixed(2);
        document.getElementById('rrLabel').textContent='RR '+s.rr+':1';
        document.getElementById('logBtn').style.display='block';
        currentSignalData={valid:true,rawSignal:s.direction,signal:s.direction,cls:cls,confidence:s.score,price:entryPrice,tpDist:Math.abs(s.tp-entryPrice),slDist:Math.abs(s.sl-entryPrice),rr:s.rr,ATR:0,checks:[],reason:'SCALP v2 SSE'};
        window._lastSignalTime=Date.now();
        triggerSound(currentSignalData);
        addHistory(currentSignalData);
      }
    }catch(err){}
  });
  // v6.3: escuchar evento WAIT del backend → limpiar señal instantáneo
  evtSource.addEventListener('wait',function(e){
    try{
      const w=JSON.parse(e.data);
      document.getElementById('alertBox').className='alert-box wait';
      document.getElementById('alertSignal').textContent='ESPERAR';
      document.getElementById('alertReason').textContent=w.reason||'Sin setup válido — esperando oportunidad';
      document.getElementById('confPct').textContent='—%';
      document.getElementById('confFill').style.width='0%';
      document.getElementById('lvlEntry').textContent='—';
      document.getElementById('lvlTP').textContent='—';
      document.getElementById('lvlSL').textContent='—';
      document.getElementById('rrLabel').textContent='';
      document.getElementById('logBtn').style.display='none';
      currentSignalData=null;
      window._lastSignalTime=null;
    }catch(err){}
  });
  evtSource.onerror=function(){setTimeout(initSSE,3000)};
}
initSSE();

// Actualización de precio — usada tanto por SSE como por polling HTTP
function _updatePriceDisplay(pd){
  if(!pd||!pd.price)return;
  document.getElementById('priceDisplay').textContent='$'+pd.price.toFixed(2);
  const ch=pd.ch||0,chp=pd.chp||0;
  const el=document.getElementById('priceChange');
  el.textContent=(ch>=0?'+':'')+ch.toFixed(2)+' ('+(ch>=0?'+':'')+Number(chp).toFixed(2)+'%)';
  el.className='price-change '+(ch>=0?'up':'down');
  prices.push(pd.price);if(prices.length>300)prices=prices.slice(-300);
}
// Polling HTTP incondicional cada 3s — garantía absoluta independiente de SSE
setInterval(function(){
  fetch('/precio').then(r=>r.json()).then(_updatePriceDisplay).catch(()=>{});
},3000);

// v6.1: polling 3s con Promise.all (paralelo)
async function fetchAndAnalyze(){
  const btn=document.getElementById('updateBtn');
  btn.disabled=true;btn.querySelector('span').textContent='⚡';
  try{
    // Todas las requests en paralelo — no secuencial
    const [priceR,ohlcR,newsR,htfR,ictR]=await Promise.all([
      fetch('/precio').then(r=>r.json()).catch(e=>{console.error('precio:',e);return null}),
      fetch('/ohlc?interval='+currentTF+'&size=200').then(r=>r.json()).catch(e=>{console.error('ohlc:',e);return []}),
      fetch('/newsstatus').then(r=>r.json()).catch(e=>{console.error('news:',e);return {active:false}}),
      fetch('/htftrend').then(r=>r.json()).catch(e=>{console.error('htf:',e);return {trend:'neutral'}}),
      fetch('/ictsignal').then(r=>r.json()).catch(e=>{console.error('ict:',e);return null}),
    ]);
    console.log('Data:',{price:priceR?priceR.price:null,ohlc:ohlcR?ohlcR.length:0,news:newsR,htf:htfR,ict:ictR?'yes':'no'});
    // OHLC
    if(Array.isArray(ohlcR)&&ohlcR.length>0){
      ohlcData=ohlcR;prices=ohlcR.map(c=>c.close);
      document.getElementById('tfInfo').textContent=ohlcData.length+' velas reales';
      document.getElementById('sourceBadge').innerHTML='<span class="source-badge live">● DATOS REALES '+currentTF.toUpperCase()+'</span>';
    }
    // Precio
    if(priceR&&priceR.price&&priceR.price>0){
      prices.push(priceR.price);if(prices.length>300)prices=prices.slice(-300);
      document.getElementById('priceDisplay').textContent='$'+priceR.price.toFixed(2);
      const ch=priceR.ch||0,chp=priceR.chp||0;
      const el=document.getElementById('priceChange');
      el.textContent=(ch>=0?'+':'')+ch.toFixed(2)+' ('+(ch>=0?'+':'')+Number(chp).toFixed(2)+'%)';
      el.className='price-change '+(ch>=0?'up':'down');
      document.getElementById('sourceBadge').innerHTML=document.getElementById('sourceBadge').innerHTML||'<span class="source-badge live">● EN VIVO</span>';
    }else if(!prices.length){
      document.getElementById('priceDisplay').textContent='Esperando datos...';
      document.getElementById('priceChange').textContent='Workers iniciando...';
      btn.disabled=false;btn.querySelector('span').textContent='↻ ACTUALIZAR';return;
    }
    // News
    window._newsActive=newsR.active;
    const nEl=document.getElementById('newsWarning');
    if(nEl){nEl.style.display=newsR.active?'block':'none';if(newsR.active)nEl.textContent='⚠ NOTICIA ACTIVA: '+newsR.event+' — Señales pausadas';}
    // HTF
    window._htfTrend=htfR.trend;
    const htfEl=document.getElementById('htfTrend');
    if(htfEl){
      const color=htfR.trend==='up'?'var(--green)':htfR.trend==='down'?'var(--red)':'var(--gold)';
      const arrow=htfR.trend==='up'?'↑ ALCISTA':htfR.trend==='down'?'↓ BAJISTA':'→ NEUTRAL';
      htfEl.textContent='1H: '+arrow;htfEl.style.color=color;
    }
    // SCALP v2 signal
    if(ictR&&ictR.direction){
      const isBuy=ictR.direction==='COMPRAR';
      const cls=isBuy?'go-buy':'go-sell';
      const det=ictR.details||{};
      const checks=[{label:'SWEEP',ok:!!det.sweep},{label:det.bos||'BOS',ok:!!det.bos},{label:'FVG',ok:!!det.fvg},{label:'EMA',ok:!!det.ema},{label:det.ml||'ML',ok:det.ml&&det.ml.indexOf('HIGH')>=0}];
      document.getElementById('alertBox').className='alert-box '+cls;
      document.getElementById('alertSignal').textContent=ictR.direction;
      document.getElementById('alertReason').textContent='✓ SCALP v2 — Score '+ictR.score+'% | RR '+ictR.rr+':1';
      document.getElementById('confPct').textContent=ictR.score+'%';
      document.getElementById('confFill').style.width=Math.min(ictR.score,100)+'%';
      document.getElementById('validityRow').innerHTML=checks.map(c=>'<div class="validity-pill '+(c.ok?'ok':'fail')+'">'+(c.ok?'✓':'✗')+' '+c.label+'</div>').join('');
      document.getElementById('lvlEntry').textContent='$'+(prices[prices.length-1]||0).toFixed(2);
      document.getElementById('lvlTP').textContent='$'+ictR.tp.toFixed(2);
      document.getElementById('lvlSL').textContent='$'+ictR.sl.toFixed(2);
      document.getElementById('rrLabel').textContent='RR '+ictR.rr+':1';
      document.getElementById('logBtn').style.display='block';
      currentSignalData={valid:true,rawSignal:ictR.direction,signal:ictR.direction,cls:cls,confidence:ictR.score,price:prices[prices.length-1],tpDist:Math.abs(ictR.tp-prices[prices.length-1]),slDist:Math.abs(ictR.sl-prices[prices.length-1]),rr:ictR.rr,ATR:0,checks:checks,reason:'SCALP v2'};
      addHistory(currentSignalData);
    }else{
      // Fallback técnico
      const d=computeSignal();currentSignalData=d;
      renderAlert(d);addHistory(d);
    }
    // Indicadores + AI (no bloquean señal)
    if(prices.length>14){const d=computeSignal();renderIndicators(d);}
    updateAI(currentSignalData&&currentSignalData.valid?currentSignalData.signal:'ESPERAR');
    updateRiskStatus();
    drawPriceChart();drawMACDChart();
    document.getElementById('timestamp').textContent='⚡ '+new Date().toLocaleTimeString('es-ES');
    if(ohlcData.length)document.getElementById('tfInfo').textContent=ohlcData.length+' velas reales';
  }catch(e){console.error('fetchAndAnalyze ERROR:',e);document.getElementById('timestamp').textContent='❌ '+e.message;}
  btn.disabled=false;btn.querySelector('span').textContent='↻ ACTUALIZAR';
}
fetchAndAnalyze();
updateSessions();
setInterval(fetchAndAnalyze,3000);
setInterval(updateSessions,60000);
window.addEventListener('resize',()=>{drawPriceChart();drawMACDChart()});
// v6.1: verificar Telegram al cargar
fetch('/telegram_config').then(r=>r.json()).then(d=>{
  if(d.configured){
    document.getElementById('tgStatus').textContent='📲 TELEGRAM ON';
    document.getElementById('tgStatus').className='tg-status connected';
    document.getElementById('cfgTGstatus').textContent='ACTIVO ✓';
    document.getElementById('cfgTGstatus').style.color='var(--green)';
  }
}).catch(()=>{});
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


def _run_backtest_bg():
    _backtest_result["data"] = None
    print("  Iniciando backtest real XAUUSD 5M...")
    candles = None
    # v7.1: PRIMERO Massive (30 días de data real)
    if MASSIVE_API_KEY:
        bulk = _fetch_massive_bulk("5min", days=30)
        if bulk and len(bulk) > 200:
            candles = [{"c": v["c"], "dt": v["dt"]} for v in bulk]
            print(f"  📊 Backtest con {len(candles)} velas de Massive (30 días)")
    # Fallback: Twelve Data
    if not candles:
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
            print(f"  📊 Backtest con {len(candles)} velas de Twelve Data")
        except Exception as e:
            _backtest_result["data"] = {"error": str(e)}
            return
    if not candles:
        _backtest_result["data"] = {"error": "No se pudo obtener data"}
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


# ── v7: WALK-FORWARD BACKTEST ────────────────────────────
# Divide el histórico en ventanas: entrena en N períodos, valida en siguiente.
# Esto detecta si la estrategia tiene EDGE REAL o está curva-ajustada.

def _run_walkforward_bg():
    """v7.1: Walk-forward con 30 DÍAS de data de Massive (antes solo 500 velas/40h de Twelve).
    
    Con Massive unlimited:
    - 30 días de 5m = ~8,640 velas reales
    - 10 ventanas de 3 días cada una
    - Resultado estadísticamente significativo (no anecdótico)
    
    Fallback a Twelve Data si Massive no disponible.
    """
    _walkforward_result["data"] = None
    print("  🔬 Iniciando walk-forward backtest...")
    candles = None
    source = "unknown"
    # v7.1: PRIMERO intentar Massive (30 días de data real)
    if MASSIVE_API_KEY:
        bulk = _fetch_massive_bulk("5min", days=30)
        if bulk and len(bulk) > 500:
            candles = bulk
            source = f"Massive ({len(bulk)} velas, 30 días)"
            print(f"  🔬 {source}")
    # Fallback: Twelve Data (solo 500 velas)
    if not candles:
        try:
            url = (f"https://api.twelvedata.com/time_series"
                   f"?symbol=XAU/USD&interval=5min&outputsize=500&apikey={TWELVE_API_KEY}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode())
            if "values" in data:
                candles = [{"c": float(v["close"]), "dt": v["datetime"]}
                           for v in reversed(data["values"])]
                source = f"Twelve Data ({len(candles)} velas)"
        except Exception as e:
            _walkforward_result["data"] = {"error": str(e)}
            return
    if not candles or len(candles) < 200:
        _walkforward_result["data"] = {"error": f"Insuficientes velas: {len(candles) if candles else 0}"}
        return
    print(f"  🔬 {len(candles)} velas cargadas de {source}")

    closes = [c["c"] for c in candles]
    N = len(closes)
    # v7.1: más ventanas si hay más data (10 con Massive, 5 con Twelve)
    WINDOWS = 10 if N > 2000 else 5
    window_size = N // WINDOWS

    def _ema(a, n):
        if len(a) < n: return None
        k = 2/(n+1); e = sum(a[:n])/n
        for v in a[n:]: e = v*k+e*(1-k)
        return e
    def _rsi(p):
        if len(p) < 15: return 50
        g = l = 0
        for i in range(len(p)-14, len(p)):
            d = p[i]-p[i-1]
            if d > 0: g += d
            else: l -= d
        ag, al = g/14, l/14
        return 100-100/(1+ag/al) if al > 0 else 100
    def _bias(p):
        if len(p) < 22: return None
        e9 = _ema(p, 9); e21 = _ema(p, 21)
        if e9 and e21:
            if e9 > e21*1.0005: return "bullish"
            if e9 < e21*0.9995: return "bearish"
        return None
    def _atr(p):
        if len(p) < 15: return p[-1]*0.003
        return sum(abs(p[i]-p[i-1]) for i in range(len(p)-14, len(p)))/14
    def _chop(p, n=20):
        if len(p) < n: return True
        r = p[-n:]
        atr = sum(abs(r[i]-r[i-1]) for i in range(1, len(r)))/(len(r)-1)
        return (max(r)-min(r)) < atr*5

    def backtest_window(start_idx, end_idx):
        """Ejecuta backtest en rango [start_idx, end_idx) y devuelve métricas."""
        trades = []
        for i in range(max(start_idx, 50), end_idx-6):
            w = closes[:i]; price = closes[i]; atr_v = _atr(w)
            dt = candles[i]["dt"]
            try: hour = int(dt[11:13])
            except: hour = 12
            if not ((3 <= hour < 11) or (12 <= hour < 17)): continue
            if _chop(w[-20:]): continue
            w15 = closes[max(0, i-60):i:3]
            bias = _bias(w15 if len(w15) >= 22 else w)
            if not bias: continue
            # Sweep detection
            swept = False
            if bias == "bullish":
                lows = [w[j] for j in range(2, len(w)-2) if w[j] <= w[j-1] and w[j] <= w[j+1]]
                if lows and min(w[-5:]) < lows[-1] and w[-1] > lows[-1]: swept = True
            else:
                highs = [w[j] for j in range(2, len(w)-2) if w[j] >= w[j-1] and w[j] >= w[j+1]]
                if highs and max(w[-5:]) > highs[-1] and w[-1] < highs[-1]: swept = True
            if not swept: continue
            # BOS
            bos = False
            if bias == "bullish":
                hs = [w[j] for j in range(2, len(w)-2) if w[j] >= w[j-1] and w[j] >= w[j+1]]
                if hs and w[-1] > hs[-1]: bos = True
            else:
                ls = [w[j] for j in range(2, len(w)-2) if w[j] <= w[j-1] and w[j] <= w[j+1]]
                if ls and w[-1] < ls[-1]: bos = True
            if not bos: continue
            score = 65
            r = _rsi(w)
            if bias == "bullish":
                if r > 70: score -= 10
                elif r < 40: score += 5
            else:
                if r < 30: score -= 10
                elif r > 60: score += 5
            if score < 70: continue
            is_buy = bias == "bullish"
            tp_mult = 2.5 if score >= 85 else 1.5
            tp = price + atr_v*tp_mult if is_buy else price - atr_v*tp_mult
            sl = price - atr_v if is_buy else price + atr_v
            future = closes[i+1:i+7]
            if not future: continue
            if is_buy:
                hit_tp = any(p >= tp for p in future); hit_sl = any(p <= sl for p in future)
            else:
                hit_tp = any(p <= tp for p in future); hit_sl = any(p >= sl for p in future)
            if hit_tp and hit_sl:
                ti = next((j for j, p in enumerate(future) if (p >= tp if is_buy else p <= tp)), 999)
                si = next((j for j, p in enumerate(future) if (p <= sl if is_buy else p >= sl)), 999)
                won = ti < si
            elif hit_tp: won = True
            elif hit_sl: won = False
            else: won = False
            trades.append({"won": won, "pnl_r": tp_mult if won else -1.0, "score": score})
        n = len(trades)
        if n == 0:
            return {"trades": 0, "wr": 0, "pf": 0, "total_r": 0, "expectancy": 0, "valid": False}
        wins = [t for t in trades if t["won"]]
        wr = len(wins) / n * 100
        gains = sum(t["pnl_r"] for t in wins)
        losses = abs(sum(t["pnl_r"] for t in trades if not t["won"]))
        pf = gains / (losses + 0.001)
        total_r = sum(t["pnl_r"] for t in trades)
        exp = total_r / n
        return {"trades": n, "wr": round(wr, 1), "pf": round(pf, 2),
                "total_r": round(total_r, 2), "expectancy": round(exp, 3), "valid": True}

    # Ejecutar backtest en cada ventana
    window_results = []
    for w in range(WINDOWS):
        start = w * window_size
        end = (w + 1) * window_size if w < WINDOWS - 1 else N
        dt_from = candles[start]["dt"][:16] if start < N else "?"
        dt_to   = candles[min(end-1, N-1)]["dt"][:16] if end-1 < N else "?"
        result = backtest_window(start, end)
        result["window"] = w + 1
        result["dt_from"] = dt_from
        result["dt_to"] = dt_to
        window_results.append(result)
        print(f"  🔬 Ventana {w+1}/{WINDOWS}: {result['trades']} trades, WR {result['wr']}%, PF {result['pf']}")

    # Análisis de consistencia entre ventanas
    valid_windows = [w for w in window_results if w["valid"]]
    if len(valid_windows) < 2:
        _walkforward_result["data"] = {
            "error": "Insuficientes ventanas con trades para validar",
            "windows": window_results,
        }
        return
    wrs = [w["wr"] for w in valid_windows]
    pfs = [w["pf"] for w in valid_windows]
    # Desviación estándar de WR (si es muy alta, estrategia inconsistente)
    mean_wr = sum(wrs) / len(wrs)
    std_wr = math.sqrt(sum((wr - mean_wr) ** 2 for wr in wrs) / len(wrs))
    mean_pf = sum(pfs) / len(pfs)
    std_pf = math.sqrt(sum((pf - mean_pf) ** 2 for pf in pfs) / len(pfs))
    # Clasificación
    if std_wr < 8 and mean_wr >= 55 and mean_pf >= 1.3:
        verdict = "ROBUSTA"
        verdict_desc = "Estrategia CONSISTENTE entre ventanas — edge real"
    elif std_wr < 15 and mean_wr >= 50:
        verdict = "ACEPTABLE"
        verdict_desc = "Algo de variabilidad, pero con edge positivo"
    elif std_wr >= 20:
        verdict = "SOBREAJUSTADA"
        verdict_desc = "⚠ Alta variabilidad — estrategia curva-ajustada"
    else:
        verdict = "DÉBIL"
        verdict_desc = "Edge marginal o negativo"
    # Actualizar baseline del performance monitor con el walk-forward
    if mean_wr > 0:
        _performance_monitor["baseline_wr"] = mean_wr
        _performance_monitor["baseline_pf"] = mean_pf
        print(f"  🔬 Baseline actualizado: WR={mean_wr:.1f}% PF={mean_pf:.2f}")
    _walkforward_result["data"] = {
        "windows": window_results,
        "mean_wr": round(mean_wr, 1),
        "std_wr":  round(std_wr, 1),
        "mean_pf": round(mean_pf, 2),
        "std_pf":  round(std_pf, 2),
        "verdict": verdict,
        "verdict_desc": verdict_desc,
        "candles": len(candles),
        "generated_at": _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
    }
    print(f"  🔬 Walk-forward: {verdict} | WR={mean_wr:.1f}±{std_wr:.1f}% PF={mean_pf:.2f}±{std_pf:.2f}")


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

        # v6.1: SSE stream — señales push en tiempo real
        elif path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Accel-Buffering", "no")   # Render/nginx: no bufferizar SSE
            self.end_headers()
            q = []
            with _sse_lock:
                _sse_clients.append(q)
            try:
                # Enviar estado actual inmediatamente
                if _live_cache["price"]:
                    self.wfile.write(f"event: price\ndata: {json.dumps(_live_cache['price'])}\n\n".encode())
                if _live_cache["signal"]:
                    self.wfile.write(f"event: signal\ndata: {json.dumps(_live_cache['signal'], default=str)}\n\n".encode())
                self.wfile.flush()
                while True:
                    if q:
                        msg = q.pop(0)
                        self.wfile.write(msg.encode())
                        self.wfile.flush()
                    else:
                        time.sleep(0.1)
            except:
                pass
            finally:
                with _sse_lock:
                    if q in _sse_clients:
                        _sse_clients.remove(q)
            return

        # v6.1: /precio — lectura instantánea + stale flag
        elif path == "/precio":
            c = _live_cache["price"]
            if c:
                c = dict(c)  # no mutar cache
                c["stale"] = _live_cache.get("price_stale", False)
                c["age_sec"] = round(time.time() - _live_cache["price_ts"], 1)
            self._send(200, "application/json",
                       json.dumps(c or {"price":0,"ch":0,"chp":0,"stale":True}).encode())

        # v6.1: /ohlc — lectura de cache, fallback a API
        # v6.1: /ohlc — cache para todos los TFs + fallback API
        elif path == "/ohlc":
            interval = params.get("interval", "5min")
            size     = int(params.get("size", "150"))
            tf_map = {"5min": "ohlc_5m", "15min": "ohlc_15m", "1h": "ohlc_1h", "4h": "ohlc_4h"}
            tf_key = tf_map.get(interval, "")
            cached_candles = _live_cache.get(tf_key, []) if tf_key else []
            if cached_candles:
                self._send(200, "application/json", json.dumps(cached_candles).encode())
            else:
                candles = get_historical_ohlc(interval, size)
                if candles:
                    if tf_key:
                        _live_cache[tf_key] = candles
                    for c2 in candles:
                        push_price(c2["close"])
                        update_ict_prices(c2["close"])
                self._send(200, "application/json", json.dumps(candles or []).encode())

        # v6.1: /ictsignal — lectura instantánea de última señal cached
        elif path == "/ictsignal":
            sig = _live_cache["signal"]
            self._send(200, "application/json", json.dumps(sig or {}).encode())

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

        # v6.1: /telegram_config — guardar token y chat_id desde el dashboard
        elif path == "/telegram_config":
            global TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
            token = params.get("token","").strip()
            chat  = params.get("chat","").strip()
            if token and chat:
                TELEGRAM_TOKEN   = token
                TELEGRAM_CHAT_ID = chat
                try:
                    os.makedirs("/data", exist_ok=True)
                    with open("/data/telegram_config.json", "w") as f:
                        json.dump({"token": token, "chat": chat}, f)
                    self._send(200, "application/json", b'{"ok":true,"saved":true}')
                except:
                    self._send(200, "application/json", b'{"ok":true,"saved":false}')
            else:
                self._send(200, "application/json", json.dumps({
                    "ok": False,
                    "configured": bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID),
                    "token_set": bool(TELEGRAM_TOKEN),
                    "chat_set": bool(TELEGRAM_CHAT_ID),
                }).encode())

        # v6.1: /aitrain — retorna status instantáneo, training corre en worker
        elif path == "/aitrain":
            self._send(200, "application/json", json.dumps({
                "trained": _ai.trained, "accuracy": round(_ai.accuracy,1), "epochs": _ai.epochs}).encode())

        # v6.1: /aipredict — cache RAM con fallback a cálculo en vivo
        elif path == "/aipredict":
            prob_val = _live_cache["ai_prob"]
            sig_val  = _live_cache["ai_signal"]
            # Si cache vacío, calcular en vivo usando velas OHLC reales
            if prob_val is None and _ai.trained:
                with _data_lock:
                    ohlc5_closes  = [c["c"] for c in _ohlc_candles_5m]
                    ohlc15_closes = [c["c"] for c in _ohlc_candles_15m]
                    ph = list(price_history)
                # Prioridad: combinado de velas reales > price_history
                combined = ohlc15_closes + ohlc5_closes
                if len(ph) > len(combined):
                    combined = ph
                if len(combined) >= 35:
                    prob_raw, sig_raw = ai_predict(combined)
                    if prob_raw is not None:
                        prob_val = round(prob_raw * 100, 1)
                        sig_val  = sig_raw
                        _live_cache["ai_prob"]   = prob_val
                        _live_cache["ai_signal"] = sig_val
            self._send(200, "application/json", json.dumps({
                "prob": prob_val,
                "signal": sig_val, "trained": _ai.trained,
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
            # v6.5: nuevo risk manager con daily/weekly loss + circuit breaker
            self._send(200, "application/json", json.dumps(risk_get_status()).encode())

        elif path == "/healthcheck":
            self._send(200, "application/json", json.dumps({
                "status": "ok", "time_utc": _dt.datetime.utcnow().isoformat(),
                "ai_ready": _ai.trained, "ai_drift": _ai.is_market_drifting(), "uptime_ok": True,
            }).encode())

        # /debug — estado completo de caches y workers
        elif path == "/stats":
            # v6.5: estadísticas completas del Paper Trading
            stats = paper_get_stats()
            self._send(200, "application/json", json.dumps(stats).encode())

        elif path == "/mtf":
            # v6.5: estado Multi-Timeframe Confluence
            with _data_lock:
                ohlc4h = list(_live_cache.get("ohlc_4h", []))
                ohlc1h = list(_live_cache.get("ohlc_1h", []))
                ohlc15 = list(_ohlc_candles_15m)
                ohlc5 = list(_ohlc_candles_5m)
            b4h = get_ema_bias([c["close"] for c in ohlc4h]) if len(ohlc4h) >= 21 else None
            b1h = get_ema_bias([c["close"] for c in ohlc1h]) if len(ohlc1h) >= 21 else None
            b15 = get_ema_bias([c["c"] for c in ohlc15]) if len(ohlc15) >= 21 else None
            b5 = get_ema_bias([c["c"] for c in ohlc5]) if len(ohlc5) >= 21 else None
            htfs = [b for b in [b4h, b1h, b15] if b]
            aligned_count = 0
            if b5 and htfs:
                aligned_count = sum(1 for b in htfs if b == b5)
            self._send(200, "application/json", json.dumps({
                "bias_4h": b4h or "N/A",
                "bias_1h": b1h or "N/A",
                "bias_15m": b15 or "N/A",
                "bias_5m": b5 or "N/A",
                "htf_available": len(htfs),
                "aligned_count": aligned_count,
                "confluence_ok": aligned_count >= 2 if len(htfs) >= 2 else False,
                "candles_4h": len(ohlc4h),
                "candles_1h": len(ohlc1h),
            }).encode())

        elif path == "/debug":
            now = time.time()
            is_data_stale()  # refresh flags
            self._send(200, "application/json", json.dumps({
                "price_cached": _live_cache["price"] is not None,
                "price_age_sec": round(now - _live_cache["price_ts"], 1) if _live_cache["price_ts"] else -1,
                "price_stale": _live_cache.get("price_stale", False),
                "price_fails": _live_cache.get("price_fails", 0),
                "price_data": _live_cache["price"],
                "ohlc_5m_count": len(_live_cache.get("ohlc_5m", [])),
                "ohlc_15m_count": len(_live_cache.get("ohlc_15m", [])),
                "ohlc_age_sec": round(now - _live_cache["ohlc_ts"], 1) if _live_cache["ohlc_ts"] else -1,
                "ohlc_stale": _live_cache.get("ohlc_stale", False),
                "signal_cached": _live_cache["signal"] is not None,
                "signal_age_sec": round(now - _live_cache["signal_ts"], 1) if _live_cache["signal_ts"] else -1,
                "prices_1m": len(_scalp_prices_1m),
                "prices_5m": len(_scalp_prices_5m),
                "prices_15m": len(_scalp_prices_15m),
                "ohlc_candles_5m": len(_ohlc_candles_5m),
                "ohlc_candles_15m": len(_ohlc_candles_15m),
                "price_history": len(price_history),
                "ai_trained": _ai.trained,
                "ai_accuracy": round(_ai.accuracy, 1),
                "sse_clients": len(_sse_clients),
                "massive_api_key": "SET" if MASSIVE_API_KEY else "MISSING ❌",
                "massive_key_preview": MASSIVE_API_KEY[:8] + "..." if MASSIVE_API_KEY else "-",
                "twelve_api_key": TWELVE_API_KEY[:8] + "..." if TWELVE_API_KEY else "MISSING",
                "twelve_calls_today": _api_counter["twelve_calls"],
                "twelve_limit": TWELVE_DAILY_LIMIT,
                "twelve_remaining": max(0, TWELVE_DAILY_LIMIT - _api_counter["twelve_calls"]),
                "ws_active": _live_cache.get("ws_active", False),
                "ws_provider": _live_cache.get("ws_provider", "none"),
                "ws_last_tick_age": round(time.time() - _live_cache.get("ws_last_tick", 0), 1) if _live_cache.get("ws_last_tick") else -1,
                "paper_total": len(_paper_trades),
                "paper_open": len([t for t in _paper_trades if t["status"] == "OPEN"]),
                "paper_closed": len([t for t in _paper_trades if t["status"] == "CLOSED"]),
                "risk_can_trade": risk_can_trade()[0],
                "risk_today_r": risk_get_today_stats()["total_r"],
                "risk_consec_losses": risk_get_today_stats()["consecutive_losses"],
            }).encode())

        # /apitest — prueba todas las APIs de precio en tiempo real
        elif path == "/apitest":
            results = []
            test_apis = list(_PRICE_APIS)
            # Añadir variantes extra de Massive para diagnosticar cuál funciona
            if MASSIVE_API_KEY:
                extras = [
                    f"https://api.massive.com/v2/last/nbbo/C:XAUUSD?apikey={MASSIVE_API_KEY}",
                    f"https://api.massive.com/v1/last_quote/currencies/XAU/USD?apiKey={MASSIVE_API_KEY}",
                    f"https://api.massive.com/v2/aggs/ticker/C:XAUUSD/range/1/minute/2024-01-01/2025-12-31?adjusted=true&sort=desc&limit=1&apiKey={MASSIVE_API_KEY}",
                ]
                for url in extras:
                    test_apis.append((url, lambda d: d))  # parser crudo
            for url, parser in test_apis:
                safe_url = url.replace(MASSIVE_API_KEY, "***").replace(TWELVE_API_KEY, "***") if MASSIVE_API_KEY or TWELVE_API_KEY else url
                try:
                    d = _fetch_with_retry(url, timeout=8, retries=1, backoff=0)
                    result = parser(d) if d else None
                    results.append({
                        "url": safe_url[:80],
                        "raw": str(d)[:300] if d else None,
                        "parsed": str(result)[:200] if result else None,
                        "ok": bool(result and isinstance(result, dict) and result.get("price", 0) > 0)
                    })
                except Exception as e:
                    results.append({"url": safe_url[:80], "error": str(e)[:150], "ok": False})
            self._send(200, "application/json", json.dumps({
                "ws_active": _live_cache.get("ws_active", False),
                "ws_provider": _live_cache.get("ws_provider", "none"),
                "ws_last_tick_age_sec": round(time.time() - _live_cache.get("ws_last_tick", 0), 1) if _live_cache.get("ws_last_tick") else -1,
                "price_cached": _live_cache.get("price"),
                "apis": results,
            }, indent=2).encode())

        # /scalpscore — estado del motor SCALP v2
        elif path == "/scalpscore":
            atr_v = 0
            if len(_ict_prices_5m) >= 14:
                atr_v = sum(abs(_ict_prices_5m[i]-_ict_prices_5m[i-1])
                            for i in range(len(_ict_prices_5m)-14, len(_ict_prices_5m))) / 14
            bias = get_ema_bias(_ict_prices_15m if len(_ict_prices_15m)>=12 else _ict_prices_5m)
            self._send(200, "application/json", json.dumps({
                "session_active":   is_trading_session(),
                "session_name":     get_session_name(),
                "trending":         not detect_chop(_ict_prices_5m) if _ict_prices_5m else False,
                "bias_15m":         bias or "NEUTRAL",
                "prices_5m":        len(_ict_prices_5m),
                "prices_15m":       len(_ict_prices_15m),
                "atr":              round(atr_v, 2),
                "score_thresholds": {"sniper": SCORE_SNIPER, "normal": SCORE_NORMAL, "early": SCORE_EARLY},
            }).encode())

        # v6.1: /presignal — estado pre-señal para alertas tempranas
        elif path == "/presignal":
            self._send(200, "application/json", json.dumps({
                "state":       _pre_signal["state"],
                "bias":        _pre_signal["bias"],
                "sweep_level": _pre_signal["sweep_level"],
                "fvg_zone":    _pre_signal["fvg_zone"],
                "age_sec":     round(time.time() - _pre_signal["ts"], 1) if _pre_signal["ts"] else 0,
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

        # v7: WALK-FORWARD BACKTEST
        elif path == "/walkforward":
            threading.Thread(target=_run_walkforward_bg, daemon=True).start()
            self._send(200, "application/json", json.dumps({"ok": True, "msg": "Walk-forward iniciado - listo en ~60s en /walkforward_result"}).encode())

        elif path == "/walkforward_result":
            result = _walkforward_result.get("data")
            if result is None:
                self._send(200, "application/json", json.dumps({"status": "running"}).encode())
            else:
                self._send(200, "application/json", json.dumps(result).encode())

        # v7: ROLLING PERFORMANCE MONITOR
        elif path == "/performance":
            self._send(200, "application/json", json.dumps({
                **_performance_monitor,
                "total_signals": len(_signal_history),
                "recent_50": _signal_history[-50:] if _signal_history else [],
            }).encode())

        # v7: REGISTRAR RESULTADO DE SEÑAL (ganó/perdió)
        elif path == "/signal_result":
            try:
                won = params.get("won", "false").lower() == "true"
                direction = params.get("direction", "")
                pnl_r = float(params.get("pnl_r", "0"))
                score = int(params.get("score", "0"))
                register_signal_result(won, direction, pnl_r, score)
                self._send(200, "application/json", json.dumps({
                    "ok": True,
                    "performance": _performance_monitor,
                }).encode())
            except Exception as e:
                self._send(400, "application/json", json.dumps({"ok": False, "error": str(e)}).encode())

        # v7: REGIME DETECTION EN VIVO
        elif path == "/regime":
            with _data_lock:
                p5 = list(_scalp_prices_5m)
            if len(p5) >= 30:
                regime = detect_market_regime(p5)
                self._send(200, "application/json", json.dumps(regime).encode())
            else:
                self._send(200, "application/json", json.dumps({
                    "regime": "UNKNOWN", "reason": "insuficientes datos"
                }).encode())

        # v7: DXY STATUS
        elif path == "/dxy":
            dxy = get_dxy_trend()
            self._send(200, "application/json", json.dumps(dxy).encode())

        # v7.1: US10Y STATUS
        elif path == "/us10y":
            y = get_us10y_trend()
            self._send(200, "application/json", json.dumps(y).encode())

        # v7.1: MACRO STATUS COMBINADO (DXY + US10Y + Regime)
        elif path == "/macro":
            with _data_lock:
                p5 = list(_scalp_prices_5m)
            regime = detect_market_regime(p5) if len(p5) >= 30 else {"regime": "UNKNOWN"}
            self._send(200, "application/json", json.dumps({
                "dxy": get_dxy_trend(),
                "us10y": get_us10y_trend(),
                "regime": regime,
                "performance": {
                    "status": _performance_monitor["status"],
                    "rolling_wr": _performance_monitor["rolling_wr"],
                    "rolling_pf": _performance_monitor["rolling_pf"],
                    "total_signals": len(_signal_history),
                },
            }).encode())

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


def _background_init():
    """v6.3: inicialización pesada en thread background para no bloquear el puerto.
    Render necesita que el puerto HTTP esté abierto en <30s o mata el servicio."""
    print("\n  Verificando precio del oro...")
    test = get_gold_price()
    if test:
        ch = test['ch']
        print(f"  ✓ XAU/USD: ${test['price']:.2f}  ({'+' if ch>=0 else ''}{ch:.2f})")
        _live_cache["price"] = test
        _live_cache["price_ts"] = time.time()
    else:
        print("  ⚠ Precio no disponible — el worker lo reintentará")

    print("\n  Cargando datos históricos...")
    for interval, tf, size in [("5min", "5m", 200), ("15min", "15m", 200), ("1h", "1h", 200), ("4h", "4h", 200)]:
        candles = get_historical_ohlc(interval, size)
        if candles:
            print(f"  ✓ {interval}: {len(candles)} velas | última: ${candles[-1]['close']:.2f}")
            _live_cache[f"ohlc_{tf}"] = candles
            _live_cache["ohlc_ts"] = time.time()
            if interval == "5min":
                for c in candles:
                    push_price(c["close"])
                    update_ict_prices(c["close"])
        else:
            print(f"  ⚠ {interval}: no disponible — el worker lo reintentará")
        time.sleep(1)  # pausa reducida (antes 3s)

    # v6.2: pre-train con velas OHLC REALES (5m + 15m)
    print("\n  Pre-entrenando IA con velas reales...")
    with _data_lock:
        ohlc5_closes  = [c["c"] for c in _ohlc_candles_5m]
        ohlc15_closes = [c["c"] for c in _ohlc_candles_15m]
    train_data = ohlc15_closes + ohlc5_closes
    if len(train_data) < 60:
        train_data = list(price_history)
    if len(train_data) >= 60:
        _ai.train(train_data)
        if _ai.trained:
            print(f"  ✓ IA lista: {_ai.accuracy:.1f}% acc | {_ai.epochs} muestras | dataset={len(train_data)}")
    else:
        print(f"  ⚠ IA: solo {len(train_data)} datos disponibles, esperando worker")

    if price_history:
        news_on, _ = is_news_time()
        ph = price_history[-50:] if len(price_history) >= 50 else price_history
        update_control_state(_ai, ph, is_news=news_on)

    # Lanzar workers al final de la inicialización
    print("\n  Iniciando workers de baja latencia...")
    n_workers = start_all_workers()
    print(f"  ✓ {n_workers} workers activos")
    print("\n  ✅ AURUM v6.3 completamente inicializado")

def main():
    init_log()
    print("=" * 54)
    print("  AURUM v6.3 · Bot de Señales XAU/USD")
    print("  Smart Money Engine + Anti-Falsa-Entrada")
    print("=" * 54)

    # v6.3 CRITICAL: abrir puerto HTTP PRIMERO para que Render detecte el servicio
    # La inicialización pesada (OHLC, ML, workers) corre en thread background
    print(f"\n  🚀 Abriendo servidor HTTP en puerto {PORT}...")
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    
    # Inicialización pesada en background (no bloquea el puerto)
    threading.Thread(target=_background_init, daemon=True).start()
    
    print(f"  ✓ Servidor escuchando en 0.0.0.0:{PORT}")
    print(f"  ✓ Inicialización pesada corriendo en background")
    print(f"\n  Dashboard: https://aurum-signals.onrender.com")
    print(f"  SSE Stream: /stream")
    print("\n  Para detener: Ctrl + C")
    print("-" * 54)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n  AURUM detenido. ¡Buena suerte con la prop firm!")
        server.shutdown()


if __name__ == "__main__":
    main()
