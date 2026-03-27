#!/usr/bin/env python3
"""
AURUM SCALP v2 — Backtesting Real
Uso: python aurum_backtest.py
Output: aurum_backtest_report.html + aurum_backtest_data.json
"""
import urllib.request, json, math, os, time
from datetime import datetime, timezone

API_KEY = os.environ.get("TWELVE_API_KEY", "dd53883de1a84cccaf65bf7f4e7a4756")

def fetch_candles(interval, size=500):
    print(f"  Descargando {size} velas {interval}...")
    url = (f"https://api.twelvedata.com/time_series"
           f"?symbol=XAU/USD&interval={interval}"
           f"&outputsize={size}&apikey={API_KEY}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode())
    if "values" not in data:
        raise Exception(f"API error: {data.get('message', data)}")
    candles = [{"o": float(v["open"]), "h": float(v["high"]),
                "l": float(v["low"]),  "c": float(v["close"]),
                "dt": v["datetime"]}
               for v in reversed(data["values"])]
    print(f"  {len(candles)} velas | {candles[0]['dt']} -> {candles[-1]['dt']}")
    return candles

def ema(arr, n):
    if len(arr) < n: return None
    k = 2/(n+1); e = sum(arr[:n])/n
    for v in arr[n:]: e = v*k+e*(1-k)
    return e

def atr14(prices):
    if len(prices) < 15: return prices[-1]*0.003
    return sum(abs(prices[i]-prices[i-1]) for i in range(len(prices)-14,len(prices)))/14

def rsi14(prices):
    if len(prices) < 15: return 50
    g=l=0
    for i in range(len(prices)-14,len(prices)):
        d=prices[i]-prices[i-1]
        if d>0: g+=d
        else: l-=d
    ag,al=g/14,l/14
    return 100-100/(1+ag/al) if al>0 else 100

def swing_lows(prices, n=2):
    lows=[]
    for i in range(n,len(prices)-n):
        if (all(prices[i]<=prices[i-j] for j in range(1,n+1)) and
            all(prices[i]<=prices[i+j] for j in range(1,n+1))):
            lows.append((i,prices[i]))
    return lows

def swing_highs(prices, n=2):
    highs=[]
    for i in range(n,len(prices)-n):
        if (all(prices[i]>=prices[i-j] for j in range(1,n+1)) and
            all(prices[i]>=prices[i+j] for j in range(1,n+1))):
            highs.append((i,prices[i]))
    return highs

def detect_sweep(prices, bias):
    if len(prices)<20: return False,0,0
    if bias=="bullish":
        lows=swing_lows(prices[:-3],n=2)
        if not lows: return False,0,0
        _,prev_low=lows[-1]
        recent_low=min(prices[-5:])
        recent_close=prices[-1]
        if recent_low<prev_low and recent_close>prev_low:
            rng=max(prices[-4:])-min(prices[-4:])+1e-9
            return True,round(prev_low,2),round((prev_low-recent_low)/rng,2)
    if bias=="bearish":
        highs=swing_highs(prices[:-3],n=2)
        if not highs: return False,0,0
        _,prev_high=highs[-1]
        recent_high=max(prices[-5:])
        recent_close=prices[-1]
        if recent_high>prev_high and recent_close<prev_high:
            rng=max(prices[-4:])-min(prices[-4:])+1e-9
            return True,round(prev_high,2),round((recent_high-prev_high)/rng,2)
    return False,0,0

def detect_bos(prices, bias):
    if len(prices)<15: return False,"none"
    if bias=="bullish":
        highs=swing_highs(prices[:-2],n=2)
        if highs and prices[-1]>highs[-1][1]: return True,"BOS alcista"
        if len(prices)>=10 and prices[-1]>prices[-5] and prices[-5]<prices[-10]: return True,"CHOCH"
    if bias=="bearish":
        lows=swing_lows(prices[:-2],n=2)
        if lows and prices[-1]<lows[-1][1]: return True,"BOS bajista"
        if len(prices)>=10 and prices[-1]<prices[-5] and prices[-5]>prices[-10]: return True,"CHOCH"
    return False,"none"

def detect_fvg(prices, bias):
    if len(prices)<5: return False,0,0
    for i in range(len(prices)-4,len(prices)-1):
        if i<2: continue
        low1=prices[i-2]; high3=prices[i]
        if bias=="bullish" and high3>low1 and (high3-low1)>prices[-1]*0.0008:
            return True,round(low1,2),round(high3,2)
        if bias=="bearish" and high3<low1 and (low1-high3)>prices[-1]*0.0008:
            return True,round(high3,2),round(low1,2)
    return False,0,0

def detect_chop(prices, n=20):
    if len(prices)<n: return True
    recent=prices[-n:]
    avg_atr=sum(abs(recent[i]-recent[i-1]) for i in range(1,len(recent)))/(len(recent)-1)
    return (max(recent)-min(recent))<avg_atr*5

def get_bias(prices):
    if len(prices)<22: return None
    e9=ema(prices,9); e21=ema(prices,21)
    if e9 and e21:
        if e9>e21*1.0005: return "bullish"
        if e9<e21*0.9995: return "bearish"
    return None

def run_scalp_v2(candles_5m):
    closes=[c["c"] for c in candles_5m]
    trades=[]
    skipped={"no_session":0,"chop":0,"no_bias":0,"no_sweep":0,"no_bos":0,"low_score":0,"total_bars":0}
    for i in range(50, len(closes)-6):
        skipped["total_bars"]+=1
        window=closes[:i]
        w15=closes[max(0,i-60):i:3]
        price=closes[i]
        atr_v=atr14(window)
        dt=candles_5m[i]["dt"]
        try: hour=int(dt[11:13])
        except: hour=12
        if not ((3<=hour<11) or (12<=hour<17)):
            skipped["no_session"]+=1; continue
        if detect_chop(window[-20:]):
            skipped["chop"]+=1; continue
        bias=get_bias(w15 if len(w15)>=22 else window)
        if not bias: skipped["no_bias"]+=1; continue
        swept,sweep_lvl,rej=detect_sweep(window,bias)
        if not swept: skipped["no_sweep"]+=1; continue
        bos,bos_type=detect_bos(window,bias)
        if not bos: skipped["no_bos"]+=1; continue
        score=65
        fvg,fvg_lo,fvg_hi=detect_fvg(window,bias)
        if fvg: score+=15
        if fvg and fvg_lo<=price<=fvg_hi: score+=5
        if bias==get_bias(window): score+=10
        r=rsi14(window)
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
        rr=round(tp_mult,1)
        future=closes[i+1:i+7]
        if not future: continue
        if is_buy:
            hit_tp=any(p>=tp for p in future)
            hit_sl=any(p<=sl for p in future)
        else:
            hit_tp=any(p<=tp for p in future)
            hit_sl=any(p>=sl for p in future)
        if hit_tp and hit_sl:
            tp_idx=next((j for j,p in enumerate(future) if (p>=tp if is_buy else p<=tp)),999)
            sl_idx=next((j for j,p in enumerate(future) if (p<=sl if is_buy else p>=sl)),999)
            won=tp_idx<sl_idx
        elif hit_tp: won=True
        elif hit_sl: won=False
        else: won=False
        pnl_r=rr if won else -1.0
        sess="Londres" if 3<=hour<11 else "New York"
        trades.append({"dt":dt,"hour":hour,"sess":sess,
            "dir":"BUY" if is_buy else "SELL",
            "score":score,"entry":round(price,2),
            "tp":round(tp,2),"sl":round(sl,2),"rr":rr,
            "won":won,"pnl_r":pnl_r,"bos":bos_type,"rsi":round(r,1),"fvg":fvg})
        if len(trades)>=300: break
    return trades, skipped

def calc_stats(trades, skipped):
    if not trades: return {"error": "Sin trades"}
    n=len(trades)
    wins=[t for t in trades if t["won"]]
    losses=[t for t in trades if not t["won"]]
    wr=round(len(wins)/n*100,1)
    total_r=round(sum(t["pnl_r"] for t in trades),2)
    avg_win=round(sum(t["rr"] for t in wins)/len(wins),2) if wins else 0
    gw=sum(t["pnl_r"] for t in wins) if wins else 0
    gl=abs(sum(t["pnl_r"] for t in losses)) if losses else 0.001
    pf=round(gw/gl,2)
    expectancy=round(total_r/n,3)
    eq=[0.0]
    for t in trades: eq.append(round(eq[-1]+t["pnl_r"],2))
    peak=eq[0]; max_dd=0
    for e in eq:
        if e>peak: peak=e
        dd=peak-e
        if dd>max_dd: max_dd=dd
    max_dd=round(max_dd,2)
    rets=[t["pnl_r"] for t in trades]
    mean_r=sum(rets)/len(rets)
    std_r=math.sqrt(sum((r-mean_r)**2 for r in rets)/len(rets)) if len(rets)>1 else 0.01
    sharpe=round(mean_r/std_r*math.sqrt(252),2) if std_r>0 else 0
    by_sess={}
    for t in trades:
        s=t["sess"]
        if s not in by_sess: by_sess[s]={"n":0,"w":0,"r":0.0}
        by_sess[s]["n"]+=1; by_sess[s]["w"]+=int(t["won"]); by_sess[s]["r"]+=t["pnl_r"]
    for s in by_sess:
        by_sess[s]["wr"]=round(by_sess[s]["w"]/by_sess[s]["n"]*100,1)
        by_sess[s]["r"]=round(by_sess[s]["r"],2)
    by_hour={}
    for t in trades:
        h=t["hour"]
        if h not in by_hour: by_hour[h]={"n":0,"w":0}
        by_hour[h]["n"]+=1; by_hour[h]["w"]+=int(t["won"])
    for h in by_hour:
        by_hour[h]["wr"]=round(by_hour[h]["w"]/by_hour[h]["n"]*100,1)
    snipers=[t for t in trades if t["score"]>=85]
    normals=[t for t in trades if 70<=t["score"]<85]
    sn_wr=round(len([t for t in snipers if t["won"]])/max(len(snipers),1)*100,1)
    no_wr=round(len([t for t in normals if t["won"]])/max(len(normals),1)*100,1)
    buys=[t for t in trades if t["dir"]=="BUY"]
    sells=[t for t in trades if t["dir"]=="SELL"]
    buy_wr=round(len([t for t in buys if t["won"]])/max(len(buys),1)*100,1)
    sell_wr=round(len([t for t in sells if t["won"]])/max(len(sells),1)*100,1)
    mw=ml=cw=cl=0; ct=None
    for t in trades:
        if t["won"]:
            cw=cw+1 if ct=="w" else 1; cl=0; ct="w"; mw=max(mw,cw)
        else:
            cl=cl+1 if ct=="l" else 1; cw=0; ct="l"; ml=max(ml,cl)
    total_bars=skipped.get("total_bars",1)
    filter_rate=round((total_bars-n)/max(total_bars,1)*100,1)
    return {"n":n,"wins":len(wins),"losses":len(losses),
        "wr":wr,"total_r":total_r,"avg_win":avg_win,"pf":pf,
        "expectancy":expectancy,"max_dd":max_dd,"sharpe":sharpe,
        "by_sess":by_sess,"by_hour":by_hour,
        "snipers":len(snipers),"normals":len(normals),
        "sn_wr":sn_wr,"no_wr":no_wr,
        "buy_wr":buy_wr,"sell_wr":sell_wr,
        "buys":len(buys),"sells":len(sells),
        "max_win_streak":mw,"max_loss_streak":ml,
        "eq":eq,"skipped":skipped,"filter_rate":filter_rate}

def build_html(stats, trades, meta):
    eq = stats["eq"]
    W, H = 900, 220
    mn = min(eq)-0.5; mx = max(eq)+0.5
    def px(i): return (i/max(len(eq)-1,1))*W
    def py(v): return H-((v-mn)/(mx-mn+0.01))*(H-30)-10
    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i,v in enumerate(eq))
    zero_y = py(0)
    eq_color = "#00CC88" if eq[-1]>=0 else "#CC3344"
    fill_pts = f"0,{H} " + pts + f" {W},{H}"
    eq_svg = f"""<svg width="100%" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
  <defs><linearGradient id="eqg" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="{eq_color}" stop-opacity="0.25"/>
    <stop offset="100%" stop-color="{eq_color}" stop-opacity="0.02"/>
  </linearGradient></defs>
  <rect width="{W}" height="{H}" fill="#0A0800"/>
  <polygon points="{fill_pts}" fill="url(#eqg)"/>
  <line x1="0" y1="{zero_y:.1f}" x2="{W}" y2="{zero_y:.1f}" stroke="rgba(201,168,76,0.4)" stroke-width="1" stroke-dasharray="6,3"/>
  <polyline points="{pts}" fill="none" stroke="{eq_color}" stroke-width="2.5" stroke-linejoin="round"/>
  <circle cx="{px(len(eq)-1):.1f}" cy="{py(eq[-1]):.1f}" r="4" fill="{eq_color}"/>
</svg>"""

    sess_rows = ""
    for s, sv in stats["by_sess"].items():
        wrc = "#00CC88" if sv["wr"]>=55 else "#CC3344"
        rc  = "#00CC88" if sv["r"]>=0 else "#CC3344"
        sess_rows += f"<tr><td>{s}</td><td>{sv['n']}</td><td style='color:{wrc}'>{sv['wr']}%</td><td style='color:{rc}'>{'+'if sv['r']>=0 else''}{sv['r']}R</td></tr>"

    hour_rows = ""
    for h in sorted(stats["by_hour"].keys()):
        hv = stats["by_hour"][h]
        wrc = "#00CC88" if hv["wr"]>=55 else "#CC3344"
        flag = "GB" if 3<=h<11 else ("US" if 12<=h<17 else "")
        hour_rows += f"<tr><td>{h:02d}:00 {flag}</td><td>{hv['n']}</td><td style='color:{wrc}'>{hv['wr']}%</td></tr>"

    trade_rows = ""
    for t in reversed(trades[-60:]):
        c  = "#00CC88" if t["won"] else "#CC3344"
        d  = "BUY" if t["dir"]=="BUY" else "SELL"
        sc = "#F0D080" if t["score"]>=85 else "#5A4820"
        pl = f"+{t['pnl_r']}R" if t["pnl_r"]>=0 else f"{t['pnl_r']}R"
        dc = "#00CC88" if d=="BUY" else "#CC3344"
        res = "WIN" if t["won"] else "LOSS"
        trade_rows += (
            f"<tr>"
            f"<td style='color:#5A4820'>{t['dt'][5:16]}</td>"
            f"<td style='color:{dc}'>{d}</td>"
            f"<td>${t['entry']}</td>"
            f"<td style='color:#00CC88'>${t['tp']}</td>"
            f"<td style='color:#CC3344'>${t['sl']}</td>"
            f"<td>{t['rr']}:1</td>"
            f"<td style='color:{sc}'>{t['score']}</td>"
            f"<td style='color:{c}'>{res}</td>"
            f"<td style='color:{c}'>{pl}</td>"
            f"<td style='color:#5A4820'>{t['sess']}</td>"
            f"<td style='color:#5A4820'>{t['bos']}</td>"
            f"</tr>"
        )

    sk = stats["skipped"]
    filter_rows = ""
    labels = {"no_session":"Fuera de sesion","chop":"Mercado lateral","no_bias":"Sin bias EMA",
              "no_sweep":"Sin Sweep","no_bos":"Sin BOS","low_score":"Score bajo"}
    for k, lbl in labels.items():
        v = sk.get(k,0)
        pct = round(v/max(sk.get("total_bars",1),1)*100,1)
        filter_rows += (f"<div style='display:flex;justify-content:space-between;"
                        f"padding:6px 0;border-bottom:1px solid rgba(201,168,76,0.05)'>"
                        f"<span style='color:#5A4820'>{lbl}</span>"
                        f"<span>{v} <span style='color:#3D3010'>({pct}%)</span></span></div>")

    pfc = "#00CC88" if stats["pf"]>=1.5 else "#CC3344"
    wrc = "#00CC88" if stats["wr"]>=55 else "#CC3344"
    rc  = "#00CC88" if stats["total_r"]>=0 else "#CC3344"
    ddc = "#00CC88" if stats["max_dd"]<=3 else "#CC3344"
    shc = "#00CC88" if stats["sharpe"]>=1.0 else "#CC3344"
    exc = "#00CC88" if stats["expectancy"]>=0.3 else "#CC3344"
    profitable = stats["total_r"] > 0

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AURUM SCALP v2 - Backtest Report</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=IBM+Plex+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root{{--bg:#080600;--bg2:#0C0A02;--gold:#C9A84C;--gold2:#F0D080;--green:#00CC88;--red:#CC3344;--dim:#5A4820;--dim2:#3D3010;--border:rgba(201,168,76,0.1)}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:radial-gradient(ellipse at 50% 0%,#0F0C02 0%,#080600 50%,#000 100%);color:#E8E0C0;font-family:"IBM Plex Mono",monospace;min-height:100vh;padding:40px 32px}}
h1{{font-family:"Playfair Display",serif;font-size:32px;font-weight:700;letter-spacing:6px;background:linear-gradient(135deg,#8B6914,#F0D080,#C9A84C);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.sub{{font-size:9px;letter-spacing:3px;color:var(--dim);margin:6px 0 8px}}
.meta{{font-size:8px;color:var(--dim2);letter-spacing:1px;margin-bottom:32px;padding:10px 14px;border-left:2px solid var(--gold);background:rgba(201,168,76,0.03)}}
.g4{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}}
.g3{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
.card{{background:var(--bg2);border:1px solid var(--border);padding:20px;position:relative;overflow:hidden}}
.card::before{{content:"";position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--gold),transparent);opacity:.25}}
.cl{{font-size:8px;letter-spacing:3px;color:var(--dim);margin-bottom:10px}}
.cv{{font-family:"Playfair Display",serif;font-size:34px;font-weight:700;line-height:1}}
.cs{{font-size:8px;color:var(--dim2);margin-top:5px}}
.sec{{font-size:8px;letter-spacing:4px;color:var(--dim);margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--border)}}
table{{width:100%;border-collapse:collapse;font-size:9px}}
th{{font-size:7px;letter-spacing:2px;color:var(--dim);padding:9px 10px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}}
td{{padding:7px 10px;border-bottom:1px solid rgba(201,168,76,0.04);white-space:nowrap}}
.panel{{background:var(--bg2);border:1px solid var(--border);padding:18px}}
.badge{{font-size:8px;letter-spacing:2px;padding:3px 12px;border:1px solid}}
::-webkit-scrollbar{{width:4px}}::-webkit-scrollbar-track{{background:var(--bg)}}
::-webkit-scrollbar-thumb{{background:var(--dim);border-radius:2px}}
@media(max-width:768px){{.g4,.g3{{grid-template-columns:1fr 1fr}}.g2{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<h1>AURUM SCALP v2</h1>
<div class="sub">BACKTEST REPORT · XAUUSD · DATOS REALES TWELVE DATA · 5M</div>
<div class="meta">
  Periodo: {meta["from"]} &rarr; {meta["to"]} &nbsp;&middot;&nbsp;
  {meta["candles"]} velas 5M &nbsp;&middot;&nbsp;
  {stats["n"]} senales generadas &nbsp;&middot;&nbsp;
  {stats["filter_rate"]}% barras filtradas &nbsp;&middot;&nbsp;
  {meta["generated"]} UTC
  &nbsp;&nbsp;
  <span class="badge" style="color:{'#00CC88' if profitable else '#CC3344'};border-color:{'rgba(0,204,136,.3)' if profitable else 'rgba(204,51,68,.3)'}">
    {'RENTABLE' if profitable else 'PERDIDA'}
  </span>
</div>

<div class="g4">
  <div class="card"><div class="cl">WIN RATE</div><div class="cv" style="color:{wrc}">{stats["wr"]}%</div><div class="cs">{stats["wins"]}W &middot; {stats["losses"]}L</div></div>
  <div class="card"><div class="cl">PROFIT FACTOR</div><div class="cv" style="color:{pfc}">{stats["pf"]}</div><div class="cs">objetivo &ge; 1.5</div></div>
  <div class="card"><div class="cl">TOTAL R</div><div class="cv" style="color:{rc}">{'+'if stats['total_r']>=0 else''}{stats["total_r"]}R</div><div class="cs">{stats["n"]} trades</div></div>
  <div class="card"><div class="cl">EXPECTANCY</div><div class="cv" style="color:{exc}">{'+'if stats['expectancy']>=0 else''}{stats["expectancy"]}R</div><div class="cs">por trade</div></div>
</div>

<div class="g4">
  <div class="card"><div class="cl">MAX DRAWDOWN</div><div class="cv" style="color:{ddc}">{stats["max_dd"]}R</div><div class="cs">consecutivo</div></div>
  <div class="card"><div class="cl">SHARPE RATIO</div><div class="cv" style="color:{shc}">{stats["sharpe"]}</div><div class="cs">objetivo &ge; 1.0</div></div>
  <div class="card"><div class="cl">AVG WIN / LOSS</div><div class="cv" style="font-size:22px"><span style="color:#00CC88">+{stats["avg_win"]}R</span> <span style="color:#5A4820">/</span> <span style="color:#CC3344">-1.0R</span></div><div class="cs">ratio {round(stats["avg_win"],1)}:1</div></div>
  <div class="card"><div class="cl">STREAKS MAX</div><div class="cv" style="font-size:22px"><span style="color:#00CC88">{stats["max_win_streak"]}W</span> <span style="color:#5A4820">/</span> <span style="color:#CC3344">{stats["max_loss_streak"]}L</span></div><div class="cs">seguidas</div></div>
</div>

<div class="panel" style="margin-bottom:20px">
  <div class="sec">CURVA DE EQUITY (en R)</div>
  {eq_svg}
  <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:8px;color:var(--dim)">
    <span>Inicio: 0.0R</span><span>Final: {'+'if stats['eq'][-1]>=0 else''}{stats["eq"][-1]}R</span>
  </div>
</div>

<div class="g2">
  <div>
    <div class="sec">POR SESION</div>
    <div class="panel"><table><thead><tr><th>SESION</th><th>TRADES</th><th>WIN RATE</th><th>TOTAL R</th></tr></thead><tbody>{sess_rows}</tbody></table></div>
  </div>
  <div>
    <div class="sec">POR HORA (UTC)</div>
    <div class="panel"><table><thead><tr><th>HORA</th><th>TRADES</th><th>WIN RATE</th></tr></thead><tbody>{hour_rows}</tbody></table></div>
  </div>
</div>

<div class="g2">
  <div>
    <div class="sec">SCORE TIERS</div>
    <div class="panel" style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
      <div style="text-align:center;padding:16px;border:1px solid var(--border)">
        <div style="font-family:'Playfair Display',serif;font-size:28px;color:var(--gold)">{stats["snipers"]}</div>
        <div style="font-size:7px;letter-spacing:2px;color:var(--dim);margin-top:4px">SNIPER &ge;85</div>
        <div style="font-size:12px;margin-top:8px;color:{'#00CC88' if stats['sn_wr']>=55 else '#CC3344'}">WR {stats["sn_wr"]}%</div>
      </div>
      <div style="text-align:center;padding:16px;border:1px solid var(--border)">
        <div style="font-family:'Playfair Display',serif;font-size:28px;color:var(--dim)">{stats["normals"]}</div>
        <div style="font-size:7px;letter-spacing:2px;color:var(--dim);margin-top:4px">NORMAL 70-84</div>
        <div style="font-size:12px;margin-top:8px;color:{'#00CC88' if stats['no_wr']>=55 else '#CC3344'}">WR {stats["no_wr"]}%</div>
      </div>
    </div>
  </div>
  <div>
    <div class="sec">BUY vs SELL</div>
    <div class="panel" style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
      <div style="text-align:center;padding:16px;border:1px solid var(--border)">
        <div style="font-family:'Playfair Display',serif;font-size:28px;color:#00CC88">{stats["buys"]}</div>
        <div style="font-size:7px;letter-spacing:2px;color:var(--dim);margin-top:4px">BUY</div>
        <div style="font-size:12px;margin-top:8px;color:{'#00CC88' if stats['buy_wr']>=55 else '#CC3344'}">WR {stats["buy_wr"]}%</div>
      </div>
      <div style="text-align:center;padding:16px;border:1px solid var(--border)">
        <div style="font-family:'Playfair Display',serif;font-size:28px;color:#CC3344">{stats["sells"]}</div>
        <div style="font-size:7px;letter-spacing:2px;color:var(--dim);margin-top:4px">SELL</div>
        <div style="font-size:12px;margin-top:8px;color:{'#00CC88' if stats['sell_wr']>=55 else '#CC3344'}">WR {stats["sell_wr"]}%</div>
      </div>
    </div>
  </div>
</div>

<div style="margin-bottom:20px">
  <div class="sec">FILTROS — BARRAS DESCARTADAS</div>
  <div class="panel">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0 40px">{filter_rows}</div>
    <div style="margin-top:12px;font-size:8px;color:var(--dim2)">
      Total barras: {sk.get("total_bars",0)} &middot; Senales: {stats["n"]} &middot; Filtradas: {stats["filter_rate"]}%
    </div>
  </div>
</div>

<div>
  <div class="sec">ULTIMOS 60 TRADES</div>
  <div style="overflow-x:auto" class="panel">
    <table>
      <thead><tr><th>DATETIME</th><th>DIR</th><th>ENTRY</th><th>TP</th><th>SL</th><th>RR</th><th>SCORE</th><th>RESULTADO</th><th>P&L</th><th>SESION</th><th>BOS</th></tr></thead>
      <tbody>{trade_rows}</tbody>
    </table>
  </div>
</div>

<div style="margin-top:40px;font-size:8px;color:var(--dim2);line-height:2;padding:16px;border:1px solid var(--border)">
  Backtest basado en datos historicos reales XAUUSD 5M (Twelve Data). Motor: SCALP v2.
  Sweep + BOS obligatorios. FVG zona entrada. EMA + RSI soft filters.
  Sin slippage, sin spread, sin comisiones. Resultados pasados no garantizan resultados futuros.
</div>
</body></html>"""
    return html


def main():
    print("=" * 54)
    print("  AURUM SCALP v2 -- BACKTESTING REAL")
    print("  XAUUSD 5M | Twelve Data | Motor SCALP v2")
    print("=" * 54)

    print("\n[1/4] Descargando datos historicos reales...")
    candles_5m = fetch_candles("5min", 500)

    print("\n[2/4] Corriendo motor SCALP v2...")
    trades, skipped = run_scalp_v2(candles_5m)
    print(f"  {len(trades)} senales generadas")

    print("\n[3/4] Calculando estadisticas...")
    stats = calc_stats(trades, skipped)
    if "error" in stats:
        print(f"  ERROR: {stats['error']}")
        return

    print(f"  Win Rate:      {stats['wr']}%")
    print(f"  Profit Factor: {stats['pf']}")
    print(f"  Total R:       {stats['total_r']}R")
    print(f"  Expectancy:    {stats['expectancy']}R/trade")
    print(f"  Max DD:        {stats['max_dd']}R")
    print(f"  Sharpe:        {stats['sharpe']}")
    print(f"  Por sesion:")
    for s, sv in stats["by_sess"].items():
        print(f"    {s}: {sv['n']} trades | WR {sv['wr']}% | {sv['r']}R")

    meta = {
        "from":      candles_5m[0]["dt"][:16],
        "to":        candles_5m[-1]["dt"][:16],
        "candles":   len(candles_5m),
        "generated": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
    }

    print("\n[4/4] Generando reporte HTML...")
    html = build_html(stats, trades, meta)

    with open("aurum_backtest_report.html", "w", encoding="utf-8") as f:
        f.write(html)

    with open("aurum_backtest_data.json", "w") as f:
        export = {k:v for k,v in stats.items() if k not in ["eq","by_hour","skipped"]}
        json.dump({"stats": export, "trades": trades[-100:], "meta": meta}, f, indent=2)

    print("\n  LISTO:")
    print("  -> aurum_backtest_report.html  (abrir en browser)")
    print("  -> aurum_backtest_data.json    (datos crudos)")
    print("-" * 54)


if __name__ == "__main__":
    main()
