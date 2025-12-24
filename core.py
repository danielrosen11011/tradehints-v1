# main.py · AI Stock Advisor  (clear $ orders, S&P-500 adds that actually BUY)
# multi-ticker, earnings/headlines, momentum, smart sizing, sector imbalance

from __future__ import annotations
import os, pathlib, warnings, re, math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json  # <-- added

from dotenv import load_dotenv
load_dotenv()  # ensure env vars are loaded before OpenAI init

try:
    from openai import OpenAI
    _OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    _OPENAI = OpenAI()
    _HAS_OPENAI = True
except Exception as e:
    print("⚠️ OpenAI disabled:", e)
    _HAS_OPENAI = False


# ── hush noisy warnings ───────────────────────────────────
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=UserWarning,  module="statsmodels")

# ── optional SARIMAX forecast ─────────────────────────────
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _HAS_STATS = True
except ImportError:
    _HAS_STATS = False

# ── optional OpenAI (for polished natural-language blurbs) ─────────
try:
    from openai import OpenAI
    _OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    _OPENAI = OpenAI()
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

analyzer = SentimentIntensityAnalyzer()
DATA_DIR = pathlib.Path("prediction_logs"); DATA_DIR.mkdir(exist_ok=True)

# ── Finnhub fallback (earnings / news) ────────────────────
try:
    import finnhub
    load_dotenv()
    _FINN = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))
    _HAS_FINN = True
except Exception:
    _HAS_FINN = False

# compact universe for the "watch list"
SCAN_UNIVERSE = [
    "MMM","AOS","ABT","ABBV","ACN","ADM","ADBE","ADP","AAP",
    "AES","AFL","A","APD","AKAM","ALK","ALB","ARE","ALL","GOOGL",
    "GOOG","MO","AMZN","AMCR"
]


# ── helpers ───────────────────────────────────────────────
def ask(msg:str)->str:      return input(f"{msg}\n> ").strip()
def dollar(txt:str)->float: return float(txt.replace("$","").replace(",","") or 0)
def _utc_date():            return datetime.now(timezone.utc).date()
def _first_col(s):          return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
def _fmt(x:float)->str:     return f"${x:,.2f}"

def pct(a: pd.Series, n: int) -> float:
    if len(a) <= n or a.iloc[-n-1] == 0: return 0.0
    return float((a.iloc[-1] / a.iloc[-n-1] - 1) * 100.0)

# ── user preferences ─────────────────────────────────────
def get_user_preferences()->dict:
    print("\n📋  Investment preferences\n" + "—"*72)
    risk = ask("Risk tolerance (low / moderate / high):").lower()
    secs = [s.strip() for s in ask("Preferred sectors (comma–separated):").split(",")]
    cap  = dollar(ask("Available **new** capital to invest ($):"))
    days = int(ask("Time horizon **in days**:"))
    port  = [normalize_ticker(t) for t in ask("Current holdings (tickers, comma–sep):").split(",") if t.strip()]

    pos={}
    if port:
        print("\n💰  Current $ value in each holding")
        for t in port:
            pos[t]=dollar(ask(f"{t} amount ($):"))

    analyse=[normalize_ticker(t) for t in ask(
        "Tickers to analyse now (comma–sep, blank = portfolio only):"
    ).split(",") if t.strip()] or port

    open_sp = ask("Open to **new** S&P 500 ideas you didn't list? (y/n):").lower().startswith("y")
    max_new = 0
    if open_sp:
        try:
            max_new = int(ask("Max number of new ideas to add (e.g., 3):"))
        except Exception:
            max_new = 3

    return dict(
        risk=risk, sectors=secs, capital=cap, horizon_days=days,
        portfolio=port, positions=pos, tickers=analyse,
        open_sp500=open_sp, max_new=max_new
    )

# ── price, technicals & momentum ─────────────────────────
def fetch_price_data(t:str, period:str="1y")->pd.DataFrame:
    for sym in (t, t.replace("-", "."), t.replace(".", "-")):
        try:
            df=yf.download(sym, period=period, progress=False, auto_adjust=True, threads=False)
            if not df.empty: return df
        except Exception:
            continue
    raise ValueError(f"No data for {t}")

def technicals(df:pd.DataFrame)->dict:
    c=_first_col(df["Close"])
    rsi  = ta.momentum.RSIIndicator(c).rsi().iloc[-1]
    macd = ta.trend.MACD(c).macd().iloc[-1]
    bb   = ta.volatility.BollingerBands(c)
    return {
        "price":round(c.iloc[-1],2),
        "rsi":round(rsi,2),
        "macd":round(macd,2),
        "bb_low":round(bb.bollinger_lband().iloc[-1],2),
        "bb_high":round(bb.bollinger_hband().iloc[-1],2),
    }

def momentum_snapshot(df: pd.DataFrame) -> dict:
    c = _first_col(df["Close"])
    return {
        "d1": round(pct(c, 1), 2),
        "w1": round(pct(c, 5), 2),
        "m1": round(pct(c, 21), 2),
        "m3": round(pct(c, 63), 2),
        "y1": round(pct(c, 252), 2),
    }

# ── earnings helpers ─────────────────────────────────────

def finnhub_earnings(t:str, look:int=5):
    if not _HAS_FINN: return False, "N/A", None
    try:
        res=_FINN.earnings_calendar(_from=_utc_date().isoformat(),
                                    to=(_utc_date()+timedelta(days=look)).isoformat(),
                                    symbol=t)
        it=res.get("earningsCalendar",[])
        if it:
            return True, f"{it[0]['date']} (EPS est {it[0].get('epsEstimate','N/A')})", it[0].get('epsEstimate')
        return False, "N/A", None
    except Exception:
        return False, "N/A", None

def last_surprises(t:str,n:int=4)->list[float]:
    try:
        q=yf.Ticker(t).quarterly_earnings
        if q is None or q.empty: return []
        eps=q["Earnings"].astype(float).dropna()
        return eps.pct_change().dropna().tail(n).tolist()
    except Exception: return []

# --- robust earnings date lookup ---------------------------------
def next_earnings_date(t: str):
    """Return the next earnings *date* for ticker or None if unknown."""
    t = normalize_ticker(t)
    today = _utc_date()

    # 1) yfinance get_earnings_dates: pick the nearest FUTURE date
    try:
        df = yf.Ticker(t).get_earnings_dates(limit=12)
        if df is not None and not df.empty:
            if "Earnings Date" in df.columns:
                raw_dates = df["Earnings Date"].tolist()
            else:
                raw_dates = list(df.index)
            dates = [pd.to_datetime(x).date() for x in raw_dates if pd.notna(x)]
            future = [d for d in dates if d >= today]
            if future:
                return min(future)
    except Exception:
        pass

    # 2) yfinance calendar (may be a scalar or a (start,end) pair)
    try:
        cal = yf.Ticker(t).calendar
        if cal is not None and not cal.empty and "Earnings Date" in cal.index:
            raw = cal.loc["Earnings Date"][0]
            # handle ranges like (start,end)
            if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) > 0:
                raw = raw[0]
            dt = pd.to_datetime(raw).date()
            if dt >= today:
                return dt
    except Exception:
        pass

    # 3) Finnhub fallback (nearest future within 30 days)
    if _HAS_FINN:
        try:
            res = _FINN.earnings_calendar(
                _from=today.isoformat(),
                to=(today + timedelta(days=30)).isoformat(),
                symbol=t,
            )
            it = res.get("earningsCalendar", [])
            if it:
                return pd.to_datetime(it[0]["date"]).date()
        except Exception:
            pass

    return None



# --- earnings signal built on top of the robust lookup -----------
def earnings_signal(t: str, look: int = 7) -> dict:
    """
    Flag if earnings are within the next `look` days.
    Confidence ramps up as the date approaches.
    """
    dt = next_earnings_date(t)
    if not dt:
        return dict(flag=False, info="N/A", score=0.0, conf=0.0)

    today = _utc_date()
    if not (today <= dt <= today + timedelta(days=look)):
        # Return the date (informational) but don't flag it for action
        return dict(flag=False, info=dt.isoformat(), score=0.0, conf=0.0)

    # keep your surprise logic
    surprises = last_surprises(t)
    sur = float(np.mean(surprises)) if surprises else 0.0
    days_left = max((dt - today).days, 0)

    return dict(
        flag=True,
        info=dt.isoformat(),
        score=float(np.tanh(sur * 4)),
        conf=float(max(0.0, 1 - days_left / look)),
    )

# ── headlines & sentiment (incl. biotech cues) ───────────
POS=r"(beats|record|acquisition|partnership|upgrade|surge|strong demand|ai|fda approval|phase (iii|3)|trial success)"
NEG=r"(misses|lawsuit|recall|downgrade|investigation|fire|bankruptcy|layoff|hack|data breach|trial failure|fda rejection|crl)"

def fetch_headlines_yf(sym: str, limit: int = 15) -> list[str]:
    sym = safe_news_ticker(normalize_ticker(sym))
    if not sym:
        return []
    try:
        items = yf.Ticker(sym).news or []
    except Exception:
        return []
    heads = []
    for it in items[:limit]:
        h = (it.get("headline") or it.get("title") or "").strip()
        if not h:                      # <- drop empties
            continue
        if len(h) < 6:                 # <- drop junky 1–2 word “headlines”
            continue
        heads.append(h)
    # de-dupe while preserving order
    return list(dict.fromkeys(heads))


def finnhub_news(t:str, days:int=7, limit:int=20)->list[str]:
    if not _HAS_FINN: return []
    try:
        frm=(_utc_date()-timedelta(days=days)).isoformat()
        news=_FINN.company_news(t, _from=frm, to=_utc_date().isoformat())[:limit]
        return [n["headline"] for n in news]
    except Exception: return []

def sentiment_score(heads:list[str])->float:
    return round(float(np.mean([analyzer.polarity_scores(h)["compound"] for h in heads])),3) if heads else 0.0

def news_signal(heads:list[str])->dict:
    if not heads:
        return dict(score=0.0,conf=0.0,tag=None)
    sent=sentiment_score(heads)
    pos=sum(bool(re.search(POS,h.lower())) for h in heads)
    neg=sum(bool(re.search(NEG,h.lower())) for h in heads)
    kw=(pos-neg)/max(len(heads),1)
    return dict(score=float(0.7*sent+0.3*kw),
                conf=float(min(1,len(heads)/20)),
                tag="positive_news" if pos>neg else "negative_news" if neg>pos else None)

# ── anomaly tag ──────────────────────────────────────────
def detect_anomaly(df,t,heads,earn)->str:
    flags=[]
    close=_first_col(df["Close"]); vol=_first_col(df["Volume"])
    z=(close-close.rolling(20).mean())/close.rolling(20).std()
    if abs(float(z.iloc[-1]))>3: flags.append("price_outlier")
    if float(vol.iloc[-1])>float(vol.rolling(20).mean().iloc[-1])*3: flags.append("volume_spike")
    if earn["flag"]: flags.append(f"earnings {earn['info']}")
    tag=news_signal(heads)["tag"]; flags.append(tag) if tag else None
    return ", ".join(flags) or "normal"

# ── forecast & Kelly sizing ──────────────────────────────
def forecast_price(df,days:int)->float:
    close=_first_col(df["Close"])
    if _HAS_STATS and len(close)>30:
        try:
            model=SARIMAX(close,order=(1,1,1),seasonal_order=(0,0,0,0)).fit(disp=False)
            return float(model.forecast(days).mean())
        except Exception: pass
    return float(close.iloc[-1])

def kelly(bank:float, win:float=0.55, r:float=1)->float:
    edge=win*(r+1)-1
    return round(max(0,min(edge/r if r else 0,1))*bank,2)

def log_forecast(t:str,val:float):
    p=DATA_DIR/f"{_utc_date()}_preds.csv"
    with p.open("a") as f:
        if p.stat().st_size==0: f.write("ts,ticker,forecast\n")
        f.write(f"{datetime.now(timezone.utc)},{t},{val}\n")

# ── portfolio utilities ──────────────────────────────────
def current_weights(pos:dict)->dict:
    tot=sum(pos.values()) or 1.0
    return {t:v/tot for t,v in pos.items()}

def smart_target_weights(tickers:list[str], risk:str)->dict:
    if not tickers: return {}
    cap=dict(low=.10,moderate=.20,high=.35).get(risk,.20)
    prices=yf.download(tickers,period="18mo",progress=False,auto_adjust=True, threads=False)["Close"]
    if isinstance(prices,pd.Series): prices=prices.to_frame(tickers[0])
    rets = prices.pct_change(fill_method=None).dropna()
    iv=(1/rets.std()).replace(np.inf,0)
    raw=(iv/iv.sum()).clip(upper=cap); raw/=raw.sum()
    return raw.to_dict()

def rebalance(cur:dict,tgt:dict,total:float)->list[dict]:
    out=[]
    for t in sorted(set(cur)|set(tgt)):
        diff=tgt.get(t,0)-cur.get(t,0)
        if abs(diff)<1e-3: continue
        out.append({"ticker":t,"cur%":round(cur.get(t,0)*100,2),
                    "tgt%":round(tgt.get(t,0)*100,2),
                    "shift$":round(diff*total,2)})
    return out

def sector_imbalances(cur_w:dict)->str:
    if not cur_w: return "No sector exposure (no current holdings)."
    out=[]
    for t in list(cur_w.keys()):
        try:
            sec=yf.Ticker(t).info.get("sector","Unknown") or "Unknown"
        except Exception:
            sec="Unknown"
        out.append((sec,cur_w[t]))
    df=pd.DataFrame(out,columns=["sector","w"]).groupby("sector")["w"].sum()
    lines=[f"{s}: {round(float(w*100),1)}%" for s,w in df.sort_values(ascending=False).items()]
    msg="; ".join(lines)
    flag = any(w>0.45 for w in df.values)
    return f"{msg} {'(Over-concentrated!)' if flag else ''}"

# ── S&P 500 helpers (new-idea selection & backfill) ──────
def load_sp500_tickers()->list[str]:
    try:
        tables=pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df=tables[0]
        col=[c for c in df.columns if "Symbol" in str(c)][0]
        syms=df[col].astype(str).str.replace(".","-",regex=False).tolist()
        return syms
    except Exception:
        # offline fallback; still lets the feature work
        return [
    "MMM","QQQ", "AOS","ABT","ABBV","ACN","ATVI","ADM","ADBE","ADP","AAP",
    "AES","AFL","A","APD","AKAM","ALK","ALB","ARE","ALL","GOOGL",
    "GOOG","MO","AMZN","AMCR","AEE","AAL","AEP","AXP","AZO","AVB",
    "AVY","BKR","BALL","BAC","BK","BAX","BDX","BRK.B","BBY","BIIB",
    "BLK","BA","BKNG","BWA","BXP","BSX","BMY","AVGO","BR","BEN",
    "CHRW","CDNS","CZR","CPB","COF","CAH","KMX","CCL","CAT","CBOE",
    "CBRE","CDW","CE","CNC","CNP","CDAY","CF","CFG","CHD","CHTR",
    "CVX","CMG","CB","CI","CINF","CTAS","CSCO","C","CLX","CME","CMS",
    "KO","CTSH","CL","CMCSA","CMA","CAG","COP","ED","STZ","COO",
    "CPRT","GLW","CTVA","COST","CCI","CSX","CMI","CVS","DHI","DHR",
    "DRI","DVA","DE","DAL","XRAY","DVN","FANG","DLR","DFS","DG",
    "DOW","DTE","DUK","DD","DXC","EMN","ETN","EBAY","ECL","EIX","EW",
    "EA","EMR","ETR","EOG","EFX","EQIX","EQR","ESS","EL","EVRG","ES",
    "RE","EXC","EXPE","EXPD","EXR","XOM","FFIV","FAST","FRT","FDX",
    "FIS","FITB","FLS","FMC","F","FTNT","FTV","FBHS","FCX","GPS",
    "GRMN","IT","GNRC","GD","GE","GIS","GM","GPC","GILD","GL","GPN",
    "GS","GWW","HAL","HBI","HOG","HIG","HAS","HCA","PEAK","HSIC",
    "HSY","HES","HPE","HLT","HWM","HOLX","HD","HON","HRL","HST",
    "HPQ","HUM","HBAN","HII","IEX","IDXX","ITW","ILMN","INCY","IR",
    "INTC","ICE","IBM","IFF","IP","IPG","IQV","IRM","JKHY","J","JBHT",
    "SJM","JNJ","JCI","JPM","JNPR","K","KEY","KEYS","KMB","KIM",
    "KMI","KLAC","KSS","KHC","KR","LHX","LH","LRCX","LVS","LW","L",
    "LEG","LDOS","LEN","LLY","LNC","LYV","LMT","LULU","MRO","MPC",
    "MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK",
    "MDT","MRK","MET","MTD","MGM","MCHP","MU","MSFT","MDLZ","MPWR",
    "MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NOV","NTAP","NFLX",
    "NWL","NEM","NWSA","NWS","NKE","NI","NSC","NTRS","NOC","NCLH",
    "NRG","NUE","NVDA","NXPI","ORLY","OXY","ODFL","OMC","OKE","ORCL",
    "OTIS","PCAR","PKG","PH","PAYX","PAYC","PYPL","PNR","PEP","PFE",
    "PM","PSX","PNC","POOL","PPG","PPL","PFG","PG","PGR","PLD","PRU",
    "PSA","PHM","PVH","QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O",
    "REGN","RMD","RHI","ROK","ROL","ROP","ROST","RCL","SPGI","CRM",
    "SBAC","SLB","STX","SEE","SRE","NOW","SHW","SPG","SWKS","SNA",
    "SO","LUV","SNPS","SBUX","STT","STE","SYK","SYF","SYY","TMUS",
    "TROW","TTWO","TEL","TDY","TFX","TER","TSLA","TXN","TXT","TMO",
    "TJX","TSCO","TT","TDG","TRV","TFC","TGT","ULTA","USB","UAA",
    "UA","UNP","UAL","UNH","UPS","URI","UHS","VLO","VTR","VRSN",
    "VRSK","VZ","VRTX","V","VFC","VTRS","VMC","WAB","WMT","WBD",
    "WM","WAT","WEC","WELL","WST","WDC","WRK","WY","WHR","WMB",
    "WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZION","ZTS"
]


def sp500_news_candidates(exclude:set[str], top_n:int=3, days:int=5)->list[str]:
    tickers = load_sp500_tickers()
    ideas=[]
    for t in tickers:
        if t in exclude: continue
        heads=fetch_headlines_yf(t) or finnhub_news(t,days,10)
        if not heads: continue
        ns=news_signal(heads); es=earnings_signal(t,look=days)
        score = max(0.0, ns.get("score",0))*ns.get("conf",0) * 0.6 \
              + max(0.0, es.get("score",0))*es.get("conf",0) * 0.4
        if score>0: ideas.append((t,score))
    ideas.sort(key=lambda x:x[1], reverse=True)
    return [t for t,_ in ideas[:top_n]]

def sp500_momentum_backfill(exclude: set[str], want: int) -> list[str]:
    """
    Backfill with S&P500 names ranked by 1m momentum (m1), keeping RSI sane.
    Always returns up to `want` tickers by progressively relaxing filters.
    """
    tickers = [t for t in load_sp500_tickers() if t not in exclude]  # no cap
    scored: list[tuple[str, float, float]] = []

    for t in tickers:
        try:
            df = fetch_price_data(t, "6mo")
            c = _first_col(df["Close"])
            m1 = pct(c, 21)  # ~1 month
            rsi = float(ta.momentum.RSIIndicator(c).rsi().iloc[-1])
            if np.isfinite(m1) and np.isfinite(rsi):
                scored.append((t, m1, rsi))
        except Exception:
            continue

    # rank: strong 1m momentum, with a small penalty for very high RSI
    # (penalty kicks in above 70)
    scored.sort(key=lambda x: (x[1] - max(0.0, x[2] - 70.0) / 2.0), reverse=True)

    def pick(cond):
        return [t for t, m1, rsi in scored if cond(m1, rsi)]

    # 1) strict: positive momentum >5% and RSI < 75
    picks = pick(lambda m1, rsi: m1 > 5 and rsi < 75)[:want]

    # 2) relaxed: any positive momentum and RSI < 80
    if len(picks) < want:
        more = [t for t in pick(lambda m1, rsi: m1 > 0 and rsi < 80) if t not in picks]
        picks += more[: (want - len(picks))]

    # 3) fallback: top remaining by ranking (even if RSI high or m1 <= 0)
    if len(picks) < want:
        more = [t for t, _, _ in scored if t not in picks]
        picks += more[: (want - len(picks))]

    return picks[:want]


# ── one-stop analysis helper ─────────────────────────────
def analyze_ticker(t:str, owned:bool, cur_amt:float, horizon:int)->dict:
    df=fetch_price_data(t)
    heads=fetch_headlines_yf(t) or finnhub_news(t,7,15)
    senti=sentiment_score(heads)
    ns=news_signal(heads); es=earnings_signal(t)
    ind=technicals(df); fcast=forecast_price(df,horizon)
    anomaly=detect_anomaly(df,t,heads,es)
    mom = momentum_snapshot(df)

    return dict(
        ticker=t, owned=owned, cur_amt=round(cur_amt,2),
        anomaly=anomaly, ind=ind, forecast=fcast,
        senti=senti, sent_desc="Positive" if senti>0 else "Negative" if senti<0 else "Neutral",
        earn_line="—" if not es["flag"] else f"{es['info']} (s={es['score']:.2f}, c={es['conf']:.2f})",
        news_line=f"s={ns['score']:.2f}, c={ns['conf']:.2f}, tag={ns['tag'] or '—'}",
        ns_score=ns["score"], ns_conf=ns["conf"], es_score=es["score"], es_conf=es["conf"],
        ns_tag=ns["tag"],                 # <-- added
        mom=mom, new_amt=0.0
    )

def alpha_score(r: dict) -> float:
    """
    Blend news/earnings (if any) + 1m momentum + technicals.
    Returns >= 0 so it can be used for proportional sizing.
    """
    ns = max(0.0, r["ns_score"]) * r["ns_conf"] * 0.5
    es = max(0.0, r["es_score"]) * r["es_conf"] * 0.2
    mom = max(0.0, r["mom"]["m1"]) / 40.0           # +0.025 per +1% m/m
    tech = max(0.0, (technical_score(r) + 1.5) / 3) # normalize ~[0,1]
    return float(max(0.0, ns + es + mom + tech))

def technical_score(r: dict) -> float:
    ind, mom = r["ind"], r["mom"]
    s = 0.0
    if ind["macd"] > 0: s += 0.8
    if ind["rsi"] < 35: s += 0.6
    if ind["rsi"] > 70: s -= 0.8
    if mom["m1"] > 3:  s += 0.5
    if mom["m1"] < -3: s -= 0.5
    try:
        if ind["price"] <= ind["bb_low"]:  s += 0.4
        if ind["price"] >= ind["bb_high"]: s -= 0.4
    except Exception:
        pass
    return float(s)

def decide_owned_action(r: dict, shift_map: dict) -> str:
    """Combine rebalance with tactical signals for OWNED tickers."""
    shift = float(shift_map.get(r["ticker"], 0.0))
    # honor big rebalances first
    if shift < -1e-6: return "SELL"
    if shift >  1e-6: return "BUY"
    # otherwise, tactical overlay
    s = technical_score(r)
    if s >= 1.3 and r["ind"]["rsi"] < 75: return "BUY"
    if s <= -1.1: return "SELL"
    if s <= -0.6: return "TRIM"
    if s >= 0.6:  return "ADD"
    return "HOLD"


# ── natural-language explanation helpers ─────────────────
def plain_explanation(action: str, r: dict, shift_map: dict) -> str:
    t = r["ticker"]; price = r["ind"]["price"]; rsi = r["ind"]["rsi"]; macd = r["ind"]["macd"]
    w1, m1, m3 = r["mom"]["w1"], r["mom"]["m1"], r["mom"]["m3"]
    ns_s, ns_c = r["ns_score"], r["ns_conf"]; tag = r.get("ns_tag") or "—"
    es_line = r["earn_line"]; forecast = r["forecast"]; alpha = round(alpha_score(r), 3)
    tech_s = round(technical_score(r), 2)

    if r["owned"]:
        shift = float(shift_map.get(t, 0.0))
        dollars_txt = ("trim " + _fmt(-shift)) if action == "SELL" else ("add " + _fmt(shift)) if action == "BUY" else "hold size"
    else:
        dollars_txt = ("open " + _fmt(r.get("new_amt", 0.0))) if action == "BUY" else "no purchase"

    msg = (
        f"{t} ~{_fmt(price)}. RSI {int(round(rsi))} / MACD {macd:.2f}. "
        f"Momentum: 1w {w1:.1f}%, 1m {m1:.1f}%, 3m {m3:.1f}%. "
        f"News {ns_s:.2f} (conf {ns_c:.2f}, tag {tag}); earnings: {es_line}. "
        f"Tech score {tech_s}, alpha {alpha:.3f}; model forecast ~{_fmt(forecast)}. "
    )
    if action in ("BUY","ADD") and r["ind"]["rsi"] < 75:
        msg += f"We’ll {dollars_txt} on improving breadth/momentum; key risk is a fade if news flow weakens."
    elif action in ("SELL","TRIM"):
        msg += f"We’ll {dollars_txt} given overextension/weak signals; risk is underweighting if trend resumes."
    else:
        msg += f"We’ll {dollars_txt} today; signals are mixed."
    return msg

def normalize_ticker(t: str) -> str:
    # prefer Yahoo's dash form for class shares, e.g., BRK-B
    return t.strip().upper().replace(".", "-")

def safe_news_ticker(t: str) -> str:
    # skip symbols known to be delisted to avoid 404 noise
    delisted = {"ATVI"}  # add more if needed
    return "" if t in delisted else t

def gpt_explanation(action: str, r: dict, shift_map: dict) -> str:
    """
    Ask ChatGPT to write a concise, specific rationale.
    Falls back to plain_explanation on any API error or if OpenAI is unavailable.
    """
    if not globals().get("_HAS_OPENAI", False):
        return plain_explanation(action, r, shift_map)
    try:
        t = r["ticker"]
        price = r["ind"]["price"]
        rsi   = r["ind"]["rsi"]
        macd  = r["ind"]["macd"]
        w1, m1, m3 = r["mom"]["w1"], r["mom"]["m1"], r["mom"]["m3"]
        ns_s, ns_c = r["ns_score"], r["ns_conf"]
        tag        = r.get("ns_tag") or "—"
        es_line    = r["earn_line"]
        forecast   = r["forecast"]
        alpha      = round(alpha_score(r), 3)

        if r["owned"]:
            shift = float(shift_map.get(t, 0.0))
            dollars = (-shift) if action == "SELL" else (shift if action == "BUY" else 0.0)
        else:
            dollars = float(r.get("new_amt", 0.0)) if action == "BUY" else 0.0

        facts = {
            "ticker": t, "action": action, "price": price,
            "rsi": rsi, "macd": macd, "momentum_1w_pct": w1,
            "momentum_1m_pct": m1, "momentum_3m_pct": m3,
            "news_score": ns_s, "news_conf": ns_c, "news_tag": tag,
            "earnings_line": es_line, "forecast_price": forecast,
            "alpha_blend": alpha, "dollars": dollars, "owned": r["owned"]
        }

        system_msg = (
            "You are a portfolio PM writing short trade rationales. "
            "Tone: confident, plain English, specific numbers, 150–200 words."
        )
        user_msg = (
            "Write a readable rationale for this trade. Include ticker, price, RSI, MACD, "
            "1w/1m/3m momentum, news tag+score+confidence, earnings line, forecast price, "
            "alpha blend, and dollar action; note one risk. Facts:\n" + json.dumps(facts)
        )

        resp = _OPENAI.chat.completions.create(
            model=_OPENAI_MODEL,
            temperature=0.6,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": user_msg}],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return plain_explanation(action, r, shift_map)

# ── MAIN ─────────────────────────────────────────────────
def run_tradehints(prefs: dict) -> dict:
    """
    Run the TradeHints engine with explicit user prefs.
    Returns a dictionary suitable for a web UI.

    Expected prefs keys:
      risk, sectors, capital, horizon_days, portfolio, positions,
      tickers, open_sp500, max_new
    """
    total_new = prefs["capital"]
    horizon = prefs["horizon_days"]

    results, analysed = [], set()

    # analyze USER tickers
    for t in prefs["tickers"]:
        res = analyze_ticker(
            t,
            owned=(t in prefs["positions"]),
            cur_amt=prefs["positions"].get(t, 0.0),
            horizon=horizon,
        )
        results.append(res)
        analysed.add(t)
        log_forecast(t, res["forecast"])

    # ensure ALL portfolio names are analyzed
    for t in prefs["portfolio"]:
        if t in analysed:
            continue
        res = analyze_ticker(
            t,
            owned=True,
            cur_amt=prefs["positions"].get(t, 0.0),
            horizon=horizon,
        )
        results.append(res)
        analysed.add(t)
        log_forecast(t, res["forecast"])

    # add S&P-500 ideas (only if we will actually BUY them)
    extra = []
    if prefs.get("open_sp500") and prefs.get("max_new", 0) > 0:
        exclude = set(prefs["portfolio"]) | analysed
        extra = sp500_news_candidates(
            exclude=exclude,
            top_n=prefs["max_new"],
            days=5,
        )
        if len(extra) < prefs["max_new"]:
            extra += sp500_momentum_backfill(
                exclude=exclude | set(extra),
                want=prefs["max_new"] - len(extra),
            )
        for t in extra:
            res = analyze_ticker(t, owned=False, cur_amt=0.0, horizon=horizon)
            results.append(res)
            analysed.add(t)
            log_forecast(t, res["forecast"])

    # build rebalance on current holdings only
    port_total = float(sum(prefs["positions"].values()))
    cur_w = current_weights(prefs["positions"])
    tgt_w = smart_target_weights(prefs["portfolio"], prefs["risk"])
    moves = rebalance(cur_w, tgt_w, port_total)
    sector_note = sector_imbalances(cur_w)

    rb_map = {m["ticker"]: m["shift$"] for m in moves}
    total_trims = sum(-m["shift$"] for m in moves if m["shift$"] < 0)
    total_adds = sum(m["shift$"] for m in moves if m["shift$"] > 0)
    need_from_new = max(0.0, total_adds - total_trims)
    new_after_rb = max(0.0, total_new - need_from_new)

    CASH_IN = float(total_trims + total_new)
    trades = []

    def tactical_dollars(r):
        cur_amt = float(r["cur_amt"])
        if cur_amt <= 0:
            return 0.0
        s = technical_score(r)
        pos_cap = 0.20 * cur_amt
        port_cap = 0.05 * port_total
        raw = s * (0.08 * cur_amt)
        return float(
            max(-min(pos_cap, port_cap), min(min(pos_cap, port_cap), raw))
        )

    owned_map = {r["ticker"]: r for r in results if r["owned"]}
    for t, r in owned_map.items():
        base = float(rb_map.get(t, 0.0))
        tact = 0.0
        act = decide_owned_action(r, rb_map)
        if act in ("BUY", "ADD"):
            tact = max(0.0, tactical_dollars(r))
        elif act in ("SELL", "TRIM"):
            tact = min(0.0, tactical_dollars(r))
        shift = base + tact
        if abs(shift) > 1e-6:
            trades.append(
                {
                    "ticker": t,
                    "side": "BUY" if shift > 0 else "SELL",
                    "dollars": abs(round(shift, 2)),
                }
            )

    buy_candidates = [
        r
        for r in results
        if (not r["owned"]) and (alpha_score(r) > 0.0) and (r["ind"]["rsi"] < 75)
    ]
    if not buy_candidates and extra:
        ranked = sorted(
            [
                r
                for r in results
                if (not r["owned"]) and (r["ticker"] in extra)
            ],
            key=lambda x: x["mom"]["m1"],
            reverse=True,
        )
        buy_candidates = ranked[:1]

    owned_buys = sum(t["dollars"] for t in trades if t["side"] == "BUY")
    owned_sells = sum(t["dollars"] for t in trades if t["side"] == "SELL")
    cash_after_owned = CASH_IN + owned_sells - owned_buys

    start_alloc = {}
    if cash_after_owned > 0 and buy_candidates:
        w = np.array(
            [max(1e-6, alpha_score(r)) for r in buy_candidates],
            dtype=float,
        )
        w /= w.sum()
        for r, ww in zip(buy_candidates, w):
            amt = float(round(cash_after_owned * ww, 2))
            if amt > 0:
                start_alloc[r["ticker"]] = amt
                trades.append(
                    {"ticker": r["ticker"], "side": "BUY", "dollars": amt}
                )

    buys_total = sum(t["dollars"] for t in trades if t["side"] == "BUY")
    sells_total = sum(t["dollars"] for t in trades if t["side"] == "SELL")
    cash_used = buys_total - sells_total
    diff = CASH_IN - cash_used
    if diff < -0.009 and buys_total > 0:
        scale = max(0.0, (buys_total + diff) / buys_total)
        for tr in trades:
            if tr["side"] == "BUY":
                tr["dollars"] = float(round(tr["dollars"] * scale, 2))
        buys_total = sum(t["dollars"] for t in trades if t["side"] == "BUY")
        sells_total = sum(t["dollars"] for t in trades if t["side"] == "SELL")
        cash_used = buys_total - sells_total
    unassigned = round(CASH_IN - cash_used, 2)

    trade_by_ticker = {}
    for tr in trades:
        trade_by_ticker.setdefault(tr["ticker"], []).append(tr)

    order = (
        sorted(
            [
                r
                for r in results
                if r["owned"] and r["ticker"] in trade_by_ticker
            ],
            key=lambda x: x["ticker"],
        )
        + sorted(
            [
                r
                for r in results
                if (not r["owned"]) and r["ticker"] in trade_by_ticker
            ],
            key=lambda x: -sum(
                t["dollars"] for t in trade_by_ticker[x["ticker"]]
            ),
        )
    )

    summaries = []
    for r in order:
        acts = trade_by_ticker[r["ticker"]]
        net = sum(
            a["dollars"] if a["side"] == "BUY" else -a["dollars"]
            for a in acts
        )
        action = "BUY" if net > 0 else "SELL"
        expl = gpt_explanation(action, r, rb_map)
        summaries.append(
            {
                "ticker": r["ticker"],
                "action": action,
                "dollars": abs(net),
                "explanation": expl,
            }
        )

    meta = {
        "cash_in": CASH_IN,
        "cash_used": cash_used,
        "cash_buffer": unassigned,
        "sector_note": sector_note,
    }

    return {"trades": trades, "summaries": summaries, "meta": meta}

def main():
    prefs = get_user_preferences()
    out = run_tradehints(prefs)
    # basic print if you care

if __name__ == "__main__":
    main()