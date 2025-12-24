import streamlit as st
from datetime import datetime
import csv
from pathlib import Path
from core import run_tradehints

LOG_PATH = Path("usage_log.csv")

st.set_page_config(page_title="TradeHints Demo", layout="wide")
st.title("TradeHints – AI Stock Recommendation Demo")
st.caption("Educational demo. Not financial advice.")

with st.form("prefs"):
    user_id = st.text_input(
        "Identifier (email or code)",
        help="Used only to count unique users and runs.",
    )

    risk = st.selectbox("Risk tolerance", ["low", "moderate", "high"])
    capital = st.number_input(
        "New capital to invest ($)", min_value=0.0, value=5000.0, step=100.0
    )
    horizon = st.number_input(
        "Time horizon (days)", min_value=1, value=30, step=1
    )

    portfolio = st.text_input(
        "Current holdings (tickers, comma-separated)",
        "AAPL, MSFT",
    ).upper()

    positions_raw = st.text_area(
        "Current positions ($) – one per line as TICKER=AMOUNT",
        "AAPL=2000\nMSFT=3000",
    )

    tickers_raw = st.text_input(
        "Tickers to analyze (blank = use portfolio)",
        "",
    ).upper()

    open_sp = st.checkbox("Include new S&P 500 ideas", value=True)
    max_new = st.slider("Max number of new ideas", 0, 5, 3)

    submitted = st.form_submit_button("Generate recommendations")

def parse_positions(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        t, v = line.split("=", 1)
        try:
            out[t.strip().upper()] = float(v.strip().replace(",", ""))
        except ValueError:
            continue
    return out

if submitted:
    port_list = [t.strip() for t in portfolio.split(",") if t.strip()]
    positions = parse_positions(positions_raw)
    tickers = [t.strip() for t in tickers_raw.split(",") if t.strip()] or port_list

    prefs = dict(
        risk=risk,
        sectors=[],
        capital=float(capital),
        horizon_days=int(horizon),
        portfolio=port_list,
        positions=positions,
        tickers=tickers,
        open_sp500=bool(open_sp),
        max_new=int(max_new),
    )

    out = run_tradehints(prefs)

    # log usage
    if user_id:
        first_time = not LOG_PATH.exists()
        with LOG_PATH.open("a", newline="") as f:
            writer = csv.writer(f)
            if first_time:
                writer.writerow(["timestamp", "user_id", "event"])
            writer.writerow([datetime.utcnow().isoformat(), user_id.strip(), "run"])

    st.subheader("Execution Plan")
    if out["trades"]:
        st.dataframe(out["trades"], use_container_width=True)
    else:
        st.write("No trades suggested for this configuration.")

    st.subheader("Per-ticker rationales")
    for s in out["summaries"]:
        st.markdown(f"### {s['ticker']} — {s['action']} ${s['dollars']:,.2f}")
        st.write(s["explanation"])
