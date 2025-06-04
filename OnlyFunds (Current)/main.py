import streamlit as st
from datetime import datetime
from core.orchestrator import get_orchestrator

st.set_page_config(page_title="OnlyFunds", layout="wide")
st.title("📈 OnlyFunds Trading Dashboard")

# Sidebar: Config & Mode
st.sidebar.header("Settings")

default_config = "config.yaml"
config_path = st.sidebar.text_input("Config file path", value=default_config)
mode = st.sidebar.selectbox("Mode", ["live", "dry_run", "backtest"])

# Backtest date selection
start_date = st.sidebar.date_input("Backtest start date", value=datetime(2024, 1, 1))
end_date = st.sidebar.date_input("Backtest end date", value=datetime.now())

# Helper: (Re)load orchestrator
def load_orch():
    return get_orchestrator(config_path=config_path)

if "orch" not in st.session_state or st.sidebar.button("Reload Config / Orchestrator"):
    st.session_state["orch"] = load_orch()

orch = st.session_state["orch"]

# Main actions
if mode in ["live", "dry_run"]:
    if st.button("Start Trading Bot", type="primary"):
        try:
            st.info(f"Starting {mode} trading bot...")
            orch.run(mode=mode)
            st.success("Trading bot started successfully!")
        except Exception as e:
            st.error(f"Failed to start trading bot: {e}")

elif mode == "backtest":
    if st.button("Run Backtest", type="primary"):
        try:
            st.info(f"Running backtest from {start_date} to {end_date}...")
            result_df = orch.run(
                mode="backtest",
                start_date=str(start_date),
                end_date=str(end_date)
            )
            if result_df is not None and not result_df.empty:
                st.success("Backtest complete!")
                st.dataframe(result_df)
            else:
                st.warning("No results from backtest.")
        except Exception as e:
            st.error(f"Failed to run backtest: {e}")

# Show logs (if any)
st.subheader("Logs")
try:
    with open("logs/latest.log", "r") as f:
        logs = f.read()
    st.text_area("Latest Logs", logs, height=200)
except Exception:
    st.info("No logs yet. Run a trading bot or backtest to see logs.")