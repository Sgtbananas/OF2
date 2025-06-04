import streamlit as st
from datetime import datetime, timedelta
from core.orchestrator import get_orchestrator

st.set_page_config(page_title="OnlyFunds", layout="wide")
st.title("📈 OnlyFunds Trading Dashboard")

# Sidebar: Config & Mode
st.sidebar.header("Settings")

default_config = "config.yaml"
config_path = st.sidebar.text_input("Config file path", value=default_config)

# (Re)load orchestrator and config
def load_orch():
    return get_orchestrator(config_path=config_path)

if "orch" not in st.session_state or st.sidebar.button("Reload Config / Orchestrator"):
    st.session_state["orch"] = load_orch()
    st.session_state["config"] = st.session_state["orch"].config
    st.session_state["last_backtest"] = None

orch = st.session_state["orch"]
config = st.session_state["config"]

# --- Strategy and risk profile selector ---
strategy = st.sidebar.selectbox(
    "Strategy",
    options=config["all_strategies"],
    index=config["all_strategies"].index(config.get("strategy", config["all_strategies"][0]))
)
risk_profile = st.sidebar.selectbox(
    "Risk Profile",
    options=["conservative", "normal", "aggressive"],
    index=["conservative", "normal", "aggressive"].index(config.get("risk_profile", "normal"))
)

# Update config in orchestrator if user changes selection
config["strategy"] = strategy
config["risk_profile"] = risk_profile
orch.config = config

# --- Main actions: run or backtest ---
mode = st.sidebar.selectbox("Mode", ["live", "dry_run", "backtest"], index=["live", "dry_run", "backtest"].index(config.get("mode", "dry_run")))

if mode in ["live", "dry_run"]:
    if st.button("Start Trading Bot", type="primary"):
        try:
            st.info(f"Starting {mode} trading bot with strategy: {strategy}, risk: {risk_profile}")
            orch.run(mode=mode)
            st.success("Trading bot started successfully!")
        except Exception as e:
            st.error(f"Failed to start trading bot: {e}")

# --- Automatic backtest: run automatically when config/orchestrator is loaded or config changes ---
def auto_backtest():
    try:
        st.info("Running automatic backtest on all available data...")
        # Use last 90 days by default
        start_date = (datetime.now() - timedelta(days=90)).date()
        end_date = datetime.now().date()
        result_df = orch.run(
            mode="backtest",
            start_date=str(start_date),
            end_date=str(end_date)
        )
        st.session_state["last_backtest"] = result_df
        return result_df
    except Exception as e:
        st.error(f"Automatic backtest failed: {e}")
        return None

if mode == "backtest":
    # Automatically run backtest ONCE per config/orchestrator load, not on every rerun
    if st.session_state.get("last_backtest") is None:
        result_df = auto_backtest()
    else:
        result_df = st.session_state["last_backtest"]
    if result_df is not None and not result_df.empty:
        st.success("Backtest complete!")
        st.dataframe(result_df)
    elif result_df is not None:
        st.warning("No results from backtest.")

# --- Show logs ---
st.subheader("Logs")
try:
    with open("logs/latest.log", "r") as f:
        logs = f.read()
    st.text_area("Latest Logs", logs, height=200)
except Exception:
    st.info("No logs yet. Run a trading bot or backtest to see logs.")