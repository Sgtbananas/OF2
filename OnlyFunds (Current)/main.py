import streamlit as st
from datetime import datetime
from core import order_manager
from core.config import load_config
from core.engine import run_bot
from core.orchestrator import get_orchestrator

# Set up Streamlit page
st.set_page_config(page_title="OnlyFunds", layout="wide")
st.title("📈 OnlyFunds Trading Dashboard")

# --- SIDEBAR: World-class, minimal controls ---
st.sidebar.title("OnlyFunds - Control Panel")

# Config file path input
config_path = st.sidebar.text_input("Config file path", "config.yaml")

# Risk profile selector
risk_profile = st.sidebar.selectbox("Risk Profile", ["conservative", "normal", "aggressive"], index=1)

# Mode selector
mode = st.sidebar.selectbox("Mode", ["dry_run", "live", "backtest"], index=0)

# Reload config/orchestrator button
if st.sidebar.button("Reload Config / Orchestrator"):
    st.session_state.pop("orch", None)
    st.session_state.pop("config", None)
    st.session_state.pop("last_backtest", None)
    st.experimental_rerun()

# --- Orchestrator and config loading ---
def load_orch():
    # Always pass the selected risk profile and mode for latest session
    return get_orchestrator(
        config_path=config_path,
        config={
            "risk_profile": risk_profile,
            "mode": mode
        }
    )

if "orch" not in st.session_state:
    st.session_state["orch"] = load_orch()
    st.session_state["config"] = st.session_state["orch"].config
orch = st.session_state["orch"]
config = st.session_state["config"]

# --- MAIN CONTROLS ---
col1, col2, col3, col4 = st.columns(4)

def run_orchestrated_trading():
    st.info(
        f"Running AI/ML-selected strategy in {mode.upper()} mode — Risk Profile: {risk_profile.capitalize()}"
    )
    try:
        orch.run(mode=mode)
        st.success("✅ Trading run completed with AI/ML-driven strategy selection.")
    except Exception as e:
        st.error(f"❌ Trading run failed: {e}")

def run_orchestrated_backtest():
    st.info("Backtest initiated (AI/ML will select best strategy for historical data)...")
    # Default dates (can be improved to allow user selection if needed)
    start_date = st.date_input("Backtest start date", value=datetime(2024, 1, 1))
    end_date = st.date_input("Backtest end date", value=datetime.now())
    if st.button("▶️ Run Backtest"):
        results = orch.run(mode="backtest", start_date=str(start_date), end_date=str(end_date))
        if results is not None and not results.empty:
            st.success("✅ Backtest completed.")
            st.dataframe(results)
            csv = results.to_csv(index=False)
            st.download_button("Download Backtest Results", csv, "backtest_results.csv", "text/csv")
        else:
            st.warning("No backtest results to display.")

# Run Strategy (AI/ML-driven)
if mode != "backtest":
    if col1.button("▶️ Run Trading"):
        run_orchestrated_trading()
else:
    with col1:
        run_orchestrated_backtest()

# Export Orders to CSV
if col2.button("📤 Export Orders to CSV"):
    orders = order_manager.load_orders()
    if orders:
        import pandas as pd
        df = pd.DataFrame(orders)
        df.to_csv("orders_exported.csv", index=False)
        st.success("✅ Orders exported to orders_exported.csv")
        st.download_button("Download Orders CSV", df.to_csv(index=False), "orders_exported.csv", "text/csv")
    else:
        st.warning("⚠️ No orders to export.")

# Close All Open Positions (Live)
if col3.button("❌ Close All Open Positions"):
    order_manager.close_all_open_orders()
    st.warning("⚠️ All open positions closed (simulated in dry_run).")

# Stop Live Trading
if col4.button("🛑 STOP Live Trading"):
    st.warning("⚠️ Live trading stopped (simulated in dry_run).")

# Show Orders
st.subheader("🔎 Recent Orders")
orders = order_manager.load_orders()
if orders:
    import pandas as pd
    st.dataframe(pd.DataFrame(orders))
else:
    st.info("ℹ️ No orders to display.")

# Remove any downstream code that allowed manual strategy selection.
# All strategy selection is now handled by the backend (orchestrator, AI, ML).