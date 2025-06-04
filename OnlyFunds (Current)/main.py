import streamlit as st
from datetime import datetime
from core import order_manager
from core.config import load_config
from core.orchestrator import TradingOrchestrator

st.set_page_config(page_title="OnlyFunds", layout="wide")
st.title("📈 OnlyFunds Trading Dashboard")

# --- UI: Risk Profile and Mode Selection ---
config = load_config("config.yaml")
profile = st.radio("Select Risk Profile", options=["conservative", "normal", "aggressive"], index=["conservative", "normal", "aggressive"].index(config.get("risk", "normal")))
mode = st.radio("Select Run Mode", options=["dry_run", "live"], index=0 if config.get("mode", "dry_run") == "dry_run" else 1)

# --- Controls (Orchestrator) ---
col1, col2, col3 = st.columns(3)
orchestrator = TradingOrchestrator()

if col1.button("▶️ Start Orchestrator"):
    # Set config profile & mode
    config["risk"] = profile
    config["mode"] = mode
    # Save updated config for orchestrator
    import yaml
    with open("config.yaml", "w") as f:
        yaml.dump(config, f)
    # Run orchestrator for one full cycle (dry_run or live)
    st.info(f"Orchestrator running in {mode.upper()} mode — Risk Profile: {profile}")
    orchestrator.orchestrate(profile=profile, retrain=True, schedule=False)
    st.success("✅ Orchestrator run completed.")

if col2.button("📤 Export Orders to CSV"):
    orders = order_manager.load_orders()
    if orders:
        import pandas as pd
        df = pd.DataFrame(orders)
        df.to_csv("orders_exported.csv", index=False)
        st.success("✅ Orders exported to orders_exported.csv")
    else:
        st.warning("⚠️ No orders to export.")

if col3.button("❌ Close All Open Positions"):
    order_manager.close_all_open_orders()
    st.warning("⚠️ All open positions closed (simulated in dry_run).")

# --- Show Orders ---
st.subheader("🔎 Recent Orders")
orders = order_manager.load_orders()
if orders:
    st.dataframe(orders)
else:
    st.info("ℹ️ No orders to display.")

# --- Orchestrator Metrics Display (optional) ---
if getattr(orchestrator, "metrics", None):
    st.subheader("🧪 Recent Orchestrator Backtest Results")
    import pandas as pd
    st.dataframe(pd.DataFrame(orchestrator.metrics))