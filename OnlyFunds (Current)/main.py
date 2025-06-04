import streamlit as st
from datetime import datetime
from core import order_manager
from core.config import load_config
from core.engine import run_bot

# Set up Streamlit page
st.set_page_config(page_title="OnlyFunds", layout="wide")
st.title("📈 OnlyFunds Trading Dashboard")

# Load configuration
config = load_config("config.yaml")

# Main controls
col1, col2, col3, col4 = st.columns(4)

# Run Strategy
if col1.button("▶️ Run Strategy"):
    st.info(f"Running strategy in {config.get('mode', 'dry_run').upper()} mode — Risk Profile: {config.get('risk', 'normal')}")
    run_bot(config)
    st.success("✅ Strategy run completed.")

# Export Orders to CSV
if col2.button("📤 Export Orders to CSV"):
    orders = order_manager.load_orders()
    if orders:
        import pandas as pd
        df = pd.DataFrame(orders)
        df.to_csv("orders_exported.csv", index=False)
        st.success("✅ Orders exported to orders_exported.csv")
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
    st.dataframe(orders)
else:
    st.info("ℹ️ No orders to display.")
