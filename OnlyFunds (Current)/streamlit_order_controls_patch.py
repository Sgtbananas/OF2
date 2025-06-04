
import streamlit as st
import pandas as pd
from core import order_manager
import datetime

st.title("📊 Trade Control Panel")

orders = order_manager.load_orders()
if orders:
    df_orders = pd.DataFrame(orders)
    df_orders["entry_time"] = pd.to_datetime(df_orders["entry_time"])
    if "exit_time" in df_orders.columns:
        df_orders["exit_time"] = pd.to_datetime(df_orders["exit_time"])
    df_orders["PnL"] = df_orders.get("pnl", 0).round(2)
    df_orders.sort_values(by="entry_time", ascending=False, inplace=True)
    st.dataframe(df_orders, use_container_width=True)

    st.success(f"💼 Total Trades: {len(df_orders)} | Open: {(df_orders['status'] == 'open').sum()} | PnL: ${df_orders['PnL'].sum():.2f}")

    if st.button("📤 Export Orders to CSV"):
        order_manager.export_orders_csv()
        st.info("✅ Exported to orders_export.csv")

    if st.button("✖ Close All Open Positions"):
        open_trades = order_manager.get_open_orders()
        now = datetime.datetime.now()
        for trade in open_trades:
            order_manager.close_order(trade["pair"], trade["entry_price"], now)
        st.warning("✖ All open trades marked as closed.")

    if st.button("🔐 Close All Trades"):
        all_trades = order_manager.load_orders()
        now = datetime.datetime.now()
        for trade in all_trades:
            if trade["status"] == "open":
                order_manager.close_order(trade["pair"], trade["entry_price"], now)
        st.error("🔐 All trades (open) have been closed.")

else:
    st.warning("No orders logged yet.")

if st.button("🛑 STOP Live Trading"):
    st.session_state["stop_live"] = True
    st.error("Live trading STOPPED")
