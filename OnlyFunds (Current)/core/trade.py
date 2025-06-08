import time
import requests
import hmac
import hashlib
import numpy as np

def backtest_strategy(df, signal, config, symbol, ml_filter=None, ml_features=None):
    trades = []
    in_position = False
    entry_price = 0
    equity = config.get("initial_capital", 10.0)
    profit_accum = 0.0
    threshold = config.get("threshold", 0.5)
    sl_pct = config.get("stop_loss_pct", 0.025)
    tp_pct = config.get("take_profit_pct", 0.05)
    trail_pct = config.get("trailing_stop_pct", 0.03)
    trail_trigger_pct = config.get("trailing_trigger_pct", 0.02)

    trail_active = False
    trail_price = 0

    wins = 0
    losses = 0
    total_return = 0

    # If ml_features is not supplied but ml_filter is, build it now
    if ml_filter is not None and ml_features is None:
        for col in ml_filter.features:
            if col not in df.columns:
                df[col] = np.nan
        ml_features = df[ml_filter.features]

    for i in range(1, len(df)):
        price = df["Close"].iloc[i]
        sig = signal.iloc[i]

        dynamic_size = max(1.0, profit_accum * 0.05)

        allow_trade = True
        if ml_filter:
            # Use only the features DataFrame in the right order!
            allow_trade = ml_filter.should_enter(ml_features, i, sig, threshold)

        if not in_position and sig > threshold and allow_trade:
            entry_price = price
            in_position = True
            trail_active = False
            trades.append({
                "symbol": symbol,
                "type": "buy",
                "entry_price": entry_price,
                "timestamp": df.index[i],
                "amount": dynamic_size,
                "status": "open"
            })

        elif in_position:
            stop_loss = entry_price * (1 - sl_pct)
            take_profit = entry_price * (1 + tp_pct)

            if not trail_active and price >= entry_price * (1 + trail_trigger_pct):
                trail_active = True
                trail_price = price * (1 - trail_pct)

            if trail_active:
                new_trail = price * (1 - trail_pct)
                trail_price = max(trail_price, new_trail)

            if (
                price <= stop_loss or
                price >= take_profit or
                (trail_active and price <= trail_price) or
                sig < 0
            ):
                exit_price = price
                pnl = (exit_price - entry_price) * dynamic_size
                trades[-1].update({
                    "exit_price": exit_price,
                    "exit_time": df.index[i],
                    "pnl": pnl,
                    "status": "closed"
                })
                in_position = False
                trail_active = False
                profit_accum += pnl
                total_return += pnl
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

    if trades:
        print(f"📈 Total Trades: {len(trades)}")
        print(f"✅ Wins: {wins} | ❌ Losses: {losses}")
        print(f"🏁 Avg Return/Trade: {total_return / len(trades):.4f} USDT")
        print(f"📊 Win Rate: {wins / len(trades) * 100:.2f}%")

    return trades

def place_live_order(pair, price, amount, order_type="buy", dry_run=True, config=None):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    if config is None or config.get("live") is not True:
        print(f"🟡 DRY TRADE [{order_type.upper()}] {pair} @ {price:.4f} x {amount:.2f} — {timestamp}")
        return {
            "pair": pair,
            "price": price,
            "amount": amount,
            "type": order_type,
            "timestamp": timestamp
        }

    try:
        api_key = config["coinex"]["api_key"]
        api_secret = config["coinex"]["api_secret"]
        url = "https://api.coinex.com/v1/order/limit"
        market = pair
        side = "buy" if order_type.lower() == "buy" else "sell"
        payload = {
            "market": market,
            "type": side,
            "amount": f"{amount:.4f}",
            "price": f"{price:.4f}",
            "access_id": api_key,
            "tonce": int(time.time() * 1000),
        }

        sorted_payload = "&".join([f"{k}={v}" for k, v in sorted(payload.items())])
        sign = hmac.new(api_secret.encode(), sorted_payload.encode(), hashlib.sha256).hexdigest()
        headers = {"Authorization": sign}

        resp = requests.post(url, data=payload, headers=headers)
        data = resp.json()

        if data["code"] == 0:
            print(f"✅ LIVE TRADE [{side.upper()}] {pair} executed")
            return {
                "pair": pair,
                "price": price,
                "amount": amount,
                "type": side,
                "timestamp": timestamp,
                "status": "executed"
            }
        else:
            print(f"❌ Live trade failed: {data}")
            return {"status": "failed", "error": data}

    except Exception as e:
        print(f"❌ Live trade exception: {e}")
        return {"status": "error", "error": str(e)}