import numpy as np
import pandas as pd
import time
from core.config import load_config
from core.engine import run_bot, backtest_bot
from core.ml_filter import MLFilter
from core.coin_selector import get_top_200_coinex_symbols
from core.risk_manager import adjust_risk_based_on_profile
import os

class TradingOrchestrator:
    """
    Orchestrates backtesting, ML training, model selection, and live deployment
    based on a user-selected risk profile (conservative, normal, aggressive).
    User only needs to choose risk profile; orchestrator does all else.
    """

    def __init__(self, config_path="OnlyFunds (Current)/config.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.available_profiles = ["conservative", "normal", "aggressive"]
        self.available_models = ["none", "logistic_regression", "random_forest"]
        self.ml_thresholds = {
            "conservative": 0.7,
            "normal": 0.6,
            "aggressive": 0.5
        }
        self.metrics = {}  # Holds backtest results

    def select_profile(self, profile=None):
        """
        Set risk profile. If none provided, use config or prompt.
        """
        if profile is None:
            profile = self.config.get("risk", "normal")
        if profile not in self.available_profiles:
            raise ValueError(f"Profile must be one of {self.available_profiles}")
        self.config["risk"] = profile
        print(f"✅ Risk profile set to: {profile}")

    def orchestrate(self, profile=None, retrain=True, schedule=False):
        """
        Main orchestration routine:
        1. Sets profile.
        2. Runs candidate pipelines (with/without ML, different models).
        3. Selects best pipeline based on backtest PnL and winrate.
        4. Deploys the optimal pipeline for live trading (or dry_run).

        Args:
            profile (str): "conservative", "normal", "aggressive"
            retrain (bool): Retrain ML models
            schedule (bool): If True, rerun orchestration every hour.
        """
        self.select_profile(profile)
        while True:
            best = self._run_and_select_best_pipeline(retrain=retrain)
            self._deploy(best)
            if schedule:
                print("⏳ Sleeping 1 hour before next orchestration...")
                time.sleep(3600)
            else:
                break

    def _run_and_select_best_pipeline(self, retrain=True):
        """
        For each ML setting (none, logistic regression, random forest), perform backtest.
        Store results and select the best by PnL and winrate.
        """
        results = []
        for model_type in self.available_models:
            print(f"\n🧪 Backtesting with model: {model_type}")
            config = self.config.copy()
            # Set up ML filter
            if model_type == "none":
                config["ml_filter"] = False
                ml_filter = None
            else:
                config["ml_filter"] = True
                config["ml_filter_type"] = model_type
                config["ml_threshold"] = self.ml_thresholds[config["risk"]]
                ml_filter = MLFilter(model_type=model_type)
                if retrain:
                    X_train, y_train = self._prepare_ml_data(config)
                    if len(X_train) > 0:
                        ml_filter.train(np.array(X_train), np.array(y_train))
                    else:
                        print("⚠️ Not enough training data for ML filter; skipping training.")
            # Backtest pipeline
            bt_result = backtest_bot(config, ml_filter=ml_filter)
            print(f"Model: {model_type} | PnL: {bt_result['total_pnl']:.4f} | Winrate: {bt_result['winrate']:.2%}")
            results.append({
                "model": model_type,
                "pnl": bt_result["total_pnl"],
                "winrate": bt_result["winrate"],
                "config": config,
                "ml_filter": ml_filter
            })
        # Select winner (highest PnL, then winrate)
        results.sort(key=lambda x: (x["pnl"], x["winrate"]), reverse=True)
        best = results[0]
        print(f"\n🏆 Selected best pipeline: {best['model']} (PnL: {best['pnl']:.4f}, Winrate: {best['winrate']:.2%})")
        self.metrics = results
        return best

    def _prepare_ml_data(self, config):
        """
        Extracts features & labels from backtest for ML training.
        Uses backtest pipeline to simulate trades and generate labels.
        """
        # Simulate pipeline WITHOUT ML for ground truth labels
        config_ = config.copy()
        config_["ml_filter"] = False
        # Use coin selector to get symbols
        symbols = config_.get("symbols", get_top_200_coinex_symbols())
        X, y = [], []
        for symbol in symbols:
            # Fetch historical data (user should implement this)
            df = self._fetch_historical_data(symbol, config_)
            if df is None or len(df) < 40:
                continue
            # Generate features and labels for ML
            for i in range(30, len(df)-1):
                feats = MLFilter.extract_features(df, i)
                # Label: 1 if next bar PnL>0, else 0
                pnl = df["Close"].iloc[i+1] - df["Close"].iloc[i]
                y.append(int(pnl > 0))
                X.append(feats)
        return X, y

    def _fetch_historical_data(self, symbol, config):
        """
        Robustly fetch OHLCV data for a given symbol.
        Priority:
        1. Local CSV at data/{symbol}_{timeframe}.csv
        2. (Optional) API or database fetch (placeholder, easily patched in)
        Ensures DataFrame contains 'Close' and 'Volume' columns.
        Returns None if data is missing or invalid.
        """
        import pandas as pd
        import logging

        timeframe = config.get("timeframe", "5min")
        # Standardize file path: e.g., data/BTC_USDT_5min.csv
        safe_symbol = symbol.replace("/", "_").replace("-", "_")
        fname = f"data/{safe_symbol}_{timeframe}.csv"
        try:
            if os.path.exists(fname):
                df = pd.read_csv(fname)
                # Try to standardize columns (case-insensitive match)
                df_cols = [col.lower() for col in df.columns]
                colmap = {}
                for req in ["close", "volume"]:
                    matches = [c for c in df.columns if c.lower() == req]
                    if matches:
                        colmap[matches[0]] = req.capitalize()
                df = df.rename(columns=colmap)
                # Ensure required columns present
                if "Close" not in df.columns or "Volume" not in df.columns:
                    logging.warning(f"{fname} missing required columns. Found: {df.columns}")
                    return None
                df = df.dropna(subset=["Close", "Volume"])
                if len(df) < 50:
                    logging.warning(f"{fname} has too few rows ({len(df)}).")
                    return None
                return df
            else:
                logging.warning(f"No local CSV for {symbol} ({fname}).")
                # (Optional) Insert API/database fetch here
                # e.g., df = fetch_from_api(symbol, timeframe)
                # if df is not None: return df
                return None
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            return None

    def _deploy(self, best_pipeline):
        """
        Deploys the selected pipeline for live or dry_run trading.
        """
        print(f"🚀 Deploying pipeline: {best_pipeline['model']} | Risk: {best_pipeline['config']['risk']}")
        run_bot(best_pipeline["config"], ml_filter=best_pipeline["ml_filter"])

if __name__ == "__main__":
    # CLI usage:
    import argparse
    parser = argparse.ArgumentParser(description="OnlyFunds Trading Orchestrator")
    parser.add_argument("--profile", choices=["conservative", "normal", "aggressive"], default=None, help="Risk profile to use.")
    parser.add_argument("--config", default="OnlyFunds (Current)/config.yaml", help="Path to config YAML.")
    parser.add_argument("--schedule", action="store_true", help="Continuously re-run orchestration every hour.")
    parser.add_argument("--no-retrain", action="store_true", help="Skip ML retraining (use existing model).")
    args = parser.parse_args()
    orchestrator = TradingOrchestrator(config_path=args.config)
    orchestrator.orchestrate(
        profile=args.profile,
        retrain=not args.no_retrain,
        schedule=args.schedule
    )
