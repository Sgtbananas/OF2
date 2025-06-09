import logging
import pandas as pd
from typing import Optional, Dict, Any, List, Union

class TradingOrchestrator:
    """
    Orchestrates trading and backtesting operations using configuration, 
    ML filtering, coin selection, and risk management.
    """

    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        # Move imports inside to avoid circular import issues
        from core.config import load_config
        from core.logger import log_message, log_error

        if config is not None:
            loaded = load_config(config_path) if config_path else {}
            merged = {**loaded, **config}
            self.config = merged
            log_message("Config provided directly to orchestrator (merged with file if present).")
        else:
            self.config = load_config(config_path)
            log_message(f"Config loaded from {config_path or 'default location'}.")

        self.ml_filter: Optional['MLFilter'] = None
        self.selected_coins: List[str] = []
        self._init_ml_filter()

    def _init_ml_filter(self):
        from core.ml_filter import MLFilter
        from core.logger import log_message, log_error
        ml_cfg = self.config.get("ml_filter", {})
        if isinstance(ml_cfg, bool):
            enabled = ml_cfg
            ml_cfg = {}
        else:
            enabled = ml_cfg.get("enabled", False)
        if enabled:
            try:
                self.ml_filter = MLFilter(model_path=ml_cfg.get("model_path", "ml_filter_model.pkl"))
                log_message("ML filter initialized.")
            except Exception as e:
                log_error(f"Failed to initialize ML filter: {e}")
                self.ml_filter = None

    def auto_select_strategy(self):
        from core.coin_selector import select_top_coins
        from core.logger import log_message
        candidate_strategies = self.config.get("all_strategies", ["ema", "rsi", "macd", "sma", "bbands"])
        results = select_top_coins(
            all_strategies=candidate_strategies,
            top_n=1,
            exchange_name=self.config.get("exchange", "coinex"),
            quote="USDT",
            timeframe=self.config.get("timeframe", "5m"),
            limit=self.config.get("limit", 300),
            ml_enabled=(self.ml_filter is not None),
            ml_model_path=getattr(self.ml_filter, "model_path", "ml_filter_model.pkl"),
            config=None
        )
        if results:
            best = results[0]
            stgy = best.get("strategies")
            if isinstance(stgy, list):
                stgy = stgy[0] if stgy else None
            log_message(f"[AI] Auto-selected strategy: {stgy} for coin {best.get('symbol')}")
            return stgy
        else:
            log_message("[AI] No suitable strategy found, defaulting to 'ema'")
            return "ema"

    def select_coins(self):
        from core.coin_selector import get_top_200_coinex_symbols
        from core.logger import log_message, log_error
        try:
            if self.config.get("exchange", "coinex").lower() == "coinex":
                self.selected_coins = get_top_200_coinex_symbols()
                log_message(f"Selected top {len(self.selected_coins)} CoinEx symbols.")
            else:
                self.selected_coins = self.config.get("symbols", [])
                log_message(f"Loaded symbols from config: {self.selected_coins}")
        except Exception as e:
            log_error(f"Error during coin selection: {e}")
            self.selected_coins = []

    def run_trading(self):
        from core.engine import run_bot
        from core.risk_manager import adjust_risk_based_on_profile
        from core.logger import log_message, log_error
        try:
            if not self.selected_coins:
                self.select_coins()
            if not self.selected_coins:
                log_error("No tradable symbols available. Please check your exchange connectivity or configuration.")
                raise RuntimeError("No tradable symbols available.")

            strategy = self.config.get("strategy")
            if not strategy:
                strategy = self.auto_select_strategy()
                self.config["strategy"] = strategy

            exchange = self.config.get("exchange")
            timeframe = self.config.get("timeframe", "5m")
            capital = self.config.get("capital", 1000)
            risk_profile = self.config.get("risk_profile", "medium")

            self.config["risk_profile"] = risk_profile
            adjust_risk_based_on_profile(self.config)

            log_message(f"Running bot for {len(self.selected_coins)} symbols on {exchange} ({timeframe}) with strategy '{strategy}'.")

            bot_config = {
                "mode": self.config.get("mode", "dry_run"),
                "risk": self.config.get("risk", 0.01),
                "target": self.config.get("target", 0.02),
                "symbols": self.selected_coins,
                "all_strategies": self.config.get("all_strategies", [strategy]),
                "ml_filter": self.ml_filter is not None,
                "ml_threshold": self.config.get("ml_threshold", 0.6),
                "timeframe": timeframe,
                "limit": self.config.get("limit", 300)
            }
            run_bot(bot_config)
        except Exception as e:
            log_error(f"Error running trading bot: {e}")
            raise

    def backtest(self, start_date: str, end_date: str) -> pd.DataFrame:
        from core.backtester import simulate_trades
        from core.features import add_all_features
        from core.risk_manager import adjust_risk_based_on_profile
        from core.logger import log_message, log_error
        try:
            if not self.selected_coins:
                self.select_coins()
            if not self.selected_coins:
                log_error("No tradable symbols available for backtest. Please check your exchange connectivity or configuration.")
                return pd.DataFrame()

            strategy = self.config.get("strategy")
            if not strategy:
                strategy = self.auto_select_strategy()
                self.config["strategy"] = strategy

            exchange = self.config.get("exchange")
            timeframe = self.config.get("timeframe", "5m")
            capital = self.config.get("capital", 1000)
            risk_profile = self.config.get("risk_profile", "medium")

            self.config["risk_profile"] = risk_profile
            adjust_risk_based_on_profile(self.config)

            log_message(f"Backtesting {strategy} on {len(self.selected_coins)} symbols [{start_date} to {end_date}]")

            results = []
            for symbol in self.selected_coins:
                try:
                    df = simulate_trades(
                        symbol=symbol,
                        strategy_name=strategy,
                        exchange_name=exchange,
                        start_date=start_date,
                        end_date=end_date,
                        timeframe=timeframe,
                        capital=capital,
                        config=self.config,
                        ml_filter=self.ml_filter
                    )
                    if df is not None:
                        df = add_all_features(df)
                        results.append(df)
                except Exception as symbol_e:
                    log_error(f"Backtest failed for {symbol}: {symbol_e}")

            if results:
                full = pd.concat(results, ignore_index=True)
                log_message(f"Backtest complete for {len(self.selected_coins)} symbols.")
                return full
            else:
                log_message("No results from backtest.")
                return pd.DataFrame()
        except Exception as e:
            log_error(f"Error during backtesting: {e}")
            return pd.DataFrame()

    def run(self, mode: str = "live", **kwargs):
        from core.logger import log_message
        log_message(f"Orchestrator run mode: {mode}")
        if mode in {"live", "dry_run"}:
            self.run_trading()
        elif mode == "backtest":
            start_date = kwargs.get("start_date")
            end_date = kwargs.get("end_date")
            if not start_date or not end_date:
                raise ValueError("start_date and end_date must be supplied for backtest mode.")
            return self.backtest(start_date, end_date)
        else:
            raise ValueError(f"Unknown mode: {mode}")

_orch_instance: Optional[TradingOrchestrator] = None

def get_orchestrator(config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> TradingOrchestrator:
    global _orch_instance
    if _orch_instance is None:
        _orch_instance = TradingOrchestrator(config_path=config_path, config=config)
    return _orch_instance