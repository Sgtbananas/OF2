import logging
import pandas as pd
from typing import Optional, Dict, Any, List, Union
from core.config import load_config
from core.engine import run_bot
from core.ml_filter import MLFilter
from core.coin_selector import get_top_200_coinex_symbols
from core.risk_manager import adjust_risk_based_on_profile
from core.logger import log_message, log_error
from core.backtester import simulate_trades

class TradingOrchestrator:
    """
    Orchestrates trading and backtesting operations using configuration, 
    ML filtering, coin selection, and risk management.
    """

    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the orchestrator with configuration.
        """
        if config is not None:
            self.config = config
            log_message("Config provided directly to orchestrator.")
        else:
            self.config = load_config(config_path)
            log_message(f"Config loaded from {config_path or 'default location'}.")

        self.ml_filter: Optional[MLFilter] = None
        self.selected_coins: List[str] = []
        self._init_ml_filter()

    def _init_ml_filter(self):
        """Initialize ML filter if enabled in config."""
        ml_cfg = self.config.get("ml_filter", {})
        if ml_cfg.get("enabled", False):
            try:
                self.ml_filter = MLFilter(model_path=ml_cfg.get("model_path", "ml_filter_model.pkl"))
                log_message("ML filter initialized.")
            except Exception as e:
                log_error(f"Failed to initialize ML filter: {e}")
                self.ml_filter = None

    def select_coins(self):
        """Selects coins to trade based on exchange and config."""
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
        """Run the live or dry-run trading bot."""
        try:
            strategy = self.config.get("strategy")
            exchange = self.config.get("exchange")
            timeframe = self.config.get("timeframe", "5m")
            capital = self.config.get("capital", 1000)
            risk_profile = self.config.get("risk_profile", "medium")

            if not self.selected_coins:
                self.select_coins()

            adjust_risk_based_on_profile(self.config, risk_profile)
            log_message(f"Running bot for {len(self.selected_coins)} symbols on {exchange} ({timeframe}).")

            run_bot(
                strategy=strategy,
                symbols=self.selected_coins,
                exchange=exchange,
                timeframe=timeframe,
                capital=capital,
                config=self.config,
                ml_filter=self.ml_filter
            )
        except Exception as e:
            log_error(f"Error running trading bot: {e}")
            raise

    def backtest(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Run a backtest across selected coins.

        Args:
            start_date: ISO8601 date, e.g. '2024-01-01'
            end_date: ISO8601 date, e.g. '2024-05-31'

        Returns:
            DataFrame with backtest results.
        """
        try:
            strategy = self.config.get("strategy")
            exchange = self.config.get("exchange")
            timeframe = self.config.get("timeframe", "5m")
            capital = self.config.get("capital", 1000)
            risk_profile = self.config.get("risk_profile", "medium")

            if not self.selected_coins:
                self.select_coins()

            adjust_risk_based_on_profile(self.config, risk_profile)
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
        """
        Entrypoint for orchestrator.

        Args:
            mode: 'live', 'dry_run', or 'backtest'
            kwargs: Additional arguments for backtesting (start_date, end_date)
        """
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

# Optional: global factory for integration
_orch_instance: Optional[TradingOrchestrator] = None

def get_orchestrator(config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> TradingOrchestrator:
    global _orch_instance
    if _orch_instance is None:
        _orch_instance = TradingOrchestrator(config_path=config_path, config=config)
    return _orch_instance