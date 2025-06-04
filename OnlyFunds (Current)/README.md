# OnlyFunds Trading Bot

**Merged with other bits n bobs.**

## Overview

OnlyFunds is an automated crypto trading bot and research platform, featuring:
- An interactive Streamlit dashboard (`main.py`)
- Orchestration CLI for full ML/backtest/live pipeline (`orchestrator_cli.py`)
- Modular strategy and ML signal selection
- Risk profile and run mode selection (live/dry_run)
- Backtesting, order management, and result export

## Directory Structure

```
OnlyFunds (Current)/
├── core/                   # Trading engine, ML, orchestration, and strategies
├── main.py                 # Streamlit dashboard
├── orchestrator_cli.py     # Command-line orchestrator interface
├── config.yaml             # Main configuration
├── requirements.txt        # Python dependencies
├── START.bat               # Double-click to launch Streamlit app (Windows)
├── tests/                  # Test suite
├── ...                     # Data, logs, strategies, etc.
```

## Quick Start (Windows)

1. **Install Python 3.9+** if not already installed.

2. **Install dependencies** (automatically run by `START.bat` if missing):
    ```
    pip install -r requirements.txt
    ```

3. **Configure API keys and settings**:  
   Edit `config.yaml` with your exchange keys and preferred trading parameters.

4. **Launch the Streamlit dashboard**:
    - Double-click `START.bat` **OR**
    - Run manually:
      ```
      cd "OnlyFunds (Current)"
      streamlit run main.py
      ```

5. **Use the dashboard** to:
    - Select risk profile and mode
    - Run strategies and orchestrator pipeline
    - Export orders or close open positions
    - View recent orders and backtest results

## Command-Line Orchestration

For advanced users, run the orchestrator directly:

```
cd "OnlyFunds (Current)"
python orchestrator_cli.py run --profile normal --dry-run
```

Other CLI options:
- `--live` to force live trading mode
- `--no-retrain` to skip ML retraining
- `--schedule` to run every hour (loop)
- `status` to view orchestrator logs
- `metrics` to view last pipeline metrics

## Configuration

`config.yaml` sample:

```yaml
all_strategies:
  - ema
  - rsi
  - macd
  - bollinger
  - trend_score

ml_filter: true
ml_threshold: 0.6

symbols_source: top_200_coinex
timeframe: 5min
limit: 300
initial_capital: 10
threshold: 0.5
live: false

coinex:
  api_key: "YOUR_API_KEY"
  api_secret: "YOUR_SECRET"

trailing_stop_pct: 0.03
trailing_trigger_pct: 0.02
```

## Dependencies

All dependencies are specified in `requirements.txt`:

```
streamlit>=1.20.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
PyYAML>=6.0
pytest>=7.0.0
python-dateutil>=2.8.0
# If you use API/database for data, add:
# requests>=2.28.0
# SQLAlchemy>=1.4.0
```

## Testing

From the `OnlyFunds (Current)` directory, run:
```
pytest
```

## Notes

- **API keys**: Never commit your real API keys to this repository.
- **Data files**: Some features require historical data CSVs in a `data/` directory.
- **Logs**: Review `orchestrator.log` and dashboard logs for troubleshooting.

---

**For source and module details, see:**  
- [core/ directory listing](https://github.com/Sgtbananas/OF2/tree/main/OnlyFunds%20(Current)/core)
- [main.py](https://github.com/Sgtbananas/OF2/blob/main/OnlyFunds%20(Current)/main.py)
- [orchestrator_cli.py](https://github.com/Sgtbananas/OF2/blob/main/OnlyFunds%20(Current)/orchestrator_cli.py)
- [requirements.txt](https://github.com/Sgtbananas/OF2/blob/main/OnlyFunds%20(Current)/requirements.txt)