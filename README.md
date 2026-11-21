# ML-Trading

**ML-Trading** is a modular Python pipeline for researching, training, evaluating and backtesting quantitative trading strategies with triple-barrier labels.

The pipeline downloads historical ETF data from Alpaca, creates technical features, assigns profit, stop-loss or time-limit targets, trains three machine-learning models and compares their Sharpe ratios.

## Project Structure

```text
ML-Trading/
├── src/
│   ├── main.py                 # Application entry point
│   ├── config.py               # Central hyperparameter configuration
│   ├── ML/
│   │   ├── Data_imports.py     # Downloaded datasets
│   │   └── Models.py           # Model definitions
│   ├── data_treatment/
│   │   ├── data_loader.py      # Alpaca data loading
│   │   ├── data_split.py       # Feature, label and train/test generation
│   │   ├── data_visualization.py
│   │   └── metadata.py         # Technical indicators
│   └── backtest/
│       ├── backtest.py         # Training and model comparison
│       └── backtest_metrics.py # Returns and Sharpe ratios
├── data/
│   └── backtest_results.csv    # Historical test results
├── .vscode/
│   ├── tasks.json              # Run task
│   ├── launch.json             # F5 debug configuration
│   └── settings.json           # Project interpreter
├── .env                        # Private Alpaca credentials
├── pyproject.toml              # Project dependencies
├── uv.lock                     # Reproducible dependency versions
├── .gitignore
└── README.md
```

## Requirements

- Python 3.14 or later
- [`uv`](https://docs.astral.sh/uv/)
- Alpaca Paper Trading API credentials

The Python environment and dependencies are managed with uv:

```bash
uv sync
```

## Alpaca Configuration

Create a `.env` file at the repository root:

```env
ALPACA_API_KEY=your_api_key
ALPACA_API_SECRET=your_api_secret
```

The `.env` file is ignored by Git. Never commit API credentials.

## Run the Project

Run the complete pipeline from the repository root:

```bash
uv run python -m src.main
```

The VS Code default build task is available with `Ctrl+Shift+B`. The project can also be launched with `F5` using the **ML-Trading: Executer** configuration.

## Pipeline

1. [`data_loader.py`](/workspaces/ML-Trading/src/data_treatment/data_loader.py) downloads daily bars from Alpaca.
2. [`metadata.py`](/workspaces/ML-Trading/src/data_treatment/metadata.py) calculates technical indicators.
3. [`data_split.py`](/workspaces/ML-Trading/src/data_treatment/data_split.py) creates rolling feature windows and triple-barrier targets.
4. [`Models.py`](/workspaces/ML-Trading/src/ML/Models.py) defines XGBoost, Random Forest and LightGBM.
5. [`backtest.py`](/workspaces/ML-Trading/src/backtest/backtest.py) trains and evaluates every model.
6. [`backtest_metrics.py`](/workspaces/ML-Trading/src/backtest/backtest_metrics.py) calculates trade returns and Sharpe ratios.
7. [`main.py`](/workspaces/ML-Trading/src/main.py) assembles and starts the complete pipeline.

## Central Configuration

All data, feature, labeling, backtest and model settings are defined in [`src/config.py`](/workspaces/ML-Trading/src/config.py). Change the values in `CONFIG = ProjectConfig()` and rerun the application.

### Data

| Parameter | Current value | Meaning |
|---|---:|---|
| `symbols` | SPY, DIA, SPLV, TLT | Alpaca tickers downloaded for the experiment |
| `start_date` | 2020-01-01 | Beginning of the downloaded history |
| `end_date` | 2023-01-01 | End of the downloaded history |
| `timeframe` | day | Resolution of each market observation |

### Features

| Parameter | Current value | Meaning |
|---|---:|---|
| `sma_window` | 20 | Period used for the 20-bar simple moving average |
| `atr_short_window` | 20 | Period used for the short ATR |
| `atr_long_window` | 3 | Period used for the long ATR |
| `volume_zscore_window` | 20 | Number of bars used to normalize volume by a rolling mean and standard deviation |

The model receives the ATR, ATR ratio, distance to SMA, volume z-score, VWAP distance, Wick ratio, body ratio and Garman-Klass value.

### Triple-barrier labels

| Parameter | Current value | Meaning |
|---|---:|---|
| `horizon_bars` | 12 | Maximum holding horizon in bars |
| `lookback_bars` | 50 | Number of consecutive bars provided to the model |
| `stop_loss_atr` | 1.0 | Stop barrier placed one ATR below entry |
| `profit_target_atr` | 3.0 | Profit barrier placed three ATR above entry |
| `train_ratio` | 0.80 | Share of chronological samples assigned to training |
| `embargo_bars` | 62 | Gap removed between train and test to reduce label overlap |
| `window_step_bars` | 5 | Distance between the start of two consecutive feature windows |

The target is `0` if the stop-loss is reached first, `1` if the horizon expires before any barrier, and `2` if the profit target is reached first.

### Backtest

| Parameter | Current value | Meaning |
|---|---:|---|
| `sharpe_window` | 20 | Rolling window used for the rolling Sharpe calculation |
| `fee_pct` | 0.001 | Cost deducted from every executed trade |
| `trading_days_per_year` | 252 | Number used to annualize the backtest frequency |
| `skip_overlapping_trades` | true | Prevents opening a trade again during an active horizon |

### XGBoost

| Parameter | Current value | Effect |
|---|---:|---|
| `max_depth` | 3 | Restricts tree depth to reduce overfitting |
| `learning_rate` | 0.03 | Makes boosting corrections gradual |
| `subsample` | 0.80 | Uses 80% of observations per boosting round |
| `colsample_bytree` | 0.80 | Uses 80% of features for each tree |
| `n_estimators` | 100 | Number of boosting trees |
| `random_state` | 42 | Makes stochastic operations reproducible |
| `n_jobs` | -1 | Uses all available CPU threads |

### Random Forest

| Parameter | Current value | Effect |
|---|---:|---|
| `n_estimators` | 200 | Number of independent decision trees |
| `max_depth` | 5 | Limits each tree depth |
| `class_weight` | balanced | Gives less common classes more influence |
| `random_state` | 42 | Makes the random trees reproducible |
| `n_jobs` | -1 | Uses all available CPU threads |

### LightGBM

| Parameter | Current value | Effect |
|---|---:|---|
| `n_estimators` | 100 | Number of boosting rounds |
| `learning_rate` | 0.03 | Makes boosting corrections gradual |
| `max_depth` | 4 | Limits tree depth |
| `num_leaves` | 15 | Maximum leaves per tree |
| `subsample` | 0.80 | Uses 80% of observations per round |
| `colsample_bytree` | 0.80 | Uses 80% of features per tree |
| `class_weight` | balanced | Gives less common classes more influence |
| `random_state` | 42 | Makes stochastic operations reproducible |
| `n_jobs` | -1 | Uses all available CPU threads |

## Results Storage

Each completed pipeline run appends one row per ETF/model pair to [`data/backtest_results.csv`](/workspaces/ML-Trading/data/backtest_results.csv).

The CSV includes:

- UTC run identifier;
- ETF symbol;
- model name;
- Sharpe ratio;
- all label, feature, backtest and data-configuration values;
- the best model for the run;
- the best model's average Sharpe ratio.

Run identifiers allow multiple experiments to be compared without overwriting previous results.

## Import Conventions

Run modules from the repository root with Python's `-m` option so package-relative imports resolve correctly:

```bash
uv run python -m src.main
```

For imports inside `src`, prefer relative imports:

```python
from .Models import Models
from ..backtest.backtest import Best_overall
```

The repository uses `uv.lock` to record resolved dependency versions. It should be committed and shared with the project. The `.venv` directory is generated locally and must remain ignored.

## Data and Validation Notes

The checked Alpaca datasets contain 756 rows each for SPY, DIA, SPLV and TLT, from 2020-01-02 through 2022-12-30. They have no duplicate dates, missing values or invalid OHLC combinations.

The model currently uses a fixed chronological split and 700 test samples for each ETF. Accuracy alone is not a sufficient validation metric: the Sharpe ratio and out-of-sample robustness should be reviewed together. Additional years, walk-forward validation and multiple runs are recommended before real-world use.
