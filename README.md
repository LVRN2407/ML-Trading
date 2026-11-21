# ML-Trading

**ML-Trading** is a modular Python machine learning pipeline built to research, train, and evaluate quantitative trading strategies using **triple-barrier labeling**. Rather than relying on fixed time horizons alone, triple-barrier labeling sets upper profit-taking, lower stop-loss, and vertical time-expiration barriers around each entry point—capturing realistic trade outcomes based on path-dependent price action.

The project organizes data ingestion, preprocessing/metadata transformations, model training, and backtesting into separate packages under `src/` so individual stages can be developed, tested, and maintained independently while sharing a consistent data and execution flow.

---

## Project Structure

```text
ML-Trading/
├── src/
│   ├── backtest/
│   │   └── backtest.py
│   ├── data_treatment/
│   │   ├── data_loader.py
│   │   ├── data_split.py
│   │   ├── data_visualization.py
│   │   └── metadata.py
│   └── ML/
│       ├── Data_imports.py
│       ├── Models.py
│       └── Training.py
├── .env
├── .gitignore
└── README.md
```
## What to Import

Imports in this project are split between **third-party libraries** and **internal modules** inside `src/`. Always run commands from the `ML-Trading` root (`py -m src.ML.Training`) so relative imports resolve cleanly.

### 1. Third-Party Packages
* **Environment & data loading:** `os`, `pathlib.Path`, `datetime.datetime`, `dotenv.load_dotenv`, and `pandas` for reading `.env` credentials and handling market data tables.
* **Market data (`alpaca-py`):** `StockHistoricalDataClient`, `StockBarsRequest`, and `TimeFrame` from `alpaca.data.*` to fetch historical bars.
* **Machine learning (`scikit-learn`):** Estimators and model tools from `sklearn` (such as `RandomForestClassifier` in `src/ML/Models.py`).

### 2. Internal Project Imports
* **`src/ML/Data_imports.py`:** Imports dataset helpers from `data_loader.py` (such as `get_dataset`) to build datasets like `SPY` and `DATASETS`.
* **`src/ML/Training.py`:** Uses relative imports to bring in `DATASETS` from `.Data_imports`, estimators from `.Models`, and backtest evaluation functions (such as `Best_overall`) from `..backtest.backtest`.
* **`src/backtest/backtest.py`:** Imports preprocessing/metadata utilities from `..data_treatment.metadata` (such as `standardize`)—make sure the imported function name matches what is actually defined in `metadata.py`.

### Quick Import Rules
* Use **relative imports** (`from .` or `from ..`) between files inside `src/` when running as a module with `py -m ...`.
* Install `scikit-learn` (not `sklearn`) via `pip`, while keeping Python imports as `from sklearn...`.
