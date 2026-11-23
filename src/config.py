from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    """Parameters for the historical market data downloaded from Alpaca."""

    symbols: tuple[str, ...] = ("SPY", "DIA", "SPLV", "TLT")
    start_date: tuple[int, int, int] = (2020, 1, 1)
    end_date: tuple[int, int, int] = (2023, 1, 1)
    timeframe: str = "day"


@dataclass(frozen=True)
class FeatureConfig:
    """Parameters for the technical indicators used as model features."""

    sma_window: int = 20
    atr_short_window: int = 20
    atr_long_window: int = 3
    volume_zscore_window: int = 20


@dataclass(frozen=True)
class LabelConfig:
    """Parameters for the triple-barrier labels used to build targets."""

    horizon_bars: int = 12
    lookback_bars: int = 50
    stop_loss_atr: float = 1.0
    profit_target_atr: float = 3.0
    train_ratio: float = 0.80
    embargo_bars: int = 62
    window_step_bars: int = 5


@dataclass(frozen=True)
class BacktestConfig:
    """Parameters for the backtest and Sharpe-ratio calculation."""

    sharpe_window: int = 20
    fee_pct: float = 0.001
    trading_days_per_year: int = 252
    skip_overlapping_trades: bool = True
    max_workers: int = 4


@dataclass(frozen=True)
class XGBoostConfig:
    """Settings for shallow trees and gradual XGBoost learning."""

    max_depth: int = 3
    learning_rate: float = 0.03
    subsample: float = 0.80
    colsample_bytree: float = 0.80
    n_estimators: int = 100
    random_state: int = 42
    n_jobs: int = -1


@dataclass(frozen=True)
class RandomForestConfig:
    """Settings for shallow trees and class-balanced Random Forest training."""

    n_estimators: int = 200
    max_depth: int = 5
    class_weight: str = "balanced"
    random_state: int = 42
    n_jobs: int = -1


@dataclass(frozen=True)
class LightGBMConfig:
    """Settings for limited leaves and robust LightGBM sampling."""

    n_estimators: int = 100
    learning_rate: float = 0.03
    max_depth: int = 4
    num_leaves: int = 15
    subsample: float = 0.80
    colsample_bytree: float = 0.80
    class_weight: str = "balanced"
    random_state: int = 42
    n_jobs: int = -1


@dataclass(frozen=True)
class ProjectConfig:
    """Central configuration for the complete ML-Trading pipeline."""

    data: DataConfig = DataConfig()
    features: FeatureConfig = FeatureConfig()
    labels: LabelConfig = LabelConfig()
    backtest: BacktestConfig = BacktestConfig()
    xgboost: XGBoostConfig = XGBoostConfig()
    random_forest: RandomForestConfig = RandomForestConfig()
    lightgbm: LightGBMConfig = LightGBMConfig()


CONFIG = ProjectConfig()
