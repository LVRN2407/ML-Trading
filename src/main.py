from .ML.Data_imports import DATASETS
from .ML.Models import Models
from .backtest.backtest import Best_overall
from .config import CONFIG
import time


def main():
    started_at = time.perf_counter()
    best_model, mean_sharpe = Best_overall(
        DATASETS,
        Models,
        T=CONFIG.labels.horizon_bars,
        t=CONFIG.labels.lookback_bars,
        SL=CONFIG.labels.stop_loss_atr,
        NP=CONFIG.labels.profit_target_atr,
        Ts=CONFIG.backtest.sharpe_window,
        SMA=CONFIG.features.sma_window,
        ATRS=CONFIG.features.atr_short_window,
        ATRL=CONFIG.features.atr_long_window,
        ZVOL=CONFIG.features.volume_zscore_window,
        train_ratio=CONFIG.labels.train_ratio,
        embargo_bars=CONFIG.labels.embargo_bars,
        window_step=CONFIG.labels.window_step_bars,
        fee_pct=CONFIG.backtest.fee_pct,
        trading_days_per_year=CONFIG.backtest.trading_days_per_year,
        skip_overlapping_trades=CONFIG.backtest.skip_overlapping_trades,
        max_workers=CONFIG.backtest.max_workers,
    )
    elapsed = time.perf_counter() - started_at
    print(f"BENCHMARK_SECONDS={elapsed:.6f}")
    return best_model, mean_sharpe


if __name__ == "__main__":
    main()
