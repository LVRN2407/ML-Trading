from .Data_imports import DATASETS
from .Models import Models

from ..backtest.backtest import Best_overall



Best_Model, Mean = Best_overall(DATASETS, Models, T=12, t=50, SL=1, NP=3, Ts=20, SMA=20, ATRS=20, ATRL=3, ZVOL=20)
