import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from ..config import CONFIG


model0 = ['XGBclassifier', ## Very bad for our problem (Sharpe ratio -1.4)
          xgb.XGBClassifier(
    max_depth=CONFIG.xgboost.max_depth,
    learning_rate=CONFIG.xgboost.learning_rate,
    subsample=CONFIG.xgboost.subsample,
    colsample_bytree=CONFIG.xgboost.colsample_bytree,
    n_estimators=CONFIG.xgboost.n_estimators,
    random_state=CONFIG.xgboost.random_state,
    n_jobs=CONFIG.xgboost.n_jobs
)]

model1 = ['RandomForestClassifier',
          RandomForestClassifier(
    n_estimators=CONFIG.random_forest.n_estimators,
    max_depth=CONFIG.random_forest.max_depth,
    class_weight=CONFIG.random_forest.class_weight,
    random_state=CONFIG.random_forest.random_state,
    n_jobs=CONFIG.random_forest.n_jobs
)]

model2 = ['LGBMClassifier',
          LGBMClassifier(
    n_estimators=CONFIG.lightgbm.n_estimators,
    learning_rate=CONFIG.lightgbm.learning_rate,
    max_depth=CONFIG.lightgbm.max_depth,
    num_leaves=CONFIG.lightgbm.num_leaves,
    subsample=CONFIG.lightgbm.subsample,
    colsample_bytree=CONFIG.lightgbm.colsample_bytree,
    class_weight=CONFIG.lightgbm.class_weight,
    verbose=-1,
    random_state=CONFIG.lightgbm.random_state,
    n_jobs=CONFIG.lightgbm.n_jobs
)]

Models = [model0, model1, model2]