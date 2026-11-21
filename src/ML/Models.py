import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


model0 = ['XGBclassifier', ## Very bad for our problem (Sharpe ratio -1.4)
          xgb.XGBClassifier(
    max_depth=3,            # Keep trees shallow so they don't overfit to noise
    learning_rate=0.03,     # Slower learning rate for better generalization
    subsample=0.8,          # Randomly sample rows to prevent dominance by majority classes
    colsample_bytree=0.8,   # Randomly sample features per tree
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)]

model1 = ['RandomForestClassifier',
          RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)]

model2 = ['LGBMClassifier',
          LGBMClassifier(
    n_estimators=100,
    learning_rate=0.03,
    max_depth=4,              # Keep trees shallow to prevent overfitting
    num_leaves=15,            # Must be less than 2^max_depth
    subsample=0.8,            # Row subsampling for robustness
    colsample_bytree=0.8,     # Feature subsampling
    class_weight='balanced',
    verbose=-1,               # Handles class balance natively
    random_state=42,
    n_jobs=-1
)]

Models = [model0, model1, model2]