import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

from ..data_treatment.data_split import standardize  
from .backtest_metrics import sharpe_ratio  



def train_predict(model, Xtrain, Ytrain, Xtest, Ytest):
    model.fit(Xtrain, Ytrain)
    Ypred = model.predict(Xtest)
    Ytest = Ytest
    return (Ypred, Ytest ,Xtest)


def model_backtest(MODELS, df, T, t, SL, NP, Ts, Xtrain, Ytrain, Xtest_scaled, Ytest_orig, Xi_test_df, SMA=20, ATRS=20, ATRL=3, ZVOL=20):
    # Unpack including the pristine Xi_test DataFrame
    Sharpe_ratios = []
    for Model in MODELS:
        print(f"\nEvaluating: {Model[0]}")
        model = Model[1]

        # Run prediction using the scaled features
        (Ypred, _, _) = train_predict(model, Xtrain, Ytrain, Xtest_scaled, Ytest_orig)
        print(classification_report(Ytest_orig, Ypred, zero_division=0))

        # Pass Xi_test_df (which has the correct columns & _df_idx) into sharpe_ratio!
        (SHR, shr) = sharpe_ratio(Xi_test_df, df, Ypred, Ytest_orig, T, Ts)

        # Shift labels ONLY for visualization plotting (-1, 0, 1)
        Ypred_plot = Ypred - 1
        Ytest_plot = Ytest_orig - 1

        plt.figure(figsize=(14, 6))
        plt.subplot(2, 1, 1)
        plt.plot(Ytest_plot[500:800], label='True Signals (Ytest)', color='royalblue', drawstyle='steps-mid')
        plt.ylabel('Signal Value')
        plt.title(f'{Model[0]} - Performance: Actual vs. Predicted Signals Over Time')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.yticks([-1, 0, 1])

        plt.subplot(2, 1, 2)
        plt.plot(Ypred_plot[500:800], label='Predicted Signals (Ypred)', color='darkorange', drawstyle='steps-mid', linewidth=1.5)
        plt.xlabel('Time Step / Index')
        plt.ylabel('Signal Value')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.yticks([-1, 0, 1])

        plt.tight_layout()
        plt.show()

        print('Total sharpe ratio : ', shr)
        Sharpe_ratios.append([Model[0], shr])

    return Sharpe_ratios


def Multi_backtest(DATASETS, Models, T, t, SL, NP, Ts, SMA=20, ATRS=20, ATRL=3, ZVOL=20):
    Scores = []
    for dataset in DATASETS:
        df = dataset[1]
        dfname = dataset[0]
        (Xtrain, Ytrain, Xtest_scaled, Ytest_orig, Xi_test_df) = standardize(df,T, t, SL, NP, SMA, ATRS, ATRL, ZVOL)
        Scores.append([dfname, model_backtest(Models, df, T, t, SL, NP, Ts, Xtrain, Ytrain, Xtest_scaled, Ytest_orig, Xi_test_df, SMA, ATRS, ATRL, ZVOL)])
    return Scores

#Sharpe_ratios = [[ Name of model, Shape Ratio ], ... ]
#Scores = [ [Dataset name, [ [model name, model score], ... ], [Dataset name, [ [model name, model score], ... ], ... ]

def Best_overall(DATASETS, MODELS, T, t, SL, NP, Ts, SMA=20, ATRS=20, ATRL=3, ZVOL=20):
    Scores = Multi_backtest(DATASETS, MODELS, T, t, SL, NP, Ts, SMA, ATRS, ATRL, ZVOL)
    all_models = {}
    Datasets = []
    for Dataset, Sharpe_ratios in Scores:

        Datasets.append(Dataset)

        for model, model_score in Sharpe_ratios :

            if model in all_models:
                pass
            else:
                all_models[model] = []

        for model, model_score in Sharpe_ratios :

            all_models[model].append(model_score)
    Mean = -9999
    for model, score in all_models.items():

        mean = np.mean(score)
        if mean > Mean:
            Mean = mean
            best_model = model

    print(' The best model overall for all the given datasets : \n', Datasets,
          '\n Is ', best_model, ' with an average sharpe ratio of : ', Mean )

    return best_model, Mean

