import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from rtdl_sklearn.sklearn_models import MLPRegressor, FTTransformerRegressor, ResNetRegressor, DCNV2Regressor, \
    FTTransformerClassifier, ResNetClassifier, DCNV2Classifier, MLPClassifier
from rtdl_sklearn.tabnet_model import TabNetClassifier

pd.set_option('display.max_columns', None)

all_score = []
# for data in [load_boston]:
for data in [load_iris]:
    X, y = data(return_X_y=True)
    # for model_func in [FTTransformerRegressor, ResNetRegressor, DCNV2Regressor, MLPRegressor]:
    # for model_func in [FTTransformerClassifier, ResNetClassifier, DCNV2Classifier, MLPClassifier]:
    for model_func in [TabNetClassifier]:
        # for model_func in [MLPClassifier]:
        # for model_func in [XGBClassifier, LGBMClassifier]:
        model = model_func()
        st = time.time()
        # score = cross_val_score(model, np.array(X), np.array(y), n_jobs=-1)
        score = cross_val_score(model, np.array(X), np.array(y))
        cost_time = time.time() - st
        score_ = (data.__name__, model_func.__name__, np.mean(score), cost_time)
        # score_ = (data, model_func.__name__, np.mean(score))
        print(score_)
        all_score.append(score_)
df = pd.DataFrame(all_score, columns=['data', 'model', 'score', 'time'])
print(pd.pivot_table(df, 'score', 'data', 'model'))
print(pd.pivot_table(df, 'time', 'data', 'model'))
# print(cross_val_score(mlp, X, y, n_jobs=-1))
# print(cross_val_score(mlp, X, y))
