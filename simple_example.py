from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score

from rtdl_sklearn.sklearn_models import MLPRegressor

X, y = load_boston(return_X_y=True)
mlp = MLPRegressor()
print(cross_val_score(mlp, X, y, n_jobs=-1))
