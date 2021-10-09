# scikit-rtdl

A scikit-learn compatible neural network library based on "Revisiting Tabular Deep Learning" (RTDL). 
## Features
* A scikit-learn compatible package supports 4 neural networks based learning algorithms for tabular data (MLP, ResNet, DCN V2, TabNet, FT-Transformer).
* A scikit-learn compatible package supports automatically identifying categorical features and automatically scaling features to facilitate neural network training.

## Usage

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score

from rtdl_sklearn.sklearn_models import MLPRegressor

X, y = load_boston(return_X_y=True)
mlp = MLPRegressor()
print(cross_val_score(mlp, X, y, n_jobs=-1))
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)