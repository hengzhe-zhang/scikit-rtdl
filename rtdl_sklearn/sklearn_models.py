import numpy as np
import torch.optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping

from .modules import FTTransformer
from .dcn2 import DCNv2
from .mlp import MLP
from .resnet import ResNet


def get_categorical_feature_index(x, threshold=5):
    # automatically determine the categorical feature
    x_num_dim = []
    x_cat_dim = []
    x_cat_cardinalities = []
    for k in range(x.shape[1]):
        count = len(np.unique(x[:, k]))
        if count <= threshold:
            x_cat_dim.append(k)
            x_cat_cardinalities.append(count)
        else:
            x_num_dim.append(k)
    return x_num_dim, x_cat_dim, x_cat_cardinalities


class MLPBase(BaseEstimator, RegressorMixin):
    def __init__(self, patience=10):
        self.x_transformer = None
        self.y_transformer = None
        self.net = None
        self.patience = patience

    def data_preprocess(self, X, y):
        x = X.astype(np.float32)
        y = y.astype(np.float32)
        x_num_dim, x_cat_dim, x_cat_cardinalities = get_categorical_feature_index(X, threshold=5)
        x_transformer = ColumnTransformer(
            [('cat_cols', OrdinalEncoder(handle_unknown='use_encoded_value',
                                         unknown_value=np.nan), x_cat_dim),
             ('num_cols', StandardScaler(), x_num_dim)])
        self.x_transformer = x_transformer
        x = x_transformer.fit_transform(x).astype(np.float32)
        y_transformer = StandardScaler()
        self.y_transformer = y_transformer
        y = y_transformer.fit_transform(np.reshape(y, (-1, 1)))
        x_cat_dim = [i for i in range(len(x_cat_dim))]
        x_num_dim = [i for i in range(len(x_cat_dim), len(x_cat_dim) + len(x_num_dim))]
        return x, y, x_num_dim, x_cat_dim, x_cat_cardinalities

    def predict(self, X):
        x = self.x_transformer.transform(X).astype(np.float32)
        x = np.nan_to_num(x)
        y = self.y_transformer.inverse_transform(self.net.predict(x))
        return np.nan_to_num(y, posinf=0, neginf=0)


class FTTransformerRegressor(MLPBase):
    def __init__(self, n_blocks=3, d_token=128, attention_dropout=0.1, ffn_dropout=0):
        super().__init__()
        self.n_blocks = n_blocks
        self.d_token = d_token
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout

    def fit(self, X, y):
        x, y, x_num_dim, x_cat_dim, x_cat_cardinalities = self.data_preprocess(X, y)
        model = FTTransformer.make_default(
            n_num_features=len(x_num_dim),
            cat_cardinalities=x_cat_cardinalities,
            feature_index=(x_num_dim, x_cat_dim),
            n_blocks=self.n_blocks,
            d_token=self.d_token,
            attention_dropout=self.attention_dropout,
            ffn_dropout=self.ffn_dropout,
            last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
            d_out=1,
        )
        net = NeuralNetRegressor(
            model,
            max_epochs=200,
            lr=1e-3,
            optimizer=torch.optim.Adam,
            callbacks=[EarlyStopping(patience=self.patience)],
            verbose=False,
        )
        self.net = net
        net.fit(x, y)


class MLPRegressor(MLPBase):
    def __init__(self,
                 layers=2,
                 layer_size=32,
                 dropout=0,
                 categorical_embedding_size=8):
        super().__init__()
        self.layers = layers
        self.layer_size = layer_size
        self.d_layers = [int(layer_size), ] * int(layers)
        self.dropout = dropout
        self.categorical_embedding_size = int(categorical_embedding_size)

    def fit(self, X, y):
        x, y, x_num_dim, x_cat_dim, x_cat_cardinalities = self.data_preprocess(X, y)
        model = MLP(
            d_in=len(x_num_dim),
            d_layers=self.d_layers,
            d_out=1,
            dropout=self.dropout,
            categories=x_cat_cardinalities,
            d_embedding=self.categorical_embedding_size,
            feature_index=(x_num_dim, x_cat_dim),
        )
        net = NeuralNetRegressor(
            model,
            max_epochs=200,
            lr=1e-3,
            optimizer=torch.optim.Adam,
            callbacks=[EarlyStopping(patience=self.patience)],
            verbose=False,
        )
        self.net = net
        net.fit(x, y)


class DCNV2Regressor(MLPBase):
    def __init__(self,
                 cross_layers=2,
                 hidden_layers=2,
                 layer_size=32,
                 hidden_dropout=0,
                 cross_dropout=0,
                 categorical_embedding_size=8):
        super().__init__()
        self.layer_size = int(layer_size)
        self.cross_layers = int(cross_layers)
        self.hidden_layers = int(hidden_layers)
        self.hidden_dropout = hidden_dropout
        self.cross_dropout = cross_dropout
        self.categorical_embedding_size = int(categorical_embedding_size)

    def fit(self, X, y):
        x, y, x_num_dim, x_cat_dim, x_cat_cardinalities = self.data_preprocess(X, y)
        model = DCNv2(
            d_in=len(x_num_dim),
            d=self.layer_size,
            n_hidden_layers=self.hidden_layers,
            n_cross_layers=self.cross_layers,
            hidden_dropout=self.hidden_dropout,
            cross_dropout=self.cross_dropout,
            d_out=1,
            stacked=True,
            categories=x_cat_cardinalities,
            d_embedding=self.categorical_embedding_size,
            feature_index=(x_num_dim, x_cat_dim),
        )
        net = NeuralNetRegressor(
            model,
            max_epochs=200,
            lr=1e-3,
            optimizer=torch.optim.Adam,
            callbacks=[EarlyStopping(patience=self.patience)],
            verbose=False,
        )
        self.net = net
        net.fit(x, y)


class ResNetRegressor(MLPBase):
    def __init__(self,
                 hidden_layers=2,
                 layer_size=32,
                 d_hidden_factor=1,
                 hidden_dropout=0,
                 residual_dropout=0,
                 categorical_embedding_size=8):
        super().__init__()
        self.layer_size = int(layer_size)
        self.d_hidden_factor = d_hidden_factor
        self.hidden_layers = int(hidden_layers)
        self.hidden_dropout = hidden_dropout
        self.residual_dropout = residual_dropout
        self.categorical_embedding_size = int(categorical_embedding_size)

    def fit(self, X, y):
        x, y, x_num_dim, x_cat_dim, x_cat_cardinalities = self.data_preprocess(X, y)
        model = ResNet(
            d_numerical=len(x_num_dim),
            categories=x_cat_cardinalities,
            d_embedding=self.categorical_embedding_size,
            d=self.layer_size,
            d_hidden_factor=self.d_hidden_factor,
            n_layers=self.hidden_layers,
            activation='reglu',
            normalization='batchnorm',
            hidden_dropout=self.hidden_dropout,
            residual_dropout=self.residual_dropout,
            d_out=1,
            feature_index=(x_num_dim, x_cat_dim),
        )
        net = NeuralNetRegressor(
            model,
            max_epochs=200,
            lr=1e-3,
            optimizer=torch.optim.Adam,
            callbacks=[EarlyStopping(patience=self.patience)],
            verbose=False,
        )
        self.net = net
        net.fit(x, y)
