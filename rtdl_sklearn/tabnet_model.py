import numpy as np
from pytorch_tabnet import tab_model
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class TabNetRegressor(RegressorMixin, tab_model.TabNetRegressor):
    max_epochs: int = 200
    verbose: int = 0

    def fit(self, X_train, y_train, eval_set=None, eval_name=None, eval_metric=None, loss_fn=None, weights=0,
            max_epochs=None, patience=10, batch_size=1024, virtual_batch_size=128, num_workers=0, drop_last=False,
            callbacks=None, pin_memory=True, from_unsupervised=None):
        # set n_a equivalent to n_d
        self.n_a = self.n_d
        assert self.n_a == self.n_d
        self.n_a = int(self.n_a)
        self.n_d = int(self.n_d)
        self.n_independent = int(self.n_independent)
        self.n_shared = int(self.n_shared)
        self.n_steps = int(self.n_steps)
        self.verbose = False
        max_epochs = self.max_epochs
        assert max_epochs > 0

        self.x_transformer = StandardScaler()
        self.y_transformer = StandardScaler()
        X_train = self.x_transformer.fit_transform(X_train)
        y_train = self.y_transformer.fit_transform(np.reshape(y_train, (-1, 1)))
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
        eval_set = [(X_valid, y_valid)]
        super().fit(X_train, y_train, eval_set, eval_name, eval_metric, loss_fn, weights, max_epochs, patience,
                    batch_size, virtual_batch_size, num_workers, drop_last, callbacks, pin_memory, from_unsupervised)

    def predict(self, X):
        return self.y_transformer.inverse_transform(super().predict(self.x_transformer.transform(X)))


class TabNetClassifier(ClassifierMixin, tab_model.TabNetClassifier):
    max_epochs: int = 200
    verbose: int = 0

    def fit(self, X_train, y_train, eval_set=None, eval_name=None, eval_metric=None, loss_fn=None, weights=0,
            max_epochs=None, patience=10, batch_size=1024, virtual_batch_size=128, num_workers=0, drop_last=False,
            callbacks=None, pin_memory=True, from_unsupervised=None):
        # set n_a equivalent to n_d
        self.n_a = self.n_d
        assert self.n_a == self.n_d
        self.n_a = int(self.n_a)
        self.n_d = int(self.n_d)
        self.n_independent = int(self.n_independent)
        self.n_shared = int(self.n_shared)
        self.n_steps = int(self.n_steps)
        self.verbose = False
        max_epochs = self.max_epochs
        assert max_epochs > 0

        self.x_transformer = StandardScaler()
        X_train = self.x_transformer.fit_transform(X_train)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
        eval_set = [(X_valid, y_valid)]
        super().fit(X_train, y_train, eval_set, eval_name, eval_metric, loss_fn, weights, max_epochs, patience,
                    batch_size, virtual_batch_size, num_workers, drop_last, callbacks, pin_memory, from_unsupervised)

    def predict(self, X):
        return np.nan_to_num(super().predict(self.x_transformer.transform(X)), posinf=0, neginf=0)
