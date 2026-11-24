"""Автоенкодер на sklearn.MLPRegressor (вхід -> 16 -> 6 -> 16 -> вхід); аномалія = помилка реконструкції."""
from sklearn.neural_network import MLPRegressor


class Autoencoder:
    def __init__(self, hidden=(16, 6, 16), alpha=1e-3):
        self.net = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, max_iter=300,
                                early_stopping=True, random_state=42)

    def fit(self, X):
        self.net.fit(X, X)
        return self

    def score(self, X):
        return ((self.net.predict(X) - X) ** 2).mean(axis=1)
