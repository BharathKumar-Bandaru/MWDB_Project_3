import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder


def projection(v, z=1):
    features_shape = v.shape[0]
    u = np.sort(v)[::-1]
    c = np.cumsum(u) - z
    indexes = np.arange(features_shape) + 1
    cond = u - c / indexes > 0
    rho = indexes[cond][-1]
    theta = c[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


class SVM_custom():
    def __init__(self, C=1, max_iter=50, tol=0.05,
                 random_state=None, verbose=0):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol,
        self.random_state = random_state
        self.verbose = verbose

    def gradient(self, X, y, i):
        g = (np.dot(X[i], self.W.T) + 1)
        g[y[i]] -= 1
        return g

    def violation(self, g, y, i):
        smallest = np.inf
        for k in range(g.shape[0]):
            if k == y[i] and self.dcoef[k, i] >= self.C:
                continue
            elif k != y[i] and self.dcoef[k, i] >= 0:
                continue
            smallest = min(smallest, g[k])
        return g.max() - smallest

    def subproblem(self, g, y, norms, i):
        Ci = np.zeros(g.shape[0])
        Ci[y[i]] = self.C
        beta_hat = norms[i] * (Ci - self.dcoef[:, i]) + g / norms[i]
        z = self.C * norms[i]
        beta = projection(beta_hat, z)

        return Ci - self.dcoef[:, i] - beta / norms[i]

    def train(self, X, y):
        n_samples, n_features = X.shape

        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(y)

        classes_unique = len(self._label_encoder.classes_)
        self.classes_num = np.unique(y).shape[0]
        self.dcoef = np.zeros((classes_unique, n_samples), dtype=np.float64)
        self.W = np.zeros((classes_unique, n_features))

        norm_values = np.sqrt(np.sum(X ** 2, axis=1))

        rs = check_random_state(self.random_state)
        ind = np.arange(n_samples)
        rs.shuffle(ind)

        violation_init = None
        for it in range(self.max_iter):
            violation_sum = 0

            for ii in range(n_samples):
                i = ind[ii]
                if norm_values[i] == 0:
                    continue

                g = self.gradient(X, y, i)
                v = self.violation(g, y, i)
                violation_sum += v

                if v < 1e-18:
                    continue

                delta = self.subproblem(g, y, norm_values, i)

                self.W += (delta * X[i][:, np.newaxis]).T
                self.dcoef[:, i] += delta

            if it == 0:
                violation_init = violation_sum
            violation_ratio = violation_sum / violation_init

            if violation_ratio < self.tol:
                break
        return self

    def predict(self, X):
        decision = np.dot(X, self.W.T)
        pred = decision.argmax(axis=1)
        return self._label_encoder.inverse_transform(pred)

    def get_weights(self, X):
        decision = np.dot(X, self.W.T)
        return decision[:, 0:self.classes_num + 1]