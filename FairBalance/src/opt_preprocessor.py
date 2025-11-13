import numpy as np

class OPTPreprocessor:
    """
    Local-DP mechanism on A (binary) using Theorem 1.
    - We DO NOT modify X or y; we only privatize A->Z on TRAIN.
    - Fairness on TEST is computed w.r.t TRUE A (standard practice).
    """
    def __init__(self, epsilon=1.0, random_state=0, enforce_order=True):
        self.epsilon = float(epsilon)
        self.rng = np.random.RandomState(random_state)
        self.enforce_order = enforce_order
        self._swapped = False
        self.p_star = None
        self.q_star = None

    def _group_stats(self, y, a):
        a = np.asarray(a).astype(int)
        y = np.asarray(y).astype(int)
        p0 = np.mean(a == 0)
        p1 = 1.0 - p0
        y0 = y[a == 0]; y1 = y[a == 1]
        p10 = y0.mean() if y0.size else 0.0
        p11 = y1.mean() if y1.size else 0.0
        return p0, p1, p10, p11

    def fit(self, y_train, a_train):
        p0, p1, p10, p11 = self._group_stats(y_train, a_train)
        # ensure p_{1|0} <= p_{1|1}
        if self.enforce_order and (p10 > p11):
            self._swapped = True
            a_train = 1 - np.asarray(a_train).astype(int)
            p0, p1, p10, p11 = self._group_stats(y_train, a_train)

        c = 1.0 - np.exp(-self.epsilon)/2.0  # 1 - e^{-ε}/2
        if p0 < p1:
            self.p_star, self.q_star = c, 0.5
        else:
            self.p_star, self.q_star = 0.5, c
        return self

    def transform(self, a):
        if self.p_star is None:
            raise RuntimeError("Call fit() before transform().")
        a = np.asarray(a).astype(int)
        if self._swapped:
            a = 1 - a

        u = self.rng.rand(a.shape[0])
        z = np.empty_like(a)

        m0 = (a == 0)             # P(Z=0|A=0)=p*
        z[m0] = (u[m0] > self.p_star).astype(int)

        m1 = ~m0                  # P(Z=1|A=1)=q*
        z[m1] = (u[m1] <= self.q_star).astype(int)

        if self._swapped:
            z = 1 - z
        return z

    def fit_transform(self, y_train, a_train):
        self.fit(y_train, a_train)
        return self.transform(a_train)
