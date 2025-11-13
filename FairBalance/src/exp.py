from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from load_data import load
import numpy as np
from preprocessor import *
from predata import *
from metrics import Metrics

# NEW: import your OPT mechanism
from opt_preprocessor import OPTPreprocessor


class Exp:
    def __init__(self, data, treatment="None", predata="None", inject=None):
        #  Load data
        self.data, self.A = load(data)
        # Separate independent variables and dependent variables
        independent = self.data.keys().tolist()
        dependent = independent.pop(-1)
        self.X = self.data[independent]
        self.y = np.array(self.data[dependent])
        self.treatment = treatment
        self.predata = predata
        self.inject = inject  # optional: can carry epsilon or other knobs
        self.clf = LogisticRegression(max_iter=100000)

    def data_stats(self):
        groups_class = {}
        for i in range(len(self.y)):
            key_class = tuple([self.X[a][i] for a in self.A])
            if key_class not in groups_class:
                groups_class[key_class] = {1: 0, 0: 0}
            groups_class[key_class][self.y[i]] += 1
        distrs = [groups_class[key][1] / (groups_class[key][0] + groups_class[key][1]) for key in groups_class]
        return np.max(distrs) - np.min(distrs)

    # NEW: no-op bias injector so non-synthetic runs don’t fail
    def inject_bias(self, X, y):
        """
        No-op for real datasets. If you later want synthetic bias injection,
        implement it here (e.g., flip labels by group using self.inject).
        """
        return y

    def one_exp(self, epsilon_opt=None, random_state_opt=None):
        """
        If self.treatment == 'OPT', we:
          - privatize the sensitive attribute on TRAIN to Z_train via OPT (epsilon_opt or 1.0)
          - replace sensitive column(s) in X_train with Z_train for *training-time* logic
          - still compute metrics on TRUE A (original X frames).
        For all other treatments, behavior is identical to the original code.
        """
        # Split
        X_train, X_test, y_train, y_test = self.train_test_split(test_size=0.3)

        # ========= OPT branch (only when selected) =========
        if self.treatment == "OPT":
            # pick epsilon and rng
            eps = 1.0
            if epsilon_opt is not None:
                eps = float(epsilon_opt)
            elif isinstance(self.inject, (int, float)):
                eps = float(self.inject)

            rs = 0 if random_state_opt is None else int(random_state_opt)

            # Build Z_train using OPT on TRUE A (train split)
            # Use the first sensitive column by default (most datasets have one; some have two like sex/race).
            sens_cols = list(self.A)
            a_train = X_train[sens_cols[0]].astype(int).to_numpy()

            opt = OPTPreprocessor(epsilon=eps, random_state=rs)
            Z_train = opt.fit_transform(y_train, a_train)  # privatized sensitive attribute on train

            # Make a training copy whose sensitive column is replaced by Z
            X_train_priv = X_train.copy()
            X_train_priv.loc[:, sens_cols[0]] = Z_train

            # ======== Pre data (RUNS ON PRIVATE GROUPS) ========
            X_train_work, y_train_work = self.pre_data(X_train_priv, y_train)

            # ======== Column transformer fit using PRIVATE groups frame ========
            self.data_preprocess(X_train_work)

            # ======== Bias injection (no-op for real datasets) on PRIVATE groups frame ========
            y_train_biased = self.inject_bias(X_train_work, y_train_work)

            # ======== Treatment (Reweighing / FairBalance) uses PRIVATE groups frame ========
            sample_weight = self.treat(X_train_work, y_train_biased)

            # ======== Fit classifier on PRIVATE groups frame ========
            self.fit(X_train_work, y_train_biased, sample_weight)

            # ======== Metrics on TRUE A (original frames) ========
            m_train = Metrics(self.clf, X_train, y_train, self.A, self.preprocessor)
            m_test = Metrics(self.clf, X_test, y_test, self.A, self.preprocessor)
            return m_train, m_test

        # ========= Non-OPT branch (original behavior) =========
        ####### Pre data #######################
        X_train2, y_train2 = self.pre_data(X_train, y_train)
        #########################################
        self.data_preprocess(X_train2)
        #########################################
        y_train_biased = self.inject_bias(X_train2, y_train2)
        #########################################

        sample_weight = self.treat(X_train2, y_train_biased)
        self.fit(X_train2, y_train_biased, sample_weight)
        m_train = Metrics(self.clf, X_train, y_train, self.A, self.preprocessor)
        m_test = Metrics(self.clf, X_test, y_test, self.A, self.preprocessor)
        return m_train, m_test

    def fit(self, X, y, sample_weight=None):
        X_train_processed = self.preprocessor.fit_transform(X)
        self.clf.fit(X_train_processed, y, sample_weight=sample_weight)

    def data_preprocess(self, X):
        numerical_columns_selector = selector(dtype_exclude=object)
        categorical_columns_selector = selector(dtype_include=object)

        numerical_columns = numerical_columns_selector(X)
        categorical_columns = categorical_columns_selector(X)

        categorical_preprocessor = OneHotEncoder(handle_unknown='ignore')
        numerical_preprocessor = StandardScaler()
        self.preprocessor = ColumnTransformer([
            ('OneHotEncoder', categorical_preprocessor, categorical_columns),
            ('StandardScaler', numerical_preprocessor, numerical_columns)
        ])

    def treat(self, X_train, y_train):
        if self.treatment == "Reweighing":
            sample_weight = Reweighing(X_train, y_train, self.A)
        elif self.treatment == "FairBalanceVariant":
            sample_weight = FairBalanceVariant(X_train, y_train, self.A)
        elif self.treatment == "FairBalance":
            sample_weight = FairBalance(X_train, y_train, self.A)
        else:
            sample_weight = None
        return sample_weight

    def pre_data(self, X_train, y_train):
        if self.predata == "fairway":
            X_new, y_new = fairway(X_train, y_train, self.A)
        elif self.predata == "fairsituation":
            X_new, y_new = fairsituation(X_train, y_train, self.A)
        elif self.predata == "fairsmote":
            X_new, y_new = fairsmote(X_train, y_train, self.A)
        elif self.predata == "fairsmote-situation":
            X_new, y_new = fairsmote(X_train, y_train, self.A)
            X_new, y_new = fairsituation(X_new, y_new, self.A)
        else:
            X_new = X_train
            y_new = y_train
        return X_new, y_new

    def train_test_split(self, test_size=0.3):
        # Split training and testing data proportionally across each group
        groups = {}
        for i in range(len(self.y)):
            key = tuple([self.X[a][i] for a in self.A] + [self.y[i]])
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        train = []
        test = []
        for key in groups:
            testing = list(np.random.choice(groups[key], int(len(groups[key]) * test_size), replace=False))
            training = list(set(groups[key]) - set(testing))
            test.extend(testing)
            train.extend(training)
        X_train = self.X.iloc[train]
        X_test = self.X.iloc[test]
        y_train = self.y[train]
        y_test = self.y[test]
        X_train.index = range(len(X_train))
        X_test.index = range(len(X_test))
        return X_train, X_test, y_train, y_test
