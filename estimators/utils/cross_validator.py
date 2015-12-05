from vectorizer import Vectorizer
from sklearn import cross_validation
from random import shuffle
from IPython import embed

class CrossValidator:
    def __init__(self, model):
        self.model = model()

        self.selected_model = None
        self.selected_model_r = None
        self.selected_model_mse = None

    def cross_validate(self, features_to_targets):
        shuffle(features_to_targets)
        v = Vectorizer(features_to_targets, features_to_targets)
        # kf = cross_validation.KFold(len(v.features), n_folds=1)

        self.mses = []
        self.rs = []
        self.correct_percentage = []

        half = len(features_to_targets)/2
        raw_features_to_train = features_to_targets[:half]
        raw_features_to_validate = features_to_targets[half:]

        v_train = Vectorizer(raw_features_to_train, raw_features_to_train)
        v_validate = Vectorizer(raw_features_to_train, raw_features_to_validate)

        features_to_train = v_train.features
        targets_to_train = v_train.targets
        features_to_validate = v_validate.features
        targets_to_validate = v_validate.targets

        self.model.train(features_to_train, targets_to_train)
        self.model.validate(features_to_validate, targets_to_validate)

        self.mses.append(self.model.mse())
        self.rs.append(self.model.r())

        # self.update_selected_model()
        self.correct_percentage.append(1 - sum(self.model.errors) / float(len(self.model.errors)))

    def update_selected_model(self):
        if len(self.mses) == 1:
            self.selected_model = self.model
        elif self.model.r() >= max(self.mses):
            self.selected_model = self.model

        self.selected_model_r = self.model.r()
        self.selected_model_mse = self.model.mse()

    def mean_mse(self):
        return sum(self.mses) / len(self.mses)

    def mean_r(self):
        return sum(self.rs) / len(self.rs)
