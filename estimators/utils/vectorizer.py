from IPython import embed
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
from estimators.utils.imputer import Imputer
import math

class Vectorizer:
    def __init__(self, training_features_to_targets, features_to_targets):
        self.event_ids = [int(features_to_target[0]) for features_to_target in features_to_targets]
        self.targets = [features_to_target[2] for features_to_target in features_to_targets]

        self.training_categorical_features = [features_to_target[1]['categorical'] for features_to_target in training_features_to_targets]
        self.training_continuous_features = [features_to_target[1]['continuous'] for features_to_target in training_features_to_targets]
        imputer = Imputer()

        raw_continuous_features = [features_to_target[1]['continuous'] for features_to_target in features_to_targets]
        continuous_features = imputer.impute_continuous(self.training_continuous_features, raw_continuous_features)
        # continuous_features = self.scale(imputed_continuous_features)

        raw_categorical_features = [features_to_target[1]['categorical'] for features_to_target in features_to_targets]
        categorical_features = imputer.impute_categorical(self.training_categorical_features, raw_categorical_features)

        self.features = self.vectorize(categorical_features, continuous_features)

    # def scale(self, continuous_features):
    #     scaler = StandardScaler()
    #
    #     imputer = Imputer()
    #     imputed_training_continuous_features = imputer.impute_continuous(self.training_continuous_features, self.training_continuous_features)
    #     scaler.fit(imputed_training_continuous_features)
    #     scaled_continuous_features = scaler.transform(continuous_features)
    #
    #     return scaled_continuous_features

    def vectorize(self, categorical_features, continuous_features):
        vec = DictVectorizer(sparse=False)

        vec.fit(self.training_categorical_features)
        enc_categorical_features = vec.transform(categorical_features)
        merged_features = []
        for cont, cat in zip(continuous_features, enc_categorical_features):
            all_features_for_item = list(cont) + list(cat)
            # TODO why do I need this ||0 ? -- why isn't the imputer handling this?
            merged_features.append([0.0 if math.isnan(y) else y for y in all_features_for_item])

        return merged_features
