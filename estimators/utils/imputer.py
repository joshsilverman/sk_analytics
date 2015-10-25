from IPython import embed
import numpy as np
from sklearn import preprocessing
import operator

class Imputer:
    def __init__(self):
        self._modes = None

    def impute_continuous(self, training_features_cont, features_cont):
        sklearn_imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
        sklearn_imputer.fit(training_features_cont)
        imputed_impute_continuous = sklearn_imputer.transform(features_cont)
        return imputed_impute_continuous

    def impute_categorical(self, training_features_cat, features_cat):
        modes = self.get_category_modes(training_features_cat)

        imputed_features_cat = []
        for features_dictionary in features_cat:
            imputed_dictionary = {}
            for category, value in features_dictionary.iteritems():
                if value is None:
                    mode = modes[category]
                    imputed_dictionary[category] = mode
                else:
                    imputed_dictionary[category] = value

            imputed_features_cat.append(imputed_dictionary)

        return imputed_features_cat

    def get_category_modes(self, training_features_cat):
        if self._modes is None:
            counts_by_category = {}
            for features_dictionary in training_features_cat:
                for category, value in features_dictionary.iteritems():
                    counts_by_category.setdefault(category, {})
                    counts_by_category[category].setdefault(value, 0)
                    counts_by_category[category][value] += 1

            self._modes = {}
            for category, counts in counts_by_category.iteritems():
                mode = max(counts.iteritems(), key=operator.itemgetter(1))[0]
                self._modes[category] = mode

            return self._modes
        else:
            return self._modes
