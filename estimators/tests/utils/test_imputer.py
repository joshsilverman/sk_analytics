from django.test import TestCase
from IPython import embed
from estimators.utils.imputer import Imputer

class ImputerTestCase(TestCase):
    def test_imputes_continuous_features_using_training_set(self):
        training_features_cont = [
            [0, 0, 1],
            [0, 10, 2],
        ]
        features_cont = [
            [1, None, 0],
        ]

        imputer = Imputer()
        imputed = imputer.impute_continuous(training_features_cont, features_cont)
        self.assertEqual(list(imputed[0]), [1.0, 5.0, 0.0])

    def test_imputes_categorical_features_using_training_set(self):
        training_features_cat = [
            {'hair':'brown','eyes':'blue'},
            {'hair':'black','eyes':'blue'},
        ]
        features_cat = [
            {'hair':'brown','eyes':None},
        ]

        imputer = Imputer()
        imputed = imputer.impute_categorical(training_features_cat, features_cat)
        self.assertEqual(imputed[0], {'hair':'brown','eyes':'blue'})

    def test_compute_category_modes(self):
        training_features_cat = [
            {'hair':'brown','eyes':'blue'},
            {'hair':'black','eyes':'blue'},
        ]

        imputer = Imputer()
        modes = imputer.get_category_modes(training_features_cat)

        self.assertEqual(modes['eyes'], 'blue')
        self.assertEqual(modes['hair'], 'brown')

    def test_removes_continuous_features_when_training_is_nil_for_all_samples_of_a_feature(self):
        training_features_cont = [
            [0, None, 1],
            [0, None, 2],
        ]
        features_cont = [
            [1, 2, 3],
        ]

        imputer = Imputer()
        imputed = imputer.impute_continuous(training_features_cont, features_cont)
        self.assertEqual(list(imputed[0]), [1.0, 3.0])
