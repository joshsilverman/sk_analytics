from django.test import TestCase
from IPython import embed
from estimators.utils.vectorizer import Vectorizer

class VectorizerTestCase(TestCase):
    def test_vectorizes_scales_imputes_cont_features(self):
        training_features_to_targets = [
            [1, {'continuous': [0., 0., 1.], 'categorical': {}}, False],
            [1, {'continuous': [0., 10., 20.], 'categorical': {}}, False],
        ]
        features_to_targets = [
            [1, {'continuous': [1., 0., 20.], 'categorical': {}}, None],
        ]

        vectorizer = Vectorizer(training_features_to_targets,features_to_targets)
        vectorized = vectorizer.features

        self.assertEqual(list(vectorized[0]), [1.0, -1., 1.0])

    def test_vectorizes_cat_features(self):
        training_features_to_targets = [
            [1, {'continuous': [0., 0., 1.], 'categorical': {'hair':'brown'}}, False],
            [1, {'continuous': [0., 10., 2.], 'categorical': {'hair':'black'}}, False],
        ]
        features_to_targets = [
            [1, {'continuous': [1., None, 5.], 'categorical': {'hair':'black'}}, None],
        ]

        vectorizer = Vectorizer(training_features_to_targets,features_to_targets)
        vectorized = vectorizer.features

        # note that cat feats are sorted by category=value
        # therefore hair=black comes before hair=brown
        # these vectorized categorical values are appended to the cont vector
        self.assertEqual(list(vectorized[0]), [1.0, 0.0, 7.0, 1.0, 0.0])

    def test_removes_continuous_values_when_always_nil_in_training_set(self):
        training_features_to_targets = [
            [1, {'continuous': [0., None, 1.], 'categorical': {}}, False],
            [1, {'continuous': [0., None, 2.], 'categorical': {}}, False],
        ]
        features_to_targets = [
            [1, {'continuous': [1., 1, 5.], 'categorical': {}}, None],
        ]

        vectorizer = Vectorizer(training_features_to_targets,features_to_targets)
        vectorized = vectorizer.features
        self.assertEqual(list(vectorized[0]), [1.0, 7.0])
