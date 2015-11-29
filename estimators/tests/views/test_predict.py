from django.test import TestCase
from django.test import Client
import json
from IPython import embed
from estimators.models import *
import logging

class PredictTestCase(TestCase):
    def test_returns_200_if_valid(self):
        c = Client()

        # train
        with open('estimators/tests/fixtures/ids_features_targets.json') as fp:
            response = c.post('/estimators/train/', {
                'sk_account_id': '1',
                'ids_features_targets': fp,
            })
            estimator_id = json.loads(response.content)['id']
            self.assertEqual(response.status_code, 200)

        # predict - with same feature obj
        with open('estimators/tests/fixtures/ids_features_targets.json') as fp:
            response = c.post('/estimators/predict/', {
                'estimator_id': estimator_id,
                'ids_feature_vectors': fp,
            })
            self.assertEqual(response.status_code, 200)

    def test_returns_400_if_invalid_estimator_id(self):
        c = Client()

        with open('estimators/tests/fixtures/ids_features_targets.json') as fp:
            logging.disable(logging.CRITICAL)
            response = c.post('/estimators/predict/', {
                'estimator_id': 123,
                'ids_feature_vectors': fp,
            })
            self.assertEqual(response.status_code, 400)

    def test_returns_400_if_estimator_id_not_provided(self):
        c = Client()

        with open('estimators/tests/fixtures/ids_features_targets.json') as fp:
            logging.disable(logging.CRITICAL)
            response = c.post('/estimators/predict/', {
                'ids_feature_vectors': fp,
            })
            self.assertEqual(response.status_code, 400)

    def test_returns_400_if_feature_vectors_not_provided(self):
        c = Client()

        with open('estimators/tests/fixtures/ids_features_targets.json') as fp:
            logging.disable(logging.CRITICAL)
            response = c.post('/estimators/predict/', {
                'estimator_id': 123,
            })
            self.assertEqual(response.status_code, 400)

    def test_ids_to_predictions_in_response_with_error_estimate(self):
        c = Client()

        # train
        with open('estimators/tests/fixtures/ids_features_targets.json') as fp:
            response = c.post('/estimators/train/', {
                'sk_account_id': '1',
                'ids_features_targets': fp,
            })
            self.assertEqual(response.status_code, 200)
            estimator_id = json.loads(response.content)['id']

        # predict
        with open('estimators/tests/fixtures/ids_features_targets.json') as fp:
            response = c.post('/estimators/predict/', {
                'estimator_id': estimator_id,
                'ids_feature_vectors': fp,
            })
            self.assertEqual(response.status_code, 200)
            response_obj = json.loads(response.content)
            ids_to_predictions = response_obj['ids_to_predictions']

            self.assertEqual(len(ids_to_predictions.keys()), 1232)
