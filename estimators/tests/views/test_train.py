from django.test import TestCase
from django.test import Client
import json
from IPython import embed
from estimators.models import *
import logging

class TrainTestCase(TestCase):
    def test_returns_200_if_valid(self):
        c = Client()

        with open('estimators/tests/fixtures/ids_features_targets.json') as fp:
            response = c.post('/estimators/train/', {
                'sk_account_id': '1',
                'ids_features_targets': fp,
            })

            self.assertEqual(response.status_code, 200)

    def test_creates_estimator(self):
        c = Client()
        self.assertEqual(Estimator.objects.count(), 0)

        with open('estimators/tests/fixtures/ids_features_targets.json') as fp:
            response = c.post('/estimators/train/', {
                'sk_account_id': '1',
                'ids_features_targets': fp,
            })

        self.assertEqual(Estimator.objects.count(), 1)

    def test_returns_400_if_no_file_attached(self):
        c = Client()
        logging.disable(logging.CRITICAL)
        with open('estimators/tests/fixtures/ids_features_targets.json') as fp:
            response = c.post('/estimators/train/', {'sk_account_id': '1'})

            self.assertEqual(response.status_code, 400)

    def test_returns_400_if_no_sk_account_id_provided(self):
        c = Client()
        logging.disable(logging.CRITICAL)
        with open('estimators/tests/fixtures/ids_features_targets.json') as fp:
            response = c.post('/estimators/train/', {'ids_features_targets': fp})

            self.assertEqual(response.status_code, 400)
