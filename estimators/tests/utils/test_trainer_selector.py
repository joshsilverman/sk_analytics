from django.test import TestCase
from IPython import embed
from estimators.trainers.elasticnet import *
from estimators.utils import trainer_selector

class TrainerSelectorTestCase(TestCase):
    def test_returns_elasticnet_as_default(self):
        trainer = trainer_selector.select_trainer('')
        self.assertEqual(trainer, Elasticnet)

    def test_returns_elasticnet(self):
        trainer = trainer_selector.select_trainer('elasticnet')
        self.assertEqual(trainer, Elasticnet)

    def test_returns_elasticnet_case_insensitive(self):
        trainer = trainer_selector.select_trainer('Elasticnet')
        self.assertEqual(trainer, Elasticnet)
