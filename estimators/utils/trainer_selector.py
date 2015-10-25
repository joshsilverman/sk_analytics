from IPython import embed
from estimators.trainers.elasticnet import *
from estimators.trainers.sgdclassifier import *

def select_trainer(trainer_token):
    if (trainer_token):
        trainer_token_lowercase = trainer_token.lower()
    else:
        trainer_token_lowercase = ''

    trainer_token_map = {
        'elasticnet': Elasticnet,
        'sgdclassifier': SGDClassifier,
    }

    trainer = trainer_token_map.get(trainer_token_lowercase, Elasticnet)
    return trainer
