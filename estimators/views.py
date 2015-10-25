from django.views.decorators.csrf import csrf_exempt
from django.http import *
from django.utils.datastructures import MultiValueDictKeyError

import json
from models import *
from utils.cross_validator import CrossValidator
import pickle
from utils.vectorizer import Vectorizer
from utils import trainer_selector
from IPython import embed
import logging

logger = logging.getLogger(__name__)

@csrf_exempt
def train(request):
    try:
        features_to_targets_raw = request.FILES['ids_features_targets'].readlines()
        sk_account_id = int(request.POST['sk_account_id'])
        features_to_targets = json.loads("".join(features_to_targets_raw))
        trainer_str = request.POST.get('trainer', '')

        e, created = Estimator.objects.get_or_create(
            trainer=trainer_str,
            features=features_to_targets,
            sk_account_id=sk_account_id,
        )
        e.save()

        trainer = trainer_selector.select_trainer(trainer_str)
        cross_validator = CrossValidator(trainer)
        cross_validator.cross_validate(features_to_targets)

        e.mean_mse = cross_validator.mean_mse()
        e.mean_r = cross_validator.mean_r()
        e.serialized_trained_model = pickle.dumps(cross_validator.selected_model)
        e.save()

        resp = {
            'id': e.id,
            'trainer': trainer.__name__,
            'sk_account_id': sk_account_id,
            'mean_mse': cross_validator.mean_mse(),
            'mean_r': cross_validator.mean_r(),
            'selected_model_mse': cross_validator.selected_model_mse,
            'selected_model_r': cross_validator.selected_model_r,
        }

        return HttpResponse(json.dumps(resp), content_type="application/json")

    except Exception as e:
        # embed()
        logger.exception(str(e))
        return HttpResponseBadRequest(str(e))


@csrf_exempt
def predict(request):
    try:
        estimator_id = int(request.POST['estimator_id'])
        ids_feature_vectors = json.loads("".join(request.FILES['ids_feature_vectors'].readlines()))
        e = Estimator.objects.get(pk=estimator_id)
        training_features_to_targets = e.features

        sklearn_trained_model = pickle.loads(e.serialized_trained_model)
        v = Vectorizer(training_features_to_targets, ids_feature_vectors)
        sklearn_trained_model.predict(v.features)

        ids_to_predictions = dict(zip(v.event_ids, sklearn_trained_model.predictions))
        resp = {
            'id': e.id,
            'mean_r': e.mean_r,
            'ids_to_predictions': ids_to_predictions,
        }

    except Exception as e:
        logger.exception(str(e))
        return HttpResponseBadRequest(str(e))

    return HttpResponse(json.dumps(resp), content_type="application/json")
