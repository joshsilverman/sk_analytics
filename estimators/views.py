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
import json
import hashlib

logger = logging.getLogger(__name__)

@csrf_exempt
def train(request):
    try:
        # features_to_targets_raw = request.FILES['ids_features_targets'].readlines()
        # embed()
        sk_account_id = 1 # int(request.POST['sk_account_id'])
        # embed()
        # features_to_targets = json.loads("".join(features_to_targets_raw))
        # embed()

        features_to_targets = [request.POST['instance_id'], 
            {   'continuous': [1,3,5,1.5],
                'categorical': {
                    'hair_color': 'brown',
                    'tshirt_size': 'L'
            }}, request.POST['rating']]

        trainer_str = request.POST.get('trainer', '')
        # embed()
        e, created = Estimator.objects.get_or_create(
            trainer=trainer_str,
            sk_account_id=sk_account_id,
        )

        if (e.features == ''):
            e.features = []

        f = e.features
        f.append(features_to_targets)
        e.features = f
        e.save()

        trainer = trainer_selector.select_trainer(trainer_str)
        cross_validator = CrossValidator(trainer)
        cross_validator.cross_validate(e.features)

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
def rebuild_model(request):
    try:
        sk_account_id = 1 # int(request.POST['sk_account_id'])
        courses = json.loads(request.POST['courses'])
        print request.POST['courses']

        features_to_targets = []
        for course in courses:
            try:
                score = float(course['score']) + 1
            except:
                score = 0

            example = {
                'continuous': [1],
                'categorical': {
                    'code': course['code']
                }
            }

            try:
                s = course['keywords']
                r = s.lower().split(' ')
                for t in r:
                    example['categorical'][t] = 1
            except:
                pass


            id = int(hashlib.sha1(course['code']).hexdigest(), 16) % (10 ** 8)
            features_to_targets.append([id, example, score])


        trainer_str = request.POST.get('trainer', '')
        # embed()
        e, created = Estimator.objects.get_or_create(
            trainer=trainer_str,
            sk_account_id=sk_account_id,
        )

        e.features = features_to_targets
        e.save()

        trainer = trainer_selector.select_trainer(trainer_str)
        cross_validator = CrossValidator(trainer)
        cross_validator.cross_validate(e.features)

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
