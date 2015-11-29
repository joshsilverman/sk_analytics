from django.db import models
from jsonfield import JSONField
from django.contrib import admin
# from IPython import embed; embed()
from django_extensions.db.models import TimeStampedModel


class Estimator(TimeStampedModel):
    features = JSONField()
    sk_account_id = models.IntegerField(default=0)
    trainer = models.CharField(default='',max_length=24)

    # estimation via cross validation
    mean_mse = models.FloatField(default=0.0)
    mean_r = models.FloatField(default=0.0)

    # selected model scores
    mse = models.FloatField(default=0.0)
    r = models.FloatField(default=0.0)

    serialized_trained_model = models.BinaryField(blank=True, editable=False)

admin.site.register(Estimator)
