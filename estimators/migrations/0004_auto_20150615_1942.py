# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('estimators', '0003_estimator_account_fk'),
    ]

    operations = [
        migrations.AddField(
            model_name='estimator',
            name='mean_mse',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='estimator',
            name='mean_r',
            field=models.FloatField(default=0.0),
        ),
    ]
