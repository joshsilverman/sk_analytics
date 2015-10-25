# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('estimators', '0004_auto_20150615_1942'),
    ]

    operations = [
        migrations.AddField(
            model_name='estimator',
            name='mse',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='estimator',
            name='r',
            field=models.FloatField(default=0.0),
        ),
    ]
