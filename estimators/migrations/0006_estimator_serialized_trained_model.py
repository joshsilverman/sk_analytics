# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('estimators', '0005_auto_20150615_2020'),
    ]

    operations = [
        migrations.AddField(
            model_name='estimator',
            name='serialized_trained_model',
            field=models.BinaryField(blank=True),
        ),
    ]
