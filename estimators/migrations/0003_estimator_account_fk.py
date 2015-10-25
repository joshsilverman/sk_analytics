# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('estimators', '0002_auto_20150612_1551'),
    ]

    operations = [
        migrations.AddField(
            model_name='estimator',
            name='account_fk',
            field=models.IntegerField(default=0),
        ),
    ]
