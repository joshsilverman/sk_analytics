# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('estimators', '0006_estimator_serialized_trained_model'),
    ]

    operations = [
        migrations.RenameField(
            model_name='estimator',
            old_name='account_fk',
            new_name='sk_account_id',
        ),
    ]
