# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.utils.timezone
import django_extensions.db.fields


class Migration(migrations.Migration):

    dependencies = [
        ('estimators', '0001_initial'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='estimator',
            options={'ordering': ('-modified', '-created'), 'get_latest_by': 'modified'},
        ),
        migrations.AddField(
            model_name='estimator',
            name='created',
            field=django_extensions.db.fields.CreationDateTimeField(default=django.utils.timezone.now, verbose_name='created', editable=False, blank=True),
        ),
        migrations.AddField(
            model_name='estimator',
            name='modified',
            field=django_extensions.db.fields.ModificationDateTimeField(default=django.utils.timezone.now, verbose_name='modified', editable=False, blank=True),
        ),
    ]
