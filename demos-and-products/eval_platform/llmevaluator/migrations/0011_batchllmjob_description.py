# Generated by Django 4.2 on 2023-10-11 10:44

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("llmevaluator", "0010_batchllmjob_resend_last_user_message"),
    ]

    operations = [
        migrations.AddField(
            model_name="batchllmjob",
            name="description",
            field=models.TextField(blank=True, null=True),
        ),
    ]
