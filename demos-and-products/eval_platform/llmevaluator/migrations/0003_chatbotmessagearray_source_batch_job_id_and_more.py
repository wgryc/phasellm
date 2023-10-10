# Generated by Django 4.2 on 2023-09-28 14:54

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("llmevaluator", "0002_batchllmjob"),
    ]

    operations = [
        migrations.AddField(
            model_name="chatbotmessagearray",
            name="source_batch_job_id",
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name="messagecollection",
            name="source_batch_job_id",
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name="messagecollection",
            name="source_collection_id",
            field=models.IntegerField(null=True),
        ),
    ]
