# Generated by Django 4.2 on 2023-10-11 11:05

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("llmevaluator", "0012_batchllmjob_message_collection_ref"),
    ]

    operations = [
        migrations.AddField(
            model_name="batchllmjob",
            name="results_array",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="results_collection",
                to="llmevaluator.messagecollection",
            ),
        ),
        migrations.AlterField(
            model_name="batchllmjob",
            name="message_collection_ref",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="source_messages_collection",
                to="llmevaluator.messagecollection",
            ),
        ),
    ]