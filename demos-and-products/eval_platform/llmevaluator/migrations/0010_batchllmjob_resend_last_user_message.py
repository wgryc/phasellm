# Generated by Django 4.2 on 2023-10-11 06:30

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("llmevaluator", "0009_batchllmjob_new_system_prompt_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="batchllmjob",
            name="resend_last_user_message",
            field=models.BooleanField(default=False),
        ),
    ]
