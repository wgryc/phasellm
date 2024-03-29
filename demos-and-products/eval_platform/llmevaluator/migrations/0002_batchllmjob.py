# Generated by Django 4.2 on 2023-09-26 18:50

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("llmevaluator", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="BatchLLMJob",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("title", models.TextField(blank=True, default="", null=True)),
                ("message_collection_id", models.IntegerField()),
                ("user_message", models.TextField(blank=True, default="", null=True)),
                (
                    "status",
                    models.TextField(blank=True, default="scheduled", null=True),
                ),
            ],
        ),
    ]
