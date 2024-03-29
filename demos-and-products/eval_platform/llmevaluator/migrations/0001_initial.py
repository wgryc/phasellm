# Generated by Django 4.2 on 2023-09-24 16:39

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="ChatBotMessageArray",
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
                ("message_array", models.JSONField(default=dict)),
                ("comments", models.TextField(blank=True, default="", null=True)),
            ],
        ),
        migrations.CreateModel(
            name="MessageCollection",
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
                ("chat_ids", models.TextField(blank=True, default="", null=True)),
            ],
        ),
    ]
