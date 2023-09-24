from django.db import models


class ChatBotMessageArray(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    message_array = models.JSONField(default=dict)
    comments = models.TextField(default="", null=True, blank=True)


class MessageCollection(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    title = models.TextField(default="", null=True, blank=True)

    # Note: we should use an ArrayField or JSONField or a ManyToManyField if we scale this up.
    # However, to keep things very simple and supportable in SQLite, we'll assume the chat_ids are in a comma-separated string for now. We'll do some basic validation when saving via the front-end.
    chat_ids = models.TextField(default="", null=True, blank=True)
