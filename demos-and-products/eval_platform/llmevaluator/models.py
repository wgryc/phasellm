from django.db import models
from django.core.serializers.json import DjangoJSONEncoder


class ChatBotMessageArray(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    message_array = models.JSONField(default=dict, encoder=DjangoJSONEncoder)
    comments = models.TextField(default="", null=True, blank=True)
    source_batch_job_id = models.IntegerField(null=True)


class MessageCollection(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    title = models.TextField(default="", null=True, blank=True)

    # Note: we should use an ArrayField or JSONField or a ManyToManyField if we scale this up.
    # However, to keep things very simple and supportable in SQLite, we'll assume the chat_ids are in a comma-separated string for now. We'll do some basic validation when saving via the front-end.
    chat_ids = models.TextField(default="", null=True, blank=True)

    # We can save source collections in cases where we have batch jobs run.
    source_collection_id = models.IntegerField(null=True)
    source_batch_job_id = models.IntegerField(null=True)


class BatchLLMJob(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    title = models.TextField(default="", null=True, blank=True)
    message_collection_id = models.IntegerField()
    user_message = models.TextField(default="", null=True, blank=True)

    # scheduled, complete
    status = models.TextField(default="scheduled", null=True, blank=True)
