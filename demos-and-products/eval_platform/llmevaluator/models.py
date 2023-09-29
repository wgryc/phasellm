from django.db import models
from django.core.serializers.json import DjangoJSONEncoder


def object_has_tag(model_object, tag_string):
    tags = model_object.tags.split(",")
    for tag in tags:
        if tag.strip() == tag_string:
            return True
    return False


class ChatBotMessageArray(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    message_array = models.JSONField(default=dict, encoder=DjangoJSONEncoder)
    comments = models.TextField(default="", null=True, blank=True)
    source_batch_job_id = models.IntegerField(null=True, blank=True)
    tags = models.TextField(default="", null=True, blank=True)

    def __str__(self):
        return f"ChatBotMessage (ID {self.id})"


class MessageCollection(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    title = models.TextField(default="", null=True, blank=True)

    # Note: we should use an ArrayField or JSONField or a ManyToManyField if we scale this up.
    # However, to keep things very simple and supportable in SQLite, we'll assume the chat_ids are in a comma-separated string for now. We'll do some basic validation when saving via the front-end.
    chat_ids = models.TextField(default="", null=True, blank=True)

    # We can save source collections in cases where we have batch jobs run.
    source_collection_id = models.IntegerField(null=True, blank=True)
    source_batch_job_id = models.IntegerField(null=True, blank=True)
    tags = models.TextField(default="", null=True, blank=True)

    def __str__(self):
        return f"MessageCollection (ID {self.id}), {self.title}"


class BatchLLMJob(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    title = models.TextField(default="", null=True, blank=True)
    message_collection_id = models.IntegerField()
    user_message = models.TextField(default="", null=True, blank=True)

    # scheduled, complete
    status = models.TextField(default="scheduled", null=True, blank=True)
    tags = models.TextField(default="", null=True, blank=True)

    def __str__(self):
        return f"Batch LLM Job (ID {self.id}), {self.title}"
