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
    title = models.TextField(default="Untitled", blank=True)

    # LLM settings for review, later
    llm_model = models.TextField(default="None", blank=True, null=True)
    llm_temperature = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"ChatBotMessage (ID {self.id}), {self.title}"


class MessageCollection(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    title = models.TextField(default="", null=True, blank=True)

    # Note: we should use an ArrayField or JSONField or a ManyToManyField if we scale this up.
    # However, to keep things very simple and supportable in SQLite, we'll assume the chat_ids are in a comma-separated string for now. We'll do some basic validation when saving via the front-end.
    chat_ids = models.TextField(default="", null=True, blank=True)
    chats = models.ManyToManyField(ChatBotMessageArray, blank=True)

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
    description = models.TextField(null=True, blank=True)
    message_collection_id = models.IntegerField()
    message_collection_ref = models.ForeignKey(
        MessageCollection,
        on_delete=models.SET_NULL,
        null=True,
        related_name="source_messages_collection",
    )
    results_array = models.ForeignKey(
        MessageCollection,
        on_delete=models.SET_NULL,
        null=True,
        related_name="results_collection",
    )

    # scheduled, complete
    status = models.TextField(default="scheduled", null=True, blank=True)
    tags = models.TextField(default="", null=True, blank=True)

    # settings
    # By default we only run the LLM on GPT-4 with a user message. The
    # settings below let you do other things.

    # Messages
    user_message = models.TextField(default="", null=True, blank=True)
    new_system_prompt = models.TextField(default="", null=True, blank=True)
    resend_last_user_message = models.BooleanField(default=False)

    # Repeat the run 'n' times
    run_n_times = models.IntegerField(default=1)

    # Which LLM models to run
    include_gpt_4 = models.BooleanField(default=True)
    include_gpt_35 = models.BooleanField(default=False)

    # Run temperature tests; True = run across 0.25 to 1.75 with 0.5 increments
    temperature_range = models.BooleanField(default=False)

    def __str__(self):
        return f"Batch LLM Job (ID {self.id}), {self.title}"
