from django.contrib import admin

from .models import ChatBotMessageArray, MessageCollection, BatchLLMJob

admin.site.register(ChatBotMessageArray)
admin.site.register(MessageCollection)
admin.site.register(BatchLLMJob)
