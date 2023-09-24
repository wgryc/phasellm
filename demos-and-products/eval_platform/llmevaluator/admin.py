from django.contrib import admin

from .models import ChatBotMessageArray, MessageCollection

admin.site.register(ChatBotMessageArray)
admin.site.register(MessageCollection)
