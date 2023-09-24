from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.http import HttpResponse, JsonResponse

from .models import *

import json


@require_http_methods(["POST"])
def createMessageArray(request):
    data = json.loads(request.body)
    if "messages" in data:
        json_messages = json.loads(data["messages"])
        cbma = ChatBotMessageArray(message_array=json_messages)
        cbma.save()
        return JsonResponse({"status": "ok"})
    return JsonResponse({"status": "error", "message": "Unknown error."}, status=500)


@require_http_methods(["POST"])
def createGroupFromCSV(request):
    data = json.loads(request.body)
    if "messagelist" in data:
        messages_csv = data["messagelist"]
        ids = messages_csv.strip().split(",")
        all_present = True
        for chat_id in ids:
            o = ChatBotMessageArray.objects.filter(id=chat_id)
            if len(o) != 1:
                all_present = False

        if all_present:
            mc = MessageCollection(
                title="New Collection", chat_ids=messages_csv.strip()
            )
            mc.save()
            return JsonResponse({"status": "ok"})
        else:
            return JsonResponse(
                {
                    "status": "error",
                    "message": "Not all IDs are present in the data; please review and try again.",
                },
                status=500,
            )

    return JsonResponse({"status": "error", "message": "Unknown error."}, status=500)
