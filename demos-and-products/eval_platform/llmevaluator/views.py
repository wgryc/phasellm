from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.http import HttpResponse, JsonResponse

from .models import *

import json


def view_chat(request, chat_id):
    chats = ChatBotMessageArray.objects.filter(id=chat_id)

    if len(chats) != 1:
        return render(
            request,
            "view-chat.html",
            {
                "contenttitle": f"Viewing Chat ID {chat_id}",
                "error_msg": "Chat not found. Are you sure it exists?",
            },
        )

    return render(
        request,
        "view-chat.html",
        {
            "contenttitle": f"Viewing Chat ID {chat_id}",
            "json_message_array": json.dumps(chats[0].message_array),
        },
    )


# Same as createMessageArray() but we don't loads() from messages.
@require_http_methods(["POST"])
def createMessageArrayJson(request):
    data = json.loads(request.body)
    if "messages" in data:
        json_messages = data["messages"]
        cbma = ChatBotMessageArray(message_array=json_messages)
        if "title" in data:
            cbma.title = data["title"]
        cbma.save()
        return JsonResponse({"status": "ok"})
    return JsonResponse({"status": "error", "message": "Unknown error."}, status=500)


@require_http_methods(["POST"])
def createMessageArray(request):
    data = json.loads(request.body)
    print(data)
    if "messages" in data:
        json_messages = json.loads(data["messages"])
        cbma = ChatBotMessageArray(message_array=json_messages)
        if "title" in data:
            cbma.title = data["title"]
        cbma.save()
        return JsonResponse({"status": "ok"})
    return JsonResponse({"status": "error", "message": "Unknown error."}, status=500)


@require_http_methods(["POST"])
def createJob(request):
    data = json.loads(request.body)

    title = ""
    if "title" in data:
        title = data["title"]

    message_collection_id = int(data["message_collection_id"])

    user_message = None
    if "user_message" in data:
        user_message = data["user_message"]

    b = BatchLLMJob(
        title=title,
        message_collection_id=message_collection_id,
        user_message=user_message,
    )
    b.save()

    return JsonResponse({"status": "ok"})


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

        title = "New Collection"
        if "title" in data:
            title = data["title"]

        if all_present:
            mc = MessageCollection(title=title, chat_ids=messages_csv.strip())
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
