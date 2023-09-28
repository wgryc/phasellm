from django.core.management.base import BaseCommand

from llmevaluator.models import *

from django.conf import settings
from phasellm.llms import OpenAIGPTWrapper, ChatBot


def run_job(job):
    print(f"Starting job: {job.title}")

    mc = MessageCollection.objects.get(id=job.message_collection_id)
    chat_ids_string = mc.chat_ids
    chat_ids = chat_ids_string.strip().split(",")

    for _cid in chat_ids:
        print(f"Analyzing chat ID: {_cid}")

        cid = int(_cid)
        cbma = ChatBotMessageArray.objects.get(id=cid)

        o = OpenAIGPTWrapper(settings.OPENAI_API_KEY, model="gpt-4")
        cb = ChatBot(o, "")
        cb.messages = cbma.message_array
        response = cb.chat(job.user_message)
        print(response)

    print("Done!")


class Command(BaseCommand):
    help = "Runs all scheduled batch jobs."

    def handle(self, *args, **options):
        jobs = BatchLLMJob.objects.filter(status="scheduled")
        for job in jobs:
            run_job(job)
