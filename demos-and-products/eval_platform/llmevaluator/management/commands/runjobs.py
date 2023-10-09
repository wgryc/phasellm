from django.core.management.base import BaseCommand

from llmevaluator.models import *

from django.conf import settings
from phasellm.llms import OpenAIGPTWrapper, ChatBot


def run_job(job):
    print(f"Starting job: {job.title}")

    mc = MessageCollection.objects.get(id=job.message_collection_id)
    chat_ids_string = mc.chat_ids
    chat_ids = chat_ids_string.strip().split(",")

    results_ids = []

    for _cid in chat_ids:
        print(f"Analyzing chat ID: {_cid}")

        cid = int(_cid)
        cbma = ChatBotMessageArray.objects.get(id=cid)

        o = OpenAIGPTWrapper(settings.OPENAI_API_KEY, model="gpt-4")
        cb = ChatBot(o, "")
        cb.messages = cbma.message_array
        response = cb.chat(job.user_message)

        new_cbma = ChatBotMessageArray(
            message_array=cb.messages, source_batch_job_id=job.id
        )

        new_cbma.save()
        results_ids.append(str(new_cbma.id))

        print(response)

    new_chats_str = ",".join(results_ids)
    results_mc = MessageCollection(
        title=f"Results from '{job.title}' job",
        chat_ids=new_chats_str,
        source_collection_id=mc.id,
        source_batch_job_id=job.id,
    )
    results_mc.save()

    job.status = "complete"
    job.save()

    print("Done!")


class Command(BaseCommand):
    help = "Runs all scheduled batch jobs."

    def handle(self, *args, **options):
        jobs = BatchLLMJob.objects.filter(status="scheduled")
        for job in jobs:
            run_job(job)
