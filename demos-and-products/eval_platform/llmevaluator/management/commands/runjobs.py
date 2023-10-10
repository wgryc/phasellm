from django.core.management.base import BaseCommand

from llmevaluator.models import *

from django.conf import settings
from phasellm.llms import OpenAIGPTWrapper, ChatBot


# Returns the new ChatBotMessageArray ID
def run_llm_task_and_save(
    message_array,
    user_message,
    job_id,
    original_title="Untitled",
    model="gpt-4",
    temperature=0.7,
    print_response=True,
    new_system_prompt=None,
):
    o = OpenAIGPTWrapper(settings.OPENAI_API_KEY, model=model, temperature=temperature)
    cb = ChatBot(o, "")

    ma_copy = message_array.copy()
    if new_system_prompt is not None:
        # If the first message is not a system prompt, then error out.
        assert ma_copy[0]["role"] == "system"
        ma_copy[0]["content"] = new_system_prompt

    cb.messages = ma_copy
    response = cb.chat(user_message)

    new_cbma = ChatBotMessageArray(
        message_array=cb.messages,
        source_batch_job_id=job_id,
        title=f"{original_title} w/ T={temperature}, model={model}",
    )

    new_cbma.llm_temperature = temperature
    new_cbma.llm_model = model

    new_cbma.save()

    if print_response:
        print(response)

    return new_cbma.id


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

        # SETTING: run_n_times
        run_n_times = job.run_n_times
        for i in range(0, run_n_times):
            # SETTING: include_gpt_4
            if job.include_gpt_4:
                if job.temperature_range:
                    for t in [0.25, 0.75, 1.25]:
                        run_llm_task_and_save(
                            cbma.message_array.copy(),
                            job.user_message,
                            job.id,
                            cbma.title,
                            model="gpt-4",
                            temperature=t,
                            new_system_prompt=job.new_system_prompt,
                        )
                else:
                    run_llm_task_and_save(
                        cbma.message_array.copy(),
                        job.user_message,
                        job.id,
                        cbma.title,
                        "gpt-4",
                        new_system_prompt=job.new_system_prompt,
                    )

            # SETTING: include_gpt_35
            if job.include_gpt_35:
                if job.temperature_range:
                    for t in [0.25, 0.75, 1.25]:
                        run_llm_task_and_save(
                            cbma.message_array.copy(),
                            job.user_message,
                            job.id,
                            cbma.title,
                            model="gpt-3.5-turbo",
                            temperature=t,
                            new_system_prompt=job.new_system_prompt,
                        )
                else:
                    run_llm_task_and_save(
                        cbma.message_array.copy(),
                        job.user_message,
                        job.id,
                        cbma.title,
                        "gpt-3.5-turbo",
                        new_system_prompt=job.new_system_prompt,
                    )

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
