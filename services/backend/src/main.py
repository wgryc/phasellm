import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

import openai

openai.api_key = openai_api_key

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080", # phasellm-ui local dev (hot-reload) UI
        "http://localhost:3000" # phasellm-ui production(compiled) UI
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PhaseAIPrompt(BaseModel):
    text: str

class PhaseAIResponse(BaseModel):
    response: object

@app.get("/")
def home():
    return "Tevs holla"

@app.get("/ping")
def ping():
    return "Phase AI back-end is running..."

@app.post("/prompt/", response_model=PhaseAIResponse)
async def llm_response_for(
    prompt: PhaseAIPrompt
) -> Any:
    print(prompt.text)

    openai_response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt.text,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    return {
        "response": openai_response
    }
