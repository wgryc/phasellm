from phasellm.llms import HuggingFaceInferenceWrapper

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")

h = HuggingFaceInferenceWrapper(hugging_face_api_key, model_url="https://api-inference.huggingface.co/models/bigcode/starcoder")

#tc = h.text_completion("Please load a data frame into pandas, generate fake data with rnorm() and print the average.")
#print(tc)

#tc2 = h.text_completion("#Build logistic regression where z ~ y + x\ndef logisticreggression(z, y, x):")
#print(tc2)

"""
prompt = ""
with open('prompt.txt', 'r') as w:
    prompt = w.read()
tc = h.text_completion(prompt)
print(tc)
"""

tc = h.text_completion("def replaceQuestionMarksWithNans(pandas_data_frame):")