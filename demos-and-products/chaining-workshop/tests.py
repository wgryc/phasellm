response_1 = """---FIXED:
你好！今天是个好日子。 (Your sentence was already correct in grammar and syntax.)

---RESPONSE:
你好！是的，今天天气很好。

---ENGLISH:
Hello! Yes, the weather is very good today."""

response_2 = """---MESSAGE
I'm sorry to hear about the discomfort you're experiencing. Is there a specific part of the shoe that's causing the blisters or is it more of a general issue? Also, how does the overall comfort and fit compare to other sneakers you've worn in the past?
---SENTIMENT-SCORE
40
---END
No"""

from demo import *

print(parseResponse(response_1))
print(parseResponse(response_2))