from phasellm.agents import EmailSenderAgent

import os
from dotenv import load_dotenv

load_dotenv()
gmail_email = os.getenv("GMAIL_EMAIL")
gmail_password = os.getenv("GMAIL_PASSWORD")

e = EmailSenderAgent('Wojciech Gryc', 'smtp.gmail.com', gmail_email, gmail_password, 587)
e.sendPlainEmail('wgryc@fastmail.com', 'this is a test', 'How are you?')