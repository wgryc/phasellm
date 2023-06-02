"""
Support for convering LLM-related classes and objects to HTML and various outputs.
"""

# Add: HTML object
# Add: Flask server to show the HTML object + store CSS

import re

def _formatContentToHtml(string):
    new_string = re.sub("<", "&lt;", string)
    new_string = re.sub(">", "&gt;", string)
    new_string = re.sub('[\r\n]+', '<br>', new_string)
    return new_string

def chatbotToHtml(chatbot):
    chatbot_html = "<div class='phasellm_chatbot_stream'>"

    messages = chatbot.messages
    for m in messages:

        m_timestamp = ""
        if "timestamp_utc" in m:
            m_timestamp = m['timestamp_utc'].strftime("%d %B %Y at %H:%M:%S")
        
        m_log_time_seconds = ""
        if "log_time_seconds" in m:
            m_log_time_seconds = str(round(m['log_time_seconds'], 3)) + " seconds"

        response_html = f"""
<div class='response_container'>
    <div class='response_{m['role']}'>{_formatContentToHtml(m['content'])}</div>
    <div class='timestamp'>{m_timestamp}</div>
    <div class='time_taken'>{m_log_time_seconds}</div>
</div>
"""

        chatbot_html += response_html

    chatbot_html += "\n</div>"

    return chatbot_html