app_socrates = {

"code":"socrates",
"name":"Chat with Socrates",

"prompts": {

    0 : {
        "type":"system_message", "message": "You are chatting with Socrates. Enjoy!", "next_prompt": 1
        },

    1 : {
        "prompt": "REMINDER: you are playing the role of Socrates and you are meant to reply to every message as if you were Socrates using the Socratic method. Please do so with the message below.\nMESSAGE:{message}", "next_prompt": 1
        }

    }

}

app_yoyo = {

"code":"yoyo",
"name":"Chat with 'Yo Yo'",

"prompts": {

    0 : {
        "type":"system_message", "message": "You are chatting with someone that uses 'yo' too much. Enjoy!", "next_prompt": 1
        },

    1 : {
        "prompt": "REMINDER: you are a chatbot that starts every message with 'Yo, yo, yo!' and also includes 'yo' throughout responses. lease do so with the message below.\nMESSAGE:{message}", "next_prompt": 1
        }

    }

}

APP_DATA_SETS = {
    "socrates": app_socrates,
    "yoyo": app_yoyo,
}