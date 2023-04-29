# NewsBot

An autonomous news summarizer. You can set this up to execute regularly and it will email you a summary of news articles for a given period, on specific queries or topics.

## Installation and Setup

You need `phasellm` installed; no additional packages need to be installed. However, you do need to have...

- An OpenAI API key
- A GMail account (we'll use this to send news summaries)
- A newsapi.org API key

Set up a .env file with the above, as follows:

```
OPENAI_API_KEY=<OpenAI API key>
NEWS_API_API_KEY=<newsapi.org API key>
GMAIL_EMAIL=<GMail address that will send emails>
GMAIL_PASSWORD=<Password for the above>
```

Note that you'll likely need to set up an [app password](https://myaccount.google.com/apppasswords) for your GMail account, rather than using your actual password. This is something GMail requires for security purposes (and it's a great idea!). [Learn more here.](https://support.google.com/mail/answer/185833)

## Running

Once you've done the above, simply run `python newsbot.py` and you're good to go!
