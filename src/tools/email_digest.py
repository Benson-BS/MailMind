from openai import OpenAI

from src.config.agent_types import Email
from src.tools.email_priority import classify_email


def daily_digest(emails: list[Email], lm: OpenAI) -> str:
    if not emails:
        return "No emails today."

    included = []
    for email in emails:
        category = classify_email(email, lm)["category"]
        if category == "ignore":
            continue

        included.append(
            f"Priority: {category}\n"
            f"From: {email['from']}\n"
            f"Subject: {email['subject']}\n"
            f"Body: {email['body']}"
        )

    if not included:
        return "No relevant emails today."

    response = lm.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "Write an English daily email digest of at most 300 words. "
                    "Cover every urgent item and explicitly name its company, "
                    "project, or location. Do not mention ignored emails."
                ),
            },
            {
                "role": "user",
                "content": "\n\n--- EMAIL ---\n\n".join(included),
            },
        ],
    )
    return response.choices[0].message.content or ""