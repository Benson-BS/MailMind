from src.config.agent_types import Email
from src.tools.load_email import load_emails


class EmailReader:
    def __init__(self, archive: list[Email]):
        self.by_id = {email["id"]: email for email in archive}

    def read(self, email_id: str) -> Email:
        return self.by_id[email_id]  # 不存在时抛出 KeyError