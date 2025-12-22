import os
from typing import NamedTuple

import requests
from dotenv import load_dotenv

load_dotenv()


class Response(NamedTuple):
    response: str
    thought: str


class APIClient:
    def __init__(self) -> None:
        self._complete_url = os.environ.get(
            "BASE_URL", "https://api.openai.com/v1/chat/completions"
        )
        assert os.environ.get("API_KEY"), (
            "You must provide an API key in your environment variables!"
        )
        self._api_key = os.environ.get("API_KEY", "")
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _query(self, data: dict) -> tuple[bool, Response]:
        response = requests.post(self._complete_url, json=data, headers=self._headers)
        error = ""
        if response.status_code == 200:
            try:
                resp = str(response.json()["choices"][0]["message"]["content"])
                think = str(
                    response.json()["choices"][0]["message"]["reasoning_content"]
                )
            except Exception as e:
                error = str(e)
                raise e
            return True, Response(resp, think)
        return False, Response(error, "")

    def query(
        self,
        model: str,
        messages: list,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs,
    ):
        retries = 0
        while retries < max_retries:
            status, response = self._query({
                "model": model,
                "messages": messages,
                "timeout": timeout,
                **kwargs,
            })
            if status:
                return response
            retries += 1
        raise RuntimeError("MAX RETRIES triggered. Failed to get response.")
