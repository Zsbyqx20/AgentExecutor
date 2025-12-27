import logging
import os
from typing import NamedTuple

import requests
from dotenv import load_dotenv

from .icons import CHECK, REFRESH, TIMES

load_dotenv()


class Response(NamedTuple):
    response: str
    thought: str


class APIClient:
    def __init__(self, prefix: str = "") -> None:
        base_url_name = prefix.upper() + "_BASE_URL" if prefix else "BASE_URL"
        api_key_name = prefix.upper() + "_API_KEY" if prefix else "API_KEY"
        self._complete_url = os.environ.get(
            base_url_name, "https://api.openai.com/v1/chat/completions"
        )
        assert os.environ.get(api_key_name), (
            "You must provide an API key in your environment variables!"
        )
        self._api_key = os.environ.get("API_KEY", "")
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _query(
        self, data: dict, logger: logging.Logger | None = None
    ) -> tuple[bool, Response]:
        if logger is None:
            logger = logging.getLogger(__name__)

        logger.debug(f"API Request to: {self._complete_url}")
        logger.debug(f"Model: {data.get('model', 'unknown')}")

        response = requests.post(self._complete_url, json=data, headers=self._headers)
        error = ""

        logger.debug(f"Response status code: {response.status_code}")

        if response.status_code == 200:
            try:
                resp = str(response.json()["choices"][0]["message"]["content"])
                think = str(
                    response.json()["choices"][0]["message"].get(
                        "reasoning_content", ""
                    )
                )
                logger.debug(
                    f"Successfully parsed API response. RESP: {resp} THINK: {think}"
                )
            except Exception as e:
                error = str(e)
                logger.error(
                    f"[bold red]{TIMES} Failed to parse API response:[/] {e}",
                    extra={"markup": True},
                )
                raise
            return True, Response(resp, think)
        else:
            logger.error(
                f"[bold red]{TIMES} API returned status {response.status_code}[/]\n"
                f"[dim]{response.text[:200]}[/]",
                extra={"markup": True},
            )
        return False, Response(error, "")

    def query(
        self,
        model: str,
        messages: list,
        timeout: int = 120,
        max_retries: int = 3,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        if logger is None:
            logger = logging.getLogger(__name__)

        logger.debug(f"Messages count: {len(messages)}")
        retries = 0
        while retries < max_retries:
            logger.debug(f"Attempt {retries + 1}/{max_retries}")
            status, response = self._query(
                {
                    "model": model,
                    "messages": messages,
                    "timeout": timeout,
                    **kwargs,
                },
                logger,
            )
            if status:
                logger.debug(f"Response length: {len(response.response)} chars")
                if retries > 0:
                    logger.info(
                        f"{CHECK} API success on attempt {retries + 1}",
                        extra={"markup": True},
                    )
                return response
            retries += 1
            logger.warning(
                f"{REFRESH} Retry {retries}/{max_retries}", extra={"markup": True}
            )

        logger.error(
            f"[bold red]{TIMES} Max retries exceeded. Failed to get response.[/]",
            extra={"markup": True},
        )
        raise RuntimeError("MAX RETRIES triggered. Failed to get response.")
