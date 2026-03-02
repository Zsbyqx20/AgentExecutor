import logging
import os
from typing import NamedTuple

import requests
from dotenv import load_dotenv
from litellm import completion

from .icons import CHECK, REFRESH, TIMES

load_dotenv()


class Response(NamedTuple):
    response: str
    thought: str


class _SuppressGeminiBlankTextWarning(logging.Filter):
    """Filter noisy LiteLLM warning about auto-injected blank text for Gemini."""

    _needle = (
        "No text in user content. Adding a blank text to user content, "
        "to ensure Gemini doesn't fail the request."
    )

    def filter(self, record: logging.LogRecord) -> bool:
        return self._needle not in record.getMessage()


class APIClient:
    def __init__(self, prefix: str = "", use_vertex_api: bool = False) -> None:
        self._use_vertex_api = use_vertex_api

        if self._use_vertex_api:
            self._configure_vertex_logging()
            self._vertex_location = os.environ.get("VERTEX_LOCATION", "global")
            required_env = ("GOOGLE_APPLICATION_CREDENTIALS", "VERTEX_PROJECT")
            missing = [key for key in required_env if not os.environ.get(key)]
            if missing:
                missing_text = ", ".join(missing)
                raise ValueError(
                    "Missing required Vertex AI environment variables: "
                    f"{missing_text}"
                )
            return

        base_url_name = prefix.upper() + "_BASE_URL" if prefix else "BASE_URL"
        api_key_name = prefix.upper() + "_API_KEY" if prefix else "API_KEY"
        self._complete_url = os.environ.get(
            base_url_name, "https://api.openai.com/v1/chat/completions"
        )
        assert os.environ.get(api_key_name), (
            "You must provide an API key in your environment variables!"
        )
        self._api_key = os.environ.get(api_key_name, "")
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _configure_vertex_logging(self) -> None:
        # LiteLLM installs its own DEBUG stream handler; clamp noisy third-party
        # logger levels when Vertex mode is enabled.
        noisy_loggers = ("LiteLLM", "litellm", "httpx", "httpcore", "openai", "urllib3")
        for logger_name in noisy_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)

        # Suppress a known benign LiteLLM Gemini warning spam while preserving
        # other warnings/errors.
        litellm_logger = logging.getLogger("LiteLLM")
        if not any(
            isinstance(existing_filter, _SuppressGeminiBlankTextWarning)
            for existing_filter in litellm_logger.filters
        ):
            litellm_logger.addFilter(_SuppressGeminiBlankTextWarning())

    def _extract_message_text(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        text_parts.append(text_value)
            return "\n".join(text_parts)
        return str(content)

    def _query(
        self, data: dict, logger: logging.Logger | None = None
    ) -> tuple[bool, Response]:
        if logger is None:
            logger = logging.getLogger(__name__)

        if self._use_vertex_api:
            return self._query_vertex(data, logger)
        return self._query_http(data, logger)

    def _query_http(
        self, data: dict, logger: logging.Logger
    ) -> tuple[bool, Response]:
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

    def _query_vertex(
        self, data: dict[str, object], logger: logging.Logger
    ) -> tuple[bool, Response]:
        model = str(data.get("model", ""))
        logger.debug("API Request to: Vertex AI via LiteLLM")
        logger.debug(f"Model: {model}")

        messages = data.get("messages")
        if not isinstance(messages, list):
            raise TypeError(f"Expected list for messages, got {type(messages).__name__}")

        kwargs = dict(data)
        kwargs.pop("model", None)
        kwargs.pop("messages", None)
        # Some OpenAI-compatible params are not accepted by Gemini Vertex requests.
        kwargs.pop("thinking", None)

        try:
            response = completion(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                vertex_location=self._vertex_location,
                **kwargs,
            )

            choices = getattr(response, "choices", None)
            if not choices:
                raise ValueError("Vertex response does not include choices")
            message = choices[0].message
            content = self._extract_message_text(getattr(message, "content", ""))
            think = str(getattr(message, "reasoning_content", ""))

            logger.debug(
                f"Successfully parsed Vertex response. RESP: {content} THINK: {think}"
            )
            return True, Response(content, think)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"[bold red]{TIMES} Vertex request failed:[/] {exc}",
                extra={"markup": True},
            )
            return False, Response(str(exc), "")

    def query(
        self,
        model: str,
        messages: list,
        timeout: int = 600,
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
