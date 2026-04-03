import ast
import json
import re
import time
from pathlib import Path
from typing import Any, Literal

from ..api import APIClient
from ..icons import CHECK, MAGIC, ROCKET, THOUGHT, WARNING
from ..logger import get_logger
from .base import Action
from .doubao_scalewob import DoubaoScaleWoBAgent
from .rgm import _SYS_PROMPT, _USER_PROMPT


class RGMScaleWoBAgent(DoubaoScaleWoBAgent):
    """ScaleWoB variant of RGM that keeps the original prompt format."""

    def __init__(
        self,
        model_name: str,
        task_name: str,
        env_id: str,
        task_id: int,
        task_params: dict[str, Any] | None = None,
        language: Literal["CN", "EN"] = "CN",
        headless: bool = False,
        platform: Literal["mobile", "desktop"] = "desktop",
        enable_tracking: bool = False,
        tracking_output_dir: str | Path | None = None,
        max_steps: int | None = None,
        use_vertex_api: bool = False,
    ) -> None:
        super().__init__(
            model_name=model_name,
            task_name=task_name,
            env_id=env_id,
            task_id=task_id,
            task_params=task_params,
            language=language,
            headless=headless,
            platform=platform,
            enable_tracking=enable_tracking,
            tracking_output_dir=tracking_output_dir,
            max_steps=max_steps,
            use_vertex_api=use_vertex_api,
        )
        self.name = "RGM"
        self.client = APIClient(prefix=self.name, use_vertex_api=use_vertex_api)
        self.history = []
        self.action_history: list[str] = []
        self.logger = get_logger(__name__)

    def reset(self, **kwargs) -> None:
        super().reset(**kwargs)
        self.action_history = []

    def _format_action_history(self) -> str:
        if not self.action_history:
            return "Empty. You are about to perform the first action."
        return "\n".join(
            f"Step {idx}. {action}" for idx, action in enumerate(self.action_history, 1)
        )

    def _build_messages(self, screenshot_b64: str) -> list[dict[str, Any]]:
        instruction = self.task_name
        if self.task_params:
            instruction += f"\n\n# Hint: This task requires you to provide your answer with params. The JSON schema required is as follows.\n{self.task_params}"
        user_prompt = _USER_PROMPT.format(
            instruction=self.task_name,
            action_history=self._format_action_history(),
        )
        return [
            {"role": "system", "content": _SYS_PROMPT.strip()},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

    def _parse_action(self, response_text: str) -> Action:
        try:
            json_match = re.search(
                r'\{"name"\s*:\s*"mobile_use"\s*,\s*"arguments"\s*:\s*\{.*?\}\}',
                response_text,
                re.DOTALL,
            )
            if not json_match:
                raise ValueError("Cannot find mobile_use tool call")

            raw_call = json_match.group(0)
            try:
                function_call = json.loads(raw_call)
            except json.JSONDecodeError:
                function_call = ast.literal_eval(raw_call)
            if function_call.get("name") != "mobile_use":
                raise ValueError(f"Unknown function: {function_call.get('name')}")

            arguments = function_call.get("arguments", {})
            action_type = str(arguments.get("action", "")).strip()

            thought_match = re.search(
                r"Thought:\s*(.*?)(?=\nAction:|\n\n|\Z)", response_text, re.DOTALL
            )
            if thought_match:
                thought = thought_match.group(1).strip()
            else:
                thought = f"Performing {action_type} action"

            action_desc_match = re.search(r"Action:\s*(.*)", response_text)
            action_description = (
                action_desc_match.group(1).strip() if action_desc_match else None
            )

            if thought:
                self.logger.info(
                    f"\n[bold cyan]{THOUGHT} Thought[/]\n[italic dim]{thought}[/]\n",
                    extra={"markup": True},
                )

            if action_type == "click":
                action = Action(
                    action_type="click",
                    params={"point": tuple(arguments.get("coordinate", [0, 0]))},
                    thought=thought,
                )
            elif action_type == "long_press":
                action = Action(
                    action_type="long_press",
                    params={"point": tuple(arguments.get("coordinate", [0, 0]))},
                    thought=thought,
                )
            elif action_type == "swipe":
                action = Action(
                    action_type="drag",
                    params={
                        "start_point": tuple(arguments.get("coordinate", [0, 0])),
                        "end_point": tuple(arguments.get("coordinate2", [0, 0])),
                    },
                    thought=thought,
                )
            elif action_type == "type":
                action = Action(
                    action_type="type",
                    params={"content": arguments.get("text", "")},
                    thought=thought,
                )
            elif action_type == "wait":
                action = Action(
                    action_type="wait",
                    params={"seconds": int(arguments.get("time", 1))},
                    thought=thought,
                )
            elif action_type == "terminate":
                action = Action(
                    action_type="finished",
                    params={
                        "content": str(
                            arguments.get("text", arguments.get("status", "success"))
                        )
                    },
                    thought=thought,
                )
            elif action_type == "open":
                action = Action(
                    action_type="wait",
                    params={"seconds": 1},
                    thought=thought
                    or "ScaleWoB has no app launcher semantics; waiting briefly.",
                )
            elif action_type == "key":
                key = str(arguments.get("text", "")).strip().lower()
                if key in {"enter", "return", "keycode_enter"}:
                    action = Action(
                        action_type="press_enter",
                        params={},
                        thought=thought,
                    )
                else:
                    action = Action(
                        action_type="wait",
                        params={"seconds": 1},
                        thought=thought
                        or f"Key '{key}' is unsupported in ScaleWoB; waiting briefly.",
                    )
            else:
                action = Action(
                    action_type="wait",
                    params={"seconds": 1},
                    thought=thought or "Unknown action, waiting briefly before retry.",
                )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                f"[bold yellow]{WARNING} Failed to parse RGM ScaleWoB action[/]: {exc}",
                extra={"markup": True},
            )
            return Action(
                action_type="wait",
                params={"seconds": 1},
                thought="Parsing failed, waiting briefly before retry.",
            )

        if action_description:
            action["params"]["description"] = action_description
        return action

    def predict(self) -> Action:
        if not self.history:
            self.logger.info(
                f"[bold magenta]{ROCKET} Task Started[/]\n"
                f"[italic dim]{self.task_name}[/]\n",
                extra={"markup": True},
            )

        forced_action = self._check_step_limit()
        if forced_action:
            self.logger.warning(
                f"[bold yellow]{WARNING} Maximum steps reached ({self.max_steps})[/]\n"
                "[dim]Forcing task completion[/]\n",
                extra={"markup": True},
            )
            duration, status = self.execute(forced_action)
            if self.tracker:
                screenshot = self.device.capture_screenshot("B64")
                self._record_action(forced_action, screenshot, duration)
                result = self._finalize_tracking(result=status)
                if result:
                    trace_path, json_path = result
                    self.logger.debug(f"Generated {trace_path} and {json_path}")
                self.task_status = False
            return forced_action

        screenshot = self.device.capture_screenshot("B64")
        screenshot_for_tracker = screenshot
        messages = self._build_messages(screenshot)

        self.logger.debug(
            f"[bold blue]{MAGIC} Requesting LLM prediction[/] ({self.model_name})",
            extra={"markup": True},
        )

        start_time = time.time()
        response = self.client.query(
            self.model_name,
            messages,
            logger=self.logger,
            thinking={"type": "disabled"},
        )
        elapsed = time.time() - start_time

        self.logger.info(
            f"[dim]{CHECK} Response received in {elapsed:.2f}s[/]",
            extra={"markup": True},
        )
        self.logger.debug(f"Response length: {len(response.response)} chars")

        action = self._parse_action(response.response)
        duration, status = self.execute(action)

        action_desc = action["params"].get("description")
        if action_desc:
            self.action_history.append(str(action_desc))
        else:
            self.action_history.append(f"{action['action_type']}: {action['params']}")

        if self.tracker:
            self._record_action(action, screenshot_for_tracker, duration)
            if action["action_type"] == "finished":
                self.task_status = status
                result = self._finalize_tracking(result=status)
                if result:
                    trace_path, json_path = result
                    self.logger.debug(f"Generated {trace_path} and {json_path}")

        self.history.append({"role": "assistant", "content": response.response})
        return action


__all__ = ["RGMScaleWoBAgent"]
