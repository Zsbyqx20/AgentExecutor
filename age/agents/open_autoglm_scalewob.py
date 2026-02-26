import ast
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ..icons import THOUGHT, WARNING
from ..logger import get_logger
from .base import Action, Language
from .doubao_scalewob import (
    _NOTEPAD_SECTION,
    _USER_INSTRUCTION,
    DoubaoScaleWoBAgent,
)

_CN_WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def _formatted_date(language: Language) -> str:
    now = datetime.today()
    if language == Language.CN:
        return now.strftime("%Y年%m月%d日") + f" {_CN_WEEKDAYS[now.weekday()]}"
    return now.strftime("%Y-%m-%d, %A")


class OpenAutoGLMScaleWoBAgent(DoubaoScaleWoBAgent):
    """ScaleWoB agent using Open-AutoGLM style prompts and output format."""

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
        enable_notepad: bool = False,
        enable_tracking: bool = False,
        tracking_output_dir: str | Path | None = None,
        max_steps: int | None = None,
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
            enable_notepad=enable_notepad,
            enable_tracking=enable_tracking,
            tracking_output_dir=tracking_output_dir,
            max_steps=max_steps,
        )
        self._consecutive_parse_failures = 0
        self._max_parse_failures = 2
        self.logger = get_logger(__name__)

    def reset(self, **kwargs) -> None:
        super().reset(**kwargs)
        self._consecutive_parse_failures = 0

    def _build_prompt(self) -> str:
        date_text = _formatted_date(self.language)
        if self.language == Language.CN:
            action_space = """
- do(action="Tap", element=[x,y])
- do(action="Long Press", element=[x,y])
- do(action="Type", text="xxx")
- do(action="Type_Name", text="xxx")
- do(action="Swipe", start=[x1,y1], end=[x2,y2])
- do(action="Wait", duration="x seconds")
- finish(message="xxx")
"""
            prompt = (
                f"今天的日期是: {date_text}\n"
                "你是一个智能体分析专家，可以根据操作历史和当前状态图执行一系列操作来完成任务。\n"
                "你必须严格按照要求输出以下格式：\n"
                "<think>{think}</think>\n<answer>{action}</answer>\n\n"
                "其中 {action} 必须是单行可执行操作。\n\n"
                "支持的操作如下：\n"
                f"{action_space}\n"
                "规则：\n"
                "1. 坐标范围是 0-999。\n"
                "2. 每次只能输出一条操作。\n"
                "3. 页面加载慢时优先使用 Wait。\n"
                "4. 做完任务后使用 finish(message=...)。\n"
            )
        else:
            action_space = """
- do(action="Tap", element=[x,y])
- do(action="Long Press", element=[x,y])
- do(action="Type", text="xxx")
- do(action="Type_Name", text="xxx")
- do(action="Swipe", start=[x1,y1], end=[x2,y2])
- do(action="Wait", duration="x seconds")
- finish(message="xxx")
"""
            prompt = (
                f"The current date: {date_text}\n"
                "You are a professional UI operation agent. Given task context and screenshot history, decide the next best action.\n"
                "Your output must strictly follow:\n"
                "<think>{think}</think>\n<answer>{action}</answer>\n\n"
                "{action} must be a single-line executable call.\n\n"
                "Supported actions:\n"
                f"{action_space}\n"
                "Rules:\n"
                "1. Coordinates are normalized in range 0-999.\n"
                "2. Return exactly one action each turn.\n"
                "3. Use Wait when content is still loading.\n"
                "4. Use finish(message=...) only after task completion.\n"
            )

        if self.enable_notepad:
            prompt += _NOTEPAD_SECTION.format(notepad_section=self._format_notepad())

        prompt += _USER_INSTRUCTION.format(instruction=self.task_name)

        if self.task_params:
            prompt += f"""
## Answer Format
### Hint: This task requires JSON content in finish(message=...).
{self.task_params}
"""
        return prompt

    def _parse_action(self, response_text: str) -> Action:
        think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)

        thought = think_match.group(1).strip() if think_match else ""
        if thought:
            self.logger.info(
                "\n[bold cyan]{} Thought[/]\n[italic dim]{}[/]\n".format(THOUGHT, thought),
                extra={"markup": True},
            )

        try:
            if not answer_match:
                raise ValueError("Missing <answer>...</answer> section")

            answer = answer_match.group(1).strip()
            action = self._parse_answer_action(answer, thought)
            self._consecutive_parse_failures = 0
            return action
        except Exception as e:
            self._consecutive_parse_failures += 1
            self.logger.warning(
                f"[bold yellow]{WARNING} Failed to parse model action[/] "
                f"(#{self._consecutive_parse_failures}): {e}",
                extra={"markup": True},
            )
            if self._consecutive_parse_failures > self._max_parse_failures:
                msg = "Parser failure threshold exceeded"
                return Action(
                    action_type="finished",
                    params={"content": msg},
                    thought=thought or msg,
                )
            return Action(
                action_type="wait",
                params={"seconds": 1},
                thought=thought or "Parsing failed, waiting briefly before retry.",
            )

    def _parse_answer_action(self, answer: str, thought: str) -> Action:
        answer = answer.strip()

        if answer.startswith("finish"):
            tree = ast.parse(answer, mode="eval")
            if not isinstance(tree.body, ast.Call):
                raise ValueError("finish() is not a valid function call")
            call = tree.body
            message = ""
            for keyword in call.keywords:
                if keyword.arg == "message":
                    message = str(ast.literal_eval(keyword.value))
                    break
            return Action(
                action_type="finished",
                params={"content": message},
                thought=thought,
            )

        if not answer.startswith("do"):
            raise ValueError(f"Unsupported answer expression: {answer}")

        answer = answer.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        tree = ast.parse(answer, mode="eval")
        if not isinstance(tree.body, ast.Call):
            raise ValueError("do() is not a valid function call")

        call = tree.body
        kwargs: dict[str, Any] = {}
        for keyword in call.keywords:
            if keyword.arg:
                kwargs[keyword.arg] = ast.literal_eval(keyword.value)

        action_name = kwargs.get("action")
        if action_name == "Tap":
            element = kwargs.get("element")
            if not isinstance(element, list) or len(element) != 2:
                raise ValueError("Tap requires element=[x,y]")
            return Action(
                action_type="click",
                params={"point": (int(element[0]), int(element[1]))},
                thought=thought,
            )
        if action_name == "Long Press":
            element = kwargs.get("element")
            if not isinstance(element, list) or len(element) != 2:
                raise ValueError("Long Press requires element=[x,y]")
            return Action(
                action_type="long_press",
                params={"point": (int(element[0]), int(element[1]))},
                thought=thought,
            )
        if action_name in {"Type", "Type_Name"}:
            return Action(
                action_type="type",
                params={"content": str(kwargs.get("text", ""))},
                thought=thought,
            )
        if action_name == "Swipe":
            start = kwargs.get("start")
            end = kwargs.get("end")
            if not isinstance(start, list) or len(start) != 2:
                raise ValueError("Swipe requires start=[x1,y1]")
            if not isinstance(end, list) or len(end) != 2:
                raise ValueError("Swipe requires end=[x2,y2]")
            return Action(
                action_type="drag",
                params={
                    "start_point": (int(start[0]), int(start[1])),
                    "end_point": (int(end[0]), int(end[1])),
                },
                thought=thought,
            )
        if action_name == "Wait":
            duration = str(kwargs.get("duration", "1 seconds"))
            match = re.search(r"(\d+(?:\.\d+)?)", duration)
            seconds = int(float(match.group(1))) if match else 1
            return Action(
                action_type="wait",
                params={"seconds": max(1, seconds)},
                thought=thought,
            )

        raise ValueError(f"Unsupported action: {action_name}")
