import ast
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ..icons import CHECK, MAGIC, ROCKET, THOUGHT, WARNING
from ..logger import get_logger
from .base import Action, Language
from .doubao_scalewob import (
    _NOTEPAD_SECTION,
    DoubaoScaleWoBAgent,
)

_CN_WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def _formatted_date(language: Language) -> str:
    now = datetime.now(UTC)
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
            enable_notepad=enable_notepad,
            enable_tracking=enable_tracking,
            tracking_output_dir=tracking_output_dir,
            max_steps=max_steps,
            use_vertex_api=use_vertex_api,
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
            prompt = (
                f"今天的日期是: {date_text}\n"
                "你是一个智能体分析专家，可以根据操作历史和当前状态图执行一系列操作来完成任务。\n"
                "你必须严格按照要求输出以下格式：\n"
                "<think>{think}</think>\n<answer>{action}</answer>\n\n"
                "其中：\n"
                "- {think} 是对你为什么选择这个操作的简短推理说明。\n"
                "- {action} 是本次执行的具体操作指令，必须严格遵循下方定义的指令格式。\n\n"
                "操作指令及其作用如下：\n"
                '- do(action="Tap", element=[x,y])  \n'
                "    Tap是点击操作，点击屏幕上的特定点。可用此操作点击按钮、选择项目、从主屏幕打开应用程序，或与任何可点击的用户界面元素进行交互。坐标系统从左上角 (0,0) 开始到右下角（999,999)结束。此操作完成后，您将自动收到结果状态的截图。\n"
                '- do(action="Type", text="xxx")  \n'
                "    Type是输入操作，在当前聚焦的输入框中输入文本。使用此操作前，请确保输入框已被聚焦（先点击它）。输入的文本将像使用键盘输入一样输入。重要提示：手机可能正在使用 ADB 键盘，该键盘不会像普通键盘那样占用屏幕空间。要确认键盘已激活，请查看屏幕底部是否显示 'ADB Keyboard {ON}' 类似的文本，或者检查输入框是否处于激活/高亮状态。不要仅仅依赖视觉上的键盘显示。自动清除文本：当你使用输入操作时，输入框中现有的任何文本（包括占位符文本和实际输入）都会在输入新文本前自动清除。你无需在输入前手动清除文本——直接使用输入操作输入所需文本即可。操作完成后，你将自动收到结果状态的截图。\n"
                '- do(action="Swipe", start=[x1,y1], end=[x2,y2])  \n'
                "    Swipe是滑动操作，通过从起始坐标拖动到结束坐标来执行滑动手势。可用于滚动内容、在屏幕之间导航、下拉通知栏以及项目栏或进行基于手势的导航。坐标系统从左上角 (0,0) 开始到右下角（999,999)结束。滑动持续时间会自动调整以实现自然的移动。此操作完成后，您将自动收到结果状态的截图。\n"
                '- do(action="Note", message="True")  \n'
                "    记录当前页面内容以便后续总结。\n"
                '- do(action="Long Press", element=[x,y])  \n'
                "    Long Pres是长按操作，在屏幕上的特定点长按指定时间。可用于触发上下文菜单、选择文本或激活长按交互。坐标系统从左上角 (0,0) 开始到右下角（999,999)结束。此操作完成后，您将自动收到结果状态的屏幕截图。\n"
                '- do(action="Double Tap", element=[x,y])  \n'
                "    Double Tap在屏幕上的特定点快速连续点按两次。使用此操作可以激活双击交互，如缩放、选择文本或打开项目。坐标系统从左上角 (0,0) 开始到右下角（999,999)结束。此操作完成后，您将自动收到结果状态的截图。\n"
                '- do(action="Wait", duration="x seconds")  \n'
                "    等待页面加载，x为需要等待多少秒。\n"
                '- finish(message="xxx")  \n'
                "    finish是结束任务的操作，表示准确完整完成任务，message是终止信息。"
            )
        else:
            prompt = (
                f"The current date: {date_text}\n"
                "# Setup\n"
                "You are a professional Android operation agent assistant that can fulfill the user's high-level instructions. Given a screenshot of the Android interface at each step, you first analyze the situation, then plan the best course of action using Python-style pseudo-code.\n\n"
                "# More details about the code\n"
                "Your response format must be structured as follows:\n\n"
                "Think first: Use <think>...</think> to analyze the current screen, identify key elements, and determine the most efficient action.\n"
                "Provide the action: Use <answer>...</answer> to return a single line of pseudo-code representing the operation.\n\n"
                "Your output should STRICTLY follow the format:\n"
                "<think>\n"
                "[Your thought]\n"
                "</think>\n"
                "<answer>\n"
                "[Your operation code]\n"
                "</answer>\n\n"
                "- **Tap**\n"
                "  Perform a tap action on a specified screen area. The element is a list of 2 integers, representing the coordinates of the tap point.\n"
                "  **Example**:\n"
                "  <answer>\n"
                '  do(action="Tap", element=[x,y])\n'
                "  </answer>\n"
                "- **Type**\n"
                "  Enter text into the currently focused input field.\n"
                "  **Example**:\n"
                "  <answer>\n"
                '  do(action="Type", text="Hello World")\n'
                "  </answer>\n"
                "- **Swipe**\n"
                "  Perform a swipe action with start point and end point.\n"
                "  **Examples**:\n"
                "  <answer>\n"
                '  do(action="Swipe", start=[x1,y1], end=[x2,y2])\n'
                "  </answer>\n"
                "- **Long Press**\n"
                "  Perform a long press action on a specified screen area.\n"
                "  You can add the element to the action to specify the long press area. The element is a list of 2 integers, representing the coordinates of the long press point.\n"
                "  **Example**:\n"
                "  <answer>\n"
                '  do(action="Long Press", element=[x,y])\n'
                "  </answer>\n"
                "- **Finish**\n"
                "  Terminate the program and optionally print a message.\n"
                "  **Example**:\n"
                "  <answer>\n"
                '  finish(message="Task completed.")\n'
                "  </answer>"
            )

        if self.enable_notepad:
            prompt += _NOTEPAD_SECTION.format(notepad_section=self._format_notepad())

        return prompt

    def _parse_action(self, response_text: str) -> Action:
        thought = ""
        try:
            thought, answer = self._split_thinking_and_action(response_text)
            if thought:
                self.logger.info(
                    f"\n[bold cyan]{THOUGHT} Thought[/]\n[italic dim]{thought}[/]\n",
                    extra={"markup": True},
                )
            action = self._parse_answer_action(answer, thought)
            self._consecutive_parse_failures = 0
        except Exception as e:  # noqa: BLE001
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
        else:
            return action

    def _split_thinking_and_action(self, response_text: str) -> tuple[str, str]:
        content = response_text.strip()
        thinking = ""
        action = ""

        if "finish(message=" in content:
            parts = content.split("finish(message=", 1)
            thinking = parts[0].strip()
            action = "finish(message=" + parts[1]
        elif "do(action=" in content:
            parts = content.split("do(action=", 1)
            thinking = parts[0].strip()
            action = "do(action=" + parts[1]
        elif "<answer>" in content:
            parts = content.split("<answer>", 1)
            thinking = parts[0].replace("<think>", "").replace("</think>", "").strip()
            action = parts[1].replace("</answer>", "").strip()

        if not action:
            raise ValueError("Missing action payload")

        if thinking.startswith("<think>") and thinking.endswith("</think>"):
            thinking = thinking[7:-8].strip()

        return thinking, action

    def predict(self) -> Action:
        if not self.history:
            prompt = self._build_prompt()
            self.history.append({"role": "system", "content": prompt})
            self.logger.info(
                f"[bold magenta]{ROCKET} Task Started[/]\n"
                f"[italic dim]{self.task_name}[/]\n",
                extra={"markup": True},
            )

        forced_action = self._check_step_limit()
        if forced_action:
            self.logger.warning(
                f"[bold yellow]{WARNING} Maximum steps reached ({self.max_steps})[/]\n"
                f"[dim]Forcing task completion[/]\n",
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

        self.logger.debug("Capturing screenshot...")
        screenshot = self.device.capture_screenshot("B64")
        self.logger.debug(f"Screenshot captured: {len(screenshot)} chars (base64)")
        screenshot_for_tracker = screenshot

        has_user_turn = any(msg["role"] == "user" for msg in self.history)
        screen_info = json.dumps({"current_app": "ScaleWoB"}, ensure_ascii=False)
        if has_user_turn:
            text_content = f"** Screen Info **\n\n{screen_info}"
        else:
            text_content = f"{self.task_name}\n\n{screen_info}"
            if self.task_params:
                text_content += f"\n\n{self.task_params}"

        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{screenshot}"},
                },
                {"type": "text", "text": text_content},
            ],
        }

        self.logger.debug(
            f"[bold blue]{MAGIC} Requesting LLM prediction[/] ({self.model_name})",
            extra={"markup": True},
        )

        start_time = time.time()
        response = self.client.query(
            self.model_name,
            self.history + [user_message],
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

        if self.tracker:
            self._record_action(action, screenshot_for_tracker, duration)
            if action["action_type"] == "finished":
                self.task_status = status
                result = self._finalize_tracking(result=status)
                if result:
                    trace_path, json_path = result
                    self.logger.debug(f"Generated {trace_path} and {json_path}")

        user_text_only = {
            "role": "user",
            "content": [{"type": "text", "text": text_content}],
        }
        try:
            thought, answer = self._split_thinking_and_action(response.response)
        except ValueError:
            thought = action.get("thought", "")
            answer = 'do(action="Wait", duration="1 seconds")'
        assistant_message = {
            "role": "assistant",
            "content": f"<think>{thought}</think><answer>{answer}</answer>",
        }

        self.history.append(user_text_only)
        self.history.append(assistant_message)

        return action

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
            raise TypeError("do() is not a valid function call")

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
