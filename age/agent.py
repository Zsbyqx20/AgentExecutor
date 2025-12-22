import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal, TypedDict

from .api import APIClient
from .device import Device


class Action(TypedDict):
    action_type: str
    params: dict
    thought: str


class Language(str, Enum):
    CN = "Chinese"
    EN = "English"


class Agent(ABC):
    def __init__(self, device_id: str | None = None) -> None:
        super().__init__()
        self.device = Device(device_id)

    @abstractmethod
    def predict(self) -> Action:
        pass

    @abstractmethod
    def reset(self, **kwargs) -> None:
        pass


DOUBAO_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```

## Action Space
click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
press_home()
press_back()
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""


class DoubaoAgent(Agent):
    def __init__(
        self,
        model_name: str,
        task_name: str,
        language: Literal["CN", "EN"] = "CN",
        device_id: str | None = None,
    ) -> None:
        super().__init__(device_id)
        self.name = "Doubao"
        self.model_name = model_name
        self.task_name = task_name
        self.prompt_template = DOUBAO_PROMPT
        self.window_size = self.device.size
        self.language = Language[language.upper()]
        self.client = APIClient()
        self.history = []

    def reset(self, **kwargs) -> None:
        self.task_name = kwargs.get("task_name", "")
        self.history = []

    def _normalize_coordinates(self, x: int, y: int) -> tuple[int, int]:
        width, height = self.window_size
        return int(x * width / 1000), int(y * height / 1000)

    def _parse_action(self, response_text: str) -> Action:
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\nAction:|\n\n|\Z)", response_text, re.DOTALL
        )
        action_match = re.search(
            r"Action:\s*(.*?)(?=\n\n|\Z)", response_text, re.DOTALL
        )

        thought = thought_match.group(1).strip() if thought_match else ""
        action_text = action_match.group(1).strip() if action_match else ""

        action_type = ""
        params = {}

        if action_text:
            if action_text.startswith("click("):
                action_type = "click"
                point_match = re.search(r"<point>(\d+)\s+(\d+)</point>", action_text)
                if point_match:
                    params["point"] = (
                        int(point_match.group(1)),
                        int(point_match.group(2)),
                    )
            elif action_text.startswith("long_press("):
                action_type = "long_press"
                point_match = re.search(r"<point>(\d+)\s+(\d+)</point>", action_text)
                if point_match:
                    params["point"] = (
                        int(point_match.group(1)),
                        int(point_match.group(2)),
                    )
            elif action_text.startswith("type("):
                action_type = "type"
                content_match = re.search(r"content='([^']*)'", action_text)
                if content_match:
                    params["content"] = content_match.group(1)
            elif action_text.startswith("scroll("):
                action_type = "scroll"
                point_match = re.search(r"<point>(\d+)\s+(\d+)</point>", action_text)
                direction_match = re.search(r"direction='([^']*)'", action_text)
                if point_match:
                    params["point"] = (
                        int(point_match.group(1)),
                        int(point_match.group(2)),
                    )
                if direction_match:
                    params["direction"] = direction_match.group(1)
            elif action_text.startswith("open_app("):
                action_type = "open_app"
                app_match = re.search(r"app_name='([^']*)'", action_text)
                if app_match:
                    params["app_name"] = app_match.group(1)
            elif action_text.startswith("drag("):
                action_type = "drag"
                start_match = re.search(
                    r"start_point='<point>(\d+)\s+(\d+)</point>", action_text
                )
                end_match = re.search(
                    r"end_point='<point>(\d+)\s+(\d+)</point>", action_text
                )
                if start_match:
                    params["start_point"] = (
                        int(start_match.group(1)),
                        int(start_match.group(2)),
                    )
                if end_match:
                    params["end_point"] = (
                        int(end_match.group(1)),
                        int(end_match.group(2)),
                    )
            elif action_text == "press_home()":
                action_type = "press_home"
            elif action_text == "press_back()":
                action_type = "press_back"
            elif action_text.startswith("finished("):
                action_type = "finished"
                content_match = re.search(r"content='([^']*)'", action_text)
                if content_match:
                    params["content"] = content_match.group(1)

        return Action(action_type=action_type, params=params, thought=thought)

    def _execute(self, action: Action) -> None:
        from .device import Direction

        action_type = action["action_type"]
        params = action["params"]

        if action_type == "click":
            x, y = self._normalize_coordinates(*params["point"])
            self.device.press(x, y)
        elif action_type == "long_press":
            x, y = self._normalize_coordinates(*params["point"])
            self.device.long_press(x, y, duration=1.0)
        elif action_type == "type":
            self.device.type_text(params["content"])
        elif action_type == "scroll":
            x, y = self._normalize_coordinates(*params["point"])
            direction = params["direction"]
            distance = self.window_size.height // 2
            self.device.scroll(x, y, Direction[direction.upper()], distance)
        elif action_type == "open_app":
            self.device.open_app(params["app_name"])
        elif action_type == "drag":
            start_x, start_y = self._normalize_coordinates(*params["start_point"])
            end_x, end_y = self._normalize_coordinates(*params["end_point"])
            self.device.drag(start_x, start_y, end_x, end_y, duration=1.0)
        elif action_type == "press_home":
            self.device.navigate_home()
        elif action_type == "press_back":
            self.device.navigate_back()
        elif action_type == "finished":
            print(f"Task completed: {params['content']}")
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    def _compress_history(self):
        if not self.history:
            return []
        history = []
        cnt = 0
        screens_num = len(self.history) // 2
        screen_start_idx = max(10000000000, screens_num - 5)
        for h in self.history:
            if h["role"] == "user":
                hc = []
                for c in h["content"]:
                    if c["type"] == "image_url":
                        if cnt < screen_start_idx:
                            continue
                        else:
                            cnt += 1
                    hc.append(c)
                history.append({"role": "user", "content": hc})
            else:
                history.append(h)
        return history

    def predict(self) -> Action:
        if not self.history:
            prompt = DOUBAO_PROMPT.format(
                language=self.language.value, instruction=self.task_name
            )
        else:
            prompt = "Please continue"
        screenshot = self.device.capture_screenshot("B64")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot}"},
                    },
                ],
            }
        ]
        response = self.client.query(
            self.model_name, self._compress_history() + messages
        )
        action = self._parse_action(response.response)
        self._execute(action)
        self.history.extend(messages)
        self.history.append({"role": "assistant", "content": response.response})
        return action
