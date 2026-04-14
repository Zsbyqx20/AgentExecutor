from age import BrownieAgent

task = "打开网易云音乐，搜索许嵩的最佳歌手，打开播放界面并暂停播放，然后拖动进度条至1分15秒"

# copy .env.example to .env and configure the base url, api key, etc.
# the execution progress will be saved under tracks/ folder
# the logs will be saved under logs/ folder
with BrownieAgent(
    model_name="BrownieGUI-8B",
    task_name=task,
    language="EN",
    enable_tracking=True,
) as agent:
    while True:
        action = agent.predict()
        if action["action_type"] == "finished":
            break
