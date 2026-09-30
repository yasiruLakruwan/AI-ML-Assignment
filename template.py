import os
from pathlib import Path
dir_lists=[
    "src/",
    Path("src/__init__.py"),
    Path("src/config.py"),
    Path("src/models.py"),\
    Path(".env"),
    Path("src/video.py"),
    Path("src/vision.py"),
    Path("src/bed.py"),
    Path("src/state_classifier.py"),
    Path("src/state_machine.py"),
    Path("src/agent.py"),
    Path("src/agent.py"),
    Path("src/events.py"),
    Path("src/repost.py"),
    Path("src/main.py"),
    Path("src/logger.py"),
    Path("src/custom_exeption.py"),
    Path("README.md")
]

for dir in dir_lists:
    dir_name,file_path = os.path.split(dir)
    print(f"dir name :{dir_name}")
    print(f"file path : {file_path}")

    if dir_name:
        os.makedirs(dir_name,exist_ok=True)

    if file_path:
        file = os.path.join(dir_name,file_path) 
        open(file,'a').close()