from subprocess import run
from pathlib import Path

for path in Path("pypython").rglob("*.py"):
    print(path)
    cmd = f"yapf -i {path} > /dev/null"
    run(cmd, shell=True)
    cmd = f"isort {path} > /dev/null"
    run(cmd, shell=True)

for path in Path("scripts").rglob("*.py"):
    print(path)
    cmd = f"yapf -i {path} > /dev/null"
    run(cmd, shell=True)
    cmd = f"isort {path} > /dev/null"
    run(cmd, shell=True)

