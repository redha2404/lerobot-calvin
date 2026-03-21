#!/bin/bash

# Wheel and specific CMake are baseline
pip install wheel cmake==3.18.4

# Install exact frozen dependencies bridging Calvin and LeRobot flawlessly
pip install -r requirements.txt

# Force-install LeRobot natively leveraging Python 3.10 architecture
pip install "lerobot @ git+https://github.com/huggingface/lerobot.git@46e19ae579f80ce66211afafd1c3c649c569131f"

# Install Calvin core modules securely without triggering pip dependency conflicts
cd calvin_env/tacto
pip install --no-deps -e .
cd ..
pip install --no-deps -e .
cd ../calvin_models
pip install --no-deps -e .

