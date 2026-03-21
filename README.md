# CALVIN via LeRobot

[<b>CALVIN - A benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks</b>](https://arxiv.org/pdf/2112.03227.pdf)

We present **CALVIN** (**C**omposing **A**ctions from **L**anguage and **Vi**sio**n**), an open-source simulated benchmark to learn long-horizon language-conditioned tasks.
This optimized fork perfectly integrates the CALVIN simulator with the [Hugging Face LeRobot](https://github.com/huggingface/lerobot) framework for state-of-the-art native **VLA** (Vision-Language-Action) policy training, deployment, and evaluation!

![](media/teaser.png)

## :computer:  Setup & Installation

To begin, clone this repository locally
```bash
git clone --recurse-submodules https://github.com/redha2404/lerobot-calvin.git
export CALVIN_ROOT=$(pwd)/calvin
```

Install requirements:
```bash
cd $CALVIN_ROOT
conda create -n calvin_venv python=3.10 # or use virtualenv
conda activate calvin_venv
sh install.sh
```
*(If you encounter problems installing pyhash, you might have to downgrade setuptools to a version below 58.)*

Download dataset (choose which split you want to download with the argument `D`, `ABC` or `ABCD`): \
If you want to get started without downloading the whole dataset, use the argument `debug` to download a small debug dataset (1.3 GB).
```bash
cd $CALVIN_ROOT/dataset
sh download_data.sh D
```

---

## 🤖 LeRobot Benchmark Integration Walkthrough

This repository contains a complete, decoupled suite of tools originally compiled into the `scripts/` directory to natively bend the CALVIN dataset to the LeRobot framework constraints.

### 1. Inspect Dataset
Use `inspect_dataset.py` to view task frequencies and filter available tasks (based on the `lang_annotations/auto_lang_ann.npy` file) in your raw downloaded CALVIN dataset.
```bash
python scripts/inspect_dataset.py --dataset_path /path/to/calvin/dataset/task_D_D --top_k 50 --min_train 15 --min_val 5
```

### 2. Convert Dataset
Convert your specifically selected CALVIN tasks into the unified LeRobot format (Parquet + MP4 videos):
```bash
python scripts/convert_lerobot.py \
    --dataset_path /path/to/calvin/dataset/task_D_D \
    --out_dir /path/to/calvin_task_D_D_pick_place \
    --tasks "pick up the pink block" "pick up the blue block" "put the grasped block on top of a block"
```

### 3. Parquet Dataset Relabeling
Dynamically remap and rename task texts within a processed LeRobot dataset without any PyBullet rendering overhead.
```bash
python scripts/relabel_lerobot_dataset.py \
    --dataset_path /path/to/calvin_task_D_D_pick_place \
    --interactive
```

### 4. Train Model
Train a SmolVLA policy completely locally or originating from the Hugging Face Hub using the converted Parquet.
```bash
python scripts/train_lerobot.py \
    --dataset_path /path/to/calvin_task_D_D_pick_place \
    --output_dir outputs/train/smolvla_model \
    --batch_size 8 --steps 120000 --lr 1e-4
```
*(You can also pass a Hugging Face Hub dataset ID directly to `--dataset_path`)*

### 5. Evaluate Policy (Multi-Step Sequencing)
Evaluate trained checkpoints natively in PyBullet using dynamically configured task matrices to test LH-MTLC long-horizon chaining. 
```bash
python calvin_models/calvin_agent/evaluation/evaluate_policy.py \
    --dataset_path /path/to/calvin/dataset/task_D_D \
    --train_folder outputs/train/smolvla_model \
    --eval_matrix '[["lift_pink_block_table", "stack_block", 5], ["lift_blue_block_table", "stack_block", 5]]' \
    --ep_len 90 \
    --num_sequences 5000
```

### 6. Single-Step Isolation Evaluation
Strictly evaluate your model on single-subtask performance (e.g. debugging precise gripping) without full-sequence chaining error accumulation.
```bash
python calvin_models/calvin_agent/evaluation/evaluate_policy_singlestep.py \
    --dataset_path /path/to/calvin/dataset/task_D_D \
    --train_folder outputs/train/smolvla_model \
    --target_tasks '["lift_pink_block_table", "open_drawer"]' \
    --episodes_per_task 10 \
    --ep_len 240 \
    --num_sequences 5000
```

### 7. Interactive Rollouts
Launch a PyBullet OpenCV GUI to manually step through the CALVIN environment and feed natural language goals interactively to your trained SmolVLA via the terminal.
```bash
python calvin_models/calvin_agent/inference/rollouts_interactive.py \
    --dataset_path /path/to/calvin/dataset/task_D_D \
    --train_folder outputs/train/smolvla_model
```
*(Press 't' to pause the simulation window and type a new language instruction!)*

### 8. Hub Utilities
Push your converted datasets and trained models to the Hugging Face hub for easy sharing and remote training.
```bash
python scripts/push_dataset_to_hub.py --repo_id your_username/calvin_task_D_D_pick_place --dataset_path /path/to/calvin_task_D_D_pick_place
python scripts/push_model_to_hub.py --repo_id your_username/smolvla_calvin_pick_place --model_path outputs/train/smolvla_model/120000
```

---

## :framed_picture: Sensory Observations
CALVIN supports a range of sensors commonly utilized for visuomotor control:
1. **Static camera RGB images** - with shape `200x200x3`.
2. **Static camera Depth maps** - with shape `200x200`.
3. **Gripper camera RGB images** - with shape `84x84x3`.
4. **Gripper camera Depth maps** - with shape `84x84`.
5. **Tactile image** - with shape `120x160x6`.
6. **Proprioceptive state** - EE position (3), EE orientation in euler angles (3), gripper width (1), joint positions (7), gripper action (1).

<p align="center">
<img src="media/sensors.png" alt="" width="50%">
</p>

## :joystick: Action Space
In CALVIN, the agent must perform closed-loop continuous control to follow unconstrained language instructions characterizing complex robot manipulation tasks, sending continuous actions to the robot at 30hz. We support the following action spaces:
1. **Absolute cartesian pose** - EE position (3), EE orientation in euler angles (3), gripper action (1).
2. **Relative cartesian displacement** - EE position (3), EE orientation in euler angles (3), gripper action (1).
3. **Joint action** - Joint positions (7), gripper action (1).

## 🌍 Generalizing to New Environments
This codebase bridges the CALVIN simulator format via the extensible Hugging Face LeRobot standard. The components here are designed to be environment-agnostic:
1. **Conversion (`scripts/convert_lerobot.py`)**: Modify `state_names` and `action_names` precisely for your robot's kinematics.
2. **Evaluation (`calvin_agent/evaluation/evaluate_policy.py`)**: The generalized `CustomModel` class loads any `SmolVLAPolicy` securely. If your target simulator exposes generic RGB views and robot states mapping to your dataset properties, this pipeline provides variable-dimension video recording and execution transparently.

## 🧪 Testing & Reproducibility
To ensure stability and provide a demonstrable working prototype baseline:
- **Test Suite**: Run `pytest tests/` to validate metadata integrity and custom policy initialization without hardware.
- **Reproducibility**: Run `sh scripts/reproduce_experiment.sh` to trigger an automatic end-to-end pipeline containing minimal dataset download, mapping, inference training, and standalone evaluation.
