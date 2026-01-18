import argparse
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import sys
import time

# Path to the LeRobot *repo root* (the one that contains lerobot/)
LEROBOT_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../lerobot/src")
)

print("Adding LeRobot repo root to PYTHONPATH:", LEROBOT_REPO_ROOT)

if LEROBOT_REPO_ROOT not in sys.path:
    sys.path.insert(0, LEROBOT_REPO_ROOT)

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from calvin_env.envs.play_table_env import get_env

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.configs.types import PolicyFeature, FeatureType
import json
from safetensors.torch import load_file


logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000


def get_epoch(checkpoint):
    if "=" not in checkpoint.stem:
        return "0"
    checkpoint.stem.split("=")[1]


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env

def normalize_features(features: dict | None):
    if features is None:
        return None

    out = {}
    for name, ft in features.items():
        if isinstance(ft, PolicyFeature):
            out[name] = ft
        elif isinstance(ft, dict):
            out[name] = PolicyFeature(
                type=FeatureType(ft["type"]),
                shape=tuple(ft.get("shape", ())),
            )
        else:
            raise TypeError(f"Unsupported feature type for {name}: {type(ft)}")
    return out


class CustomModel(CalvinBaseModel):
    def __init__(self, checkpoint_dir, device="cpu"):
        checkpoint_dir = Path(checkpoint_dir)
        model_dir = checkpoint_dir / "pretrained_model"

        self.device = torch.device(device)

        # -------------------------
        # 1. Load preprocessors
        # -------------------------
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            config_filename="policy_preprocessor.json",
            overrides={"device_processor": {"device": "cpu"}},
        )

        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            config_filename="policy_postprocessor.json",
            overrides={"device_processor": {"device": "cpu"}},
        )

        # -------------------------
        # 2. Load config
        # -------------------------
        with open(model_dir / "config.json", "r") as f:
            cfg_dict = json.load(f)

        cfg_dict = dict(cfg_dict)          # sécurité
        cfg_dict.pop("type", None)         # ← LIGNE CRUCIALE
        config = SmolVLAConfig(**cfg_dict)
        config.input_features = normalize_features(config.input_features)
        config.output_features = normalize_features(config.output_features)

        # -------------------------
        # 3. Instantiate policy
        # -------------------------
        self.policy = SmolVLAPolicy(config)
        self.policy.to(self.device)

        # -------------------------
        # 4. Load weights
        # -------------------------
        state_dict = load_file(
            str(model_dir / "model.safetensors"),
            device=str(self.device),
        )
        self.policy.load_state_dict(state_dict, strict=False)


        self.policy.eval()
        self.reset()

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def step(self, obs, goal):
        batch = {
            "observation.images.camera1": torch.from_numpy(
                obs["rgb_obs"]["rgb_static"]
            ).permute(2, 0, 1),

            "observation.images.camera2": torch.from_numpy(
                obs["rgb_obs"]["rgb_gripper"]
            ).permute(2, 0, 1),

            "observation.state": torch.from_numpy(
                obs["robot_obs"]
            ).to(dtype=self.policy.model.state_proj.weight.dtype),

            "task": goal,
        }

        # batch dim + device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0).to(self.device)
            else:
                # e.g. task string → leave untouched
                batch[k] = v

        batch = self.preprocessor(batch)

        action = self.policy.select_action(batch)

        action = self.postprocessor({"action": action})["action"]

        action = action.squeeze(0).cpu().numpy()

        ee_pos = action[:3]
        ee_orn = action[3:6]

        gripper_continuous = action[6]
        gripper = 1 if gripper_continuous > 0 else -1

        return (ee_pos, ee_orn, gripper)




def evaluate_policy(model, env, epoch, eval_log_dir=None, debug=False, create_plan_tsne=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    results = []
    plans = defaultdict(list)

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)
    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    for step in range(EP_LEN):
        action = model.step(obs, lang_annotation)
        obs, _, _, current_info = env.step(action)
        if debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(model, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
    return False


def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )

    # arguments for loading custom model or custom language embeddings
    parser.add_argument(
        "--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

    parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    # evaluate a custom model
    if args.custom_model:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CustomModel(
        checkpoint_dir=Path(args.checkpoint),
        device=device,

    )
        env = make_env(args.dataset_path)
        evaluate_policy(
        model,
        env,
        epoch="custom",
        eval_log_dir=args.eval_log_dir,
        debug=args.debug,
    )
    else:
        assert "train_folder" in args

        checkpoints = []
        if args.checkpoints is None and args.last_k_checkpoints is None and args.checkpoint is None:
            print("Evaluating model with last checkpoint.")
            checkpoints = [get_last_checkpoint(Path(args.train_folder))]
        elif args.checkpoints is not None:
            print(f"Evaluating model with checkpoints {args.checkpoints}.")
            checkpoints = get_checkpoints_for_epochs(Path(args.train_folder), args.checkpoints)
        elif args.checkpoints is None and args.last_k_checkpoints is not None:
            print(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
            checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints :]
        elif args.checkpoint is not None:
            checkpoints = [Path(args.checkpoint)]

        env = None
        for checkpoint in checkpoints:
            epoch = get_epoch(checkpoint)
            model, env, _ = get_default_model_and_env(
                args.train_folder,
                args.dataset_path,
                checkpoint,
                env=env,
                device_id=args.device,
            )
            evaluate_policy(model, env, epoch, eval_log_dir=args.eval_log_dir, debug=args.debug, create_plan_tsne=True)


if __name__ == "__main__":
    main()
