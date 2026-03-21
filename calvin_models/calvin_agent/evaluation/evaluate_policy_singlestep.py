import argparse
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import sys

import hydra
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch

# Inherit the exact same LeRobot setup logic
from evaluate_policy import CustomModel, make_env

# Import CALVIN utils
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import join_vis_lang, print_and_save, get_log_dir

logger = logging.getLogger(__name__)

def evaluate_policy_singlestep(model, env, eval_log_dir=None, debug=False, target_tasks=None, ep_len=240, num_sequences=5000):
    """
    Evaluates the model on exactly one isolated subtask at a time.
    """
    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    ALL_SEQS = get_sequences(num_sequences)
    
    # Select sequences that start with our target tasks
    eval_sequences = []
    
    if target_tasks is None:
        target_tasks = {"lift_pink_block_table": 10, "lift_blue_block_table": 10}
        
    counts = {k: 0 for k in target_tasks.keys()}
    
    for init, seq in ALL_SEQS:
        if len(seq) >= 1:
            first_task = seq[0]
            if first_task in target_tasks and counts[first_task] < target_tasks[first_task]:
                eval_sequences.append((init, first_task))
                counts[first_task] += 1
                
        if all(counts[k] >= target_tasks[k] for k in target_tasks):
            break

    print("\n===== ISOLATED SEQUENCES =====")
    for task, count in counts.items():
        print(f" - {task:25s} : {count} / {target_tasks[task]}")
    print("==============================\n")

    results = Counter()

    for init_state, task in eval_sequences:
        # Get language annotation for the task
        lang_annotation = val_annotations[task][0]

        # Reset env to exact initial state
        obs = env.reset(robot_obs=init_state["robot_obs"][0], scene_obs=init_state["scene_obs"][0])
        model.reset()
        start_info = env.get_info()

        success = False
        for step in range(ep_len):
            action = model.step(obs, lang_annotation)
            obs, _, _, current_info = env.step(action)
            
            if debug:
                img = env.render(mode="rgb_array")
                join_vis_lang(img, lang_annotation)
                
            # check if current step solves a task
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {task})
            if len(current_task_info) > 0:
                if debug:
                    print(colored("S", "green"), end=" ")
                success = True
                break
                
        if not success and debug:
            print(colored("F", "red"), end=" ")

        if success:
            results[task] += 1

    print("\n\n===== FINAL SINGLE-STEP RESULTS =====")
    for task in target_tasks.keys():
        success_rate = (results[task] / target_tasks[task]) * 100 if target_tasks[task] > 0 else 0
        print(f"{task:25s}: {results[task]}/{target_tasks[task]} ({success_rate:.1f}%)")
        
    print_and_save(results, eval_sequences, {"results": results}, eval_log_dir)
    return results

def main():
    seed_everything(0, workers=True)
    parser = argparse.ArgumentParser(description="Evaluate a trained LeRobot model on isolated single tasks.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to CALVIN dataset directory (e.g. task_D_D).")
    parser.add_argument("--train_folder", type=str, required=True, help="Path to trained SmolVLA checkpoint folder.")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device")
    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment OpenCV.")
    parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")
    
    # Target tasks input
    parser.add_argument("--target_tasks", type=str, default=None, help="JSON list of task names to test isolated, e.g. '[\"lift_pink_block_table\", \"open_drawer\"]'")
    parser.add_argument("--episodes_per_task", type=int, default=10, help="Number of attempts per task")
    parser.add_argument("--ep_len", type=int, default=240, help="Length of the evaluation episode horizon")
    parser.add_argument("--num_sequences", type=int, default=5000, help="Size of the raw sequence data to parse goals from")
    
    args = parser.parse_args()

    target_tasks = None
    if args.target_tasks:
        import json
        tasks_list = json.loads(args.target_tasks)
        target_tasks = {t: args.episodes_per_task for t in tasks_list}
        
    print("Initializing CustomModel for LeRobot...")
    model = CustomModel(checkpoint_dir=args.train_folder, device=args.device)
    
    print("Loading PyBullet Environment...")
    env = make_env(args.dataset_path)

    print("Beginning Single-Step evaluation...")
    evaluate_policy_singlestep(
        model, 
        env, 
        eval_log_dir=args.eval_log_dir, 
        debug=args.debug, 
        target_tasks=target_tasks,
        ep_len=args.ep_len,
        num_sequences=args.num_sequences
    )

if __name__ == "__main__":
    main()
