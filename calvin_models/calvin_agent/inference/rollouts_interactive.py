import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_lightning import seed_everything
import sys

# Assume this script runs within the main calvin repo directory
sys.path.insert(0, str(Path(__file__).absolute().parents[2]))

# Assuming calvin_agent/evaluation/evaluate_policy.py's CustomModel
# We will construct an absolute import path for CustomModel
from calvin_models.calvin_agent.evaluation.evaluate_policy import CustomModel, make_env

logger = logging.getLogger(__name__)

def imshow_tensor(window, img_tensor, wait=0, resize=True, keyhandler=None):
    """
    Shows a tensor image via OpenCV
    """
    img = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    # BGR format for CV2
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if resize:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 600, 600)
    cv2.imshow(window, img)
    if keyhandler is not None:
        return cv2.waitKey(wait)
    else:
        cv2.waitKey(wait)


def interactive_rollout(model, env, initial_lang_goal, max_steps=500):
    print("\n" + "="*50)
    print("🤖 INTERACTIVE LEROBOT ROLLOUT")
    print(f"Goal: '{initial_lang_goal}'")
    print("="*50)
    print("Controls (Make sure OpenCV window is focused):")
    print(" [t] : Pause and Type a new text instruction")
    print(" [n] : End current rollout")
    print("="*50)

    obs = env.reset()
    lang_goal = initial_lang_goal
    
    cv2.namedWindow("LeRobot Calvin Agent", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("LeRobot Calvin Agent", 600, 600)

    for step in range(max_steps):
        # 1. Get Action from LeRobot CustomModel wrapper
        # The wrapper expects obs["rgb_obs"] exactly as given by CALVIN env
        action_tuple = model.step(obs, lang_goal)
        
        # LeRobot wrapper returns (relative_pos, relative_euler, gripper_action)
        action_array = np.concatenate([action_tuple[0], action_tuple[1], [action_tuple[2]]])
        
        # 2. Step physics
        obs, _, _, current_info = env.step(action_array)

        # 3. Visualize & Handle Input
        img = env.render(mode="rgb_array")
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Overlay current instruction text
        cv2.putText(img_bgr, f"Goal: {lang_goal}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("LeRobot Calvin Agent", img_bgr)
        k = cv2.waitKey(1) % 256

        if k == ord('t'):
            print("\n[PAUSED] Enter new command:")
            # Wait for user input in terminal
            new_goal = input("> ")
            if new_goal.strip():
                lang_goal = new_goal.strip()
                print(f"Goal updated to: '{lang_goal}'")
                
        elif k == ord('n') or k == 27: # 'n' or ESC
            print("\n[STOPPING ROLLOUT]")
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Interactive Rollout for LeRobot policies.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to CALVIN dataset (e.g. task_D_D)")
    parser.add_argument("--train_folder", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda/cpu)")
    args = parser.parse_args()

    seed_everything(0, workers=True)

    # Load Model (CustomModel wrapper we created for SmolVLA)
    print(f"Loading model from {args.train_folder}...")
    model = CustomModel(checkpoint_dir=args.train_folder, device=args.device)

    # Make Environment
    print(f"Loading environment from {args.dataset_path}...")
    env = make_env(args.dataset_path)

    while True:
        lang_goal = input("\nEnter initial instruction (or 'q' to quit): ")
        if lang_goal.lower() == 'q':
            break
            
        model.reset()
        interactive_rollout(model, env, lang_goal)

    print("Exiting...")

if __name__ == "__main__":
    main()
