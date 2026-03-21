import argparse
import logging
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Upload a trained SmolVLA checkpoint to the Hugging Face Hub.")
    parser.add_argument("--repo_id", required=True, help="Hugging Face repository ID (e.g., your_username/smolvla_calvin_pick_place)")
    parser.add_argument("--model_path", required=True, help="Path to the checkpoint directory (containing config.json, model.safetensors, preprocessor/ etc.)")
    parser.add_argument("--path_in_repo", default=".", help="Where to upload inside the repo (default is root '.')")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    api = HfApi()
    
    logging.info(f"Creating or checking model repository {args.repo_id}...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    
    logging.info(f"Uploading checkpoint from {args.model_path} to {args.repo_id} path {args.path_in_repo}...")
    api.upload_folder(
        folder_path=args.model_path,
        repo_id=args.repo_id,
        repo_type="model",
        path_in_repo=args.path_in_repo
    )
    logging.info(f"✅ Model checkpoint successfully pushed to https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()
