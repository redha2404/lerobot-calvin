import argparse
import logging
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Upload a converted LeRobot dataset (Parquet & MP4s) to the Hugging Face Hub.")
    parser.add_argument("--repo_id", required=True, help="Hugging Face repository ID (e.g., your_username/calvin_dataset)")
    parser.add_argument("--dataset_path", required=True, help="Local path to the converted dataset")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    api = HfApi()
    
    logging.info(f"Creating or checking dataset repository {args.repo_id}...")
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
    
    logging.info(f"Uploading files from {args.dataset_path} ... this might take a while.")
    api.upload_folder(
        folder_path=args.dataset_path,
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    logging.info(f"✅ Dataset successfully pushed to https://huggingface.co/datasets/{args.repo_id}")

if __name__ == "__main__":
    main()
