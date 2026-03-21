import argparse
import json
import logging
import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Bulk relabel task instructions in a LeRobot dataset.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the converted LeRobot dataset directory.")
    parser.add_argument("--mapping_json", type=str, help="Path to a JSON file containing { 'old_task_string' : 'new_task_string' }.")
    parser.add_argument("--interactive", action="store_true", help="If set, interactively prompt to rename each task.")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    tasks_parquet_path = dataset_path / "meta" / "tasks.parquet"
    
    if not tasks_parquet_path.exists():
        logger.error(f"Cannot find {tasks_parquet_path}. Is this a valid LeRobot dataset format?")
        return

    # Read tasks dataframe
    df = pd.read_parquet(tasks_parquet_path)
    
    # In LeRobot, the string instruction is typically saved as the index if created via write_tasks
    # Let's handle if it's a column or index.
    task_col = "task"
    if "task" not in df.columns:
        # Assuming the index holds the text
        strings = df.index.tolist()
        is_index = True
    else:
        strings = df["task"].tolist()
        is_index = False

    logger.info(f"Loaded {len(strings)} unique tasks.")

    mapping = {}
    if args.mapping_json:
        with open(args.mapping_json, "r") as f:
            mapping = json.load(f)
            
    elif args.interactive:
        print("\nInteractive Relabeling. Press Enter to keep the original label.\n")
        for old_string in strings:
            new_val = input(f"[{old_string}] -> ")
            if new_val.strip() != "":
                mapping[old_string] = new_val.strip()

    if not mapping:
        logger.info("No mapping provided. Exiting without modifications.")
        return

    # Apply mapping
    new_strings = []
    changes = 0
    for old_string in strings:
        if old_string in mapping:
            new_strings.append(mapping[old_string])
            changes += 1
        else:
            new_strings.append(old_string)
            
    if is_index:
        df.index = pd.Index(new_strings)
    else:
        df["task"] = new_strings

    # Save it back
    table = pa.Table.from_pandas(df)
    pq.write_table(table, tasks_parquet_path)
    
    logger.info(f"Successfully relabeled {changes} tasks in {tasks_parquet_path}.")
    
    # Optional: Update info.json or README.md if they mention specific strings, 
    # but normally the string is only rigorously read from tasks.parquet by LeRobot.
    
if __name__ == "__main__":
    main()
