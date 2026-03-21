import argparse
import numpy as np
from collections import Counter
from pathlib import Path

# ============================================================
# Utils
# ============================================================
def contains_excluded_keyword(task: str, keywords) -> bool:
    task_lower = task.lower()
    return any(kw in task_lower for kw in keywords)

def main():
    parser = argparse.ArgumentParser(description="Inspect CALVIN dataset annotations and rank tasks.")
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Path to the dataset directory (e.g., /mnt/d/LeRobot/calvin/dataset/task_D_D)")
    parser.add_argument("--top_k", type=int, default=200, 
                        help="Number of top tasks to display")
    parser.add_argument("--min_train", type=int, default=15, 
                        help="Minimum number of training occurrences to select a task")
    parser.add_argument("--min_val", type=int, default=5, 
                        help="Minimum number of validation occurrences to select a task")
    parser.add_argument("--excluded_keywords", nargs="*", default=[], 
                        help="Keywords to exclude from task selection")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    train_ann_path = dataset_path / "training" / "lang_annotations" / "auto_lang_ann.npy"
    val_ann_path = dataset_path / "validation" / "lang_annotations" / "auto_lang_ann.npy"

    # ============================================================
    # Load TRAIN annotations
    # ============================================================
    if not train_ann_path.exists():
        print(f"Error: Training annotations not found at {train_ann_path}")
        return

    ann_train = np.load(train_ann_path, allow_pickle=True).item()
    
    # Normally "task" holds the task names, but we fallback to "ann" just in case
    # as in the provided snippet
    if "task" in ann_train["language"]:
        texts_train = ann_train["language"]["task"]
    else:
        texts_train = ann_train["language"]["ann"]
        
    # Flatten if it's an array of arrays
    if isinstance(texts_train, np.ndarray) and texts_train.ndim > 1:
        texts_train = texts_train.flatten().tolist()
    elif isinstance(texts_train, list) and len(texts_train) > 0 and isinstance(texts_train[0], list):
        texts_train = [item for sub in texts_train for item in sub]

    counter_train = Counter(texts_train)

    print("===== TOP 50 TASKS (TRAIN) =====")
    for task, count in counter_train.most_common(50):
        print(f"{count:5d}  |  {task}")


    # ============================================================
    # Load VALIDATION annotations
    # ============================================================
    if not val_ann_path.exists():
        print(f"Error: Validation annotations not found at {val_ann_path}")
        return

    ann_val = np.load(val_ann_path, allow_pickle=True).item()
    
    if "task" in ann_val["language"]:
        texts_val = ann_val["language"]["task"]
    else:
        texts_val = ann_val["language"]["ann"]
        
    if isinstance(texts_val, np.ndarray) and texts_val.ndim > 1:
        texts_val = texts_val.flatten().tolist()
    elif isinstance(texts_val, list) and len(texts_val) > 0 and isinstance(texts_val[0], list):
        texts_val = [item for sub in texts_val for item in sub]
        
    counter_val = Counter(texts_val)

    print("\n===== TOP 50 TASKS (VALIDATION) =====")
    for task, count in counter_val.most_common(50):
        print(f"{count:5d}  |  {task}")


    # ============================================================
    # All tasks (union)
    # ============================================================
    all_tasks = set(counter_train.keys()) | set(counter_val.keys())

    # ============================================================
    # Excluded tasks (for inspection)
    # ============================================================
    if args.excluded_keywords:
        print("\n===== EXCLUDED TASKS (by keyword) =====")
        for task in sorted(all_tasks):
            if contains_excluded_keyword(task, args.excluded_keywords):
                print(task)


    # ============================================================
    # Rank tasks by TOTAL frequency (train + val)
    # ============================================================
    rows = []
    for task in all_tasks:
        if args.excluded_keywords and contains_excluded_keyword(task, args.excluded_keywords):
            continue

        ct = counter_train.get(task, 0)
        cv = counter_val.get(task, 0)
        total = ct + cv
        ratio = cv / ct if ct > 0 else 0.0

        rows.append((task, ct, cv, total, ratio))

    # Sort by total occurrences (descending)
    rows.sort(key=lambda x: x[3], reverse=True)

    # Keep top-K
    rows = rows[:args.top_k]


    # ============================================================
    # Print ranked tasks
    # ============================================================
    print("\n===== TASKS RANKED BY TOTAL FREQUENCY (TRAIN + VAL) =====")
    print("train |  val  | total | val/train | task")
    print("-" * 100)

    for task, ct, cv, total, ratio in rows:
        print(f"{ct:5d} | {cv:5d} | {total:5d} | {ratio:9.3f} | {task}")


    # ============================================================
    # Optional: hard filter by minimum counts
    # ============================================================
    selected_tasks = [
        task for task, ct, cv, total, ratio in rows
        if ct >= args.min_train and cv >= args.min_val
    ]

    print("\n===== FINAL SELECTED TASKS =====")
    print(f"Criteria: train >= {args.min_train}, val >= {args.min_val}")
    for task in selected_tasks:
        print(
            f"{counter_train.get(task, 0):5d} train | "
            f"{counter_val.get(task, 0):5d} val | "
            f"{task}"
        )

    print(f"\nTotal selected tasks: {len(selected_tasks)}")

if __name__ == "__main__":
    main()
