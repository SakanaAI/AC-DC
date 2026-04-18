"""
Exploration script to understand the structure of each benchmark dataset.
Run this to document the schema of each dataset before implementing loaders.
"""

import sys
from datasets import load_dataset


def explore_dataset(name, subset=None, split="test", limit=2):
    """Load and print structure of a dataset"""
    print(f"\n{'='*80}")
    print(f"Dataset: {name}")
    if subset:
        print(f"Subset: {subset}")
    print(f"Split: {split}")
    print(f"{'='*80}\n")

    try:
        if subset:
            ds = load_dataset(name, subset, split=split)
        else:
            ds = load_dataset(name, split=split)

        print(f"Total samples: {len(ds)}")
        print(f"\nFeatures: {ds.features}")
        print(f"\nColumn names: {ds.column_names}")

        # Check if there are subgroups/categories
        if "subject" in ds.column_names:
            subjects = set(ds["subject"])
            print(f"\nSubjects/Categories ({len(subjects)}): {sorted(subjects)[:10]}...")
        elif "category" in ds.column_names:
            categories = set(ds["category"])
            print(
                f"\nCategories ({len(categories)}): {sorted(categories)[:10]}..."
            )
        elif "task" in ds.column_names:
            tasks = set(ds["task"])
            print(f"\nTasks ({len(tasks)}): {sorted(tasks)[:10]}...")
        elif "type" in ds.column_names:
            types = set(ds["type"])
            print(f"\nTypes ({len(types)}): {sorted(types)[:10]}...")

        print(f"\nFirst {limit} samples:")
        for i, example in enumerate(ds.select(range(min(limit, len(ds))))):
            print(f"\n--- Sample {i+1} ---")
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 200:
                    print(f"{key}: {value[:200]}...")
                else:
                    print(f"{key}: {value}")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    datasets_to_explore = [
        # (name, subset, split)
        ("cais/mmlu", "all", "test"),
        ("TIGER-Lab/MMLU-Pro", None, "test"),
        ("Idavidrein/gpqa", "gpqa_main", "train"),
        ("maveriq/bigbenchhard", None, "train"),
        ("openai/gsm8k", "main", "test"),
        ("hendrycks/math", None, "test"),
        ("openai/openai_humaneval", None, "test"),
        ("google-research-datasets/mbpp", None, "test"),
    ]

    print("Starting dataset exploration...")
    print(
        "This will help us understand the structure before implementing loaders."
    )

    results = {}
    for name, subset, split in datasets_to_explore:
        key = f"{name}_{subset}" if subset else name
        results[key] = explore_dataset(name, subset, split)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for key, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {key}")


if __name__ == "__main__":
    main()
