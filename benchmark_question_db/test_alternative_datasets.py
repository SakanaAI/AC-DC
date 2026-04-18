"""Test alternative dataset names for failed benchmarks"""

from datasets import load_dataset


def test_dataset(name, subset=None, split="test", description=""):
    """Test if a dataset can be loaded"""
    print(f"\nTesting: {name} {f'({subset})' if subset else ''} - {description}")
    try:
        if subset:
            ds = load_dataset(name, subset, split=split)
        else:
            ds = load_dataset(name, split=split)
        print(f"  ✓ Success! {len(ds)} samples")
        print(f"  Columns: {ds.column_names}")
        if "subject" in ds.column_names:
            subjects = set(ds["subject"])
            print(f"  Subjects: {len(subjects)}")
        elif "category" in ds.column_names:
            categories = set(ds["category"])
            print(f"  Categories: {len(categories)}")
        elif "task" in ds.column_names:
            tasks = set(ds["task"])
            print(f"  Tasks: {len(tasks)}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:100]}")
        return False


# Test alternatives for GPQA
print("=" * 80)
print("GPQA Alternatives")
print("=" * 80)
test_dataset("Idavidrein/gpqa", "gpqa_main", "train", "Original (gated)")
test_dataset("qnguyen3/gpqa", None, "train", "Alternative mirror")

# Test alternatives for Big Bench Hard
print("\n" + "=" * 80)
print("Big Bench Hard Alternatives")
print("=" * 80)
test_dataset("lukaemon/bbh", None, "test", "lukaemon's version")
test_dataset("lighteval/big_bench_hard", None, "test", "lighteval version")
test_dataset("tasksource/bigbench", "bbh", "default", "tasksource version")

# Test alternatives for MATH
print("\n" + "=" * 80)
print("MATH Dataset Alternatives")
print("=" * 80)
test_dataset("competition_math", None, "test", "competition_math")
test_dataset("lighteval/MATH", None, "test", "lighteval version")
test_dataset("hendrycks/math", None, "test", "Original (if works)")
test_dataset("hendrycks_math", None, "test", "Alternative name")
test_dataset("EleutherAI/hendrycks_math", None, "test", "EleutherAI version")

print("\n" + "=" * 80)
print("Testing Complete")
print("=" * 80)
