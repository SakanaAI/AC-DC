"""Test alternative datasets with proper configs"""

from datasets import load_dataset, get_dataset_config_names


def explore_multiconfig_dataset(name):
    """Explore datasets that have multiple configs"""
    print(f"\n{'='*80}")
    print(f"Dataset: {name}")
    print(f"{'='*80}")

    try:
        # Get all available configs
        configs = get_dataset_config_names(name)
        print(f"\nAvailable configs ({len(configs)}): {configs[:10]}")
        if len(configs) > 10:
            print(f"... and {len(configs) - 10} more")

        # Try loading the first config as a sample
        if configs:
            sample_config = configs[0]
            print(f"\nLoading sample config: {sample_config}")
            ds = load_dataset(name, sample_config, split="test")
            print(f"  Total samples: {len(ds)}")
            print(f"  Columns: {ds.column_names}")
            print(f"\n  First example:")
            example = ds[0]
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 150:
                    print(f"    {key}: {value[:150]}...")
                else:
                    print(f"    {key}: {value}")
        return configs

    except Exception as e:
        print(f"ERROR: {e}")
        return []


# Test BBH with configs
configs_bbh = explore_multiconfig_dataset("lukaemon/bbh")

# Test MATH with configs
configs_math = explore_multiconfig_dataset("EleutherAI/hendrycks_math")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"BBH has {len(configs_bbh)} tasks")
print(f"MATH has {len(configs_math)} categories")
