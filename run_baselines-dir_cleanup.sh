# !/bin/bash

# Parse command line arguments
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run]"
            echo "  --dry-run        Dry run mode - no files will be moved or deleted"
            echo "  -h, --help       Show this help message and exit"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run]"
            echo "  --dry-run        Dry run mode - no files will be moved or deleted"
            echo "  -h, --help       Show this help message and exit"
            exit 1
            ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - No files will be moved or deleted"
    echo ""
fi

START_SEEDS=(50 58)

# Cleanup

## Result files are saved in outputs/baselines/qwen2.5/$SEED/<lm_harness_name>/<model_name>/seed_XX/<model_name>/
## This function organizes the data so that the result files are saved in outputs/baselines/qwen2.5/$SEED/<lm_harness_name>/<model_name>/seed_XX/
organize_result_files(){
    local lm_harness_dir=$1

    if [ -d "$lm_harness_dir" ]; then
        echo "Processing $lm_harness_dir directory..."
        for model_dir in "$lm_harness_dir"/*; do
            if [ -d "$model_dir" ]; then
                for seed_dir in "$model_dir"/seed_*; do
                    if [ -d "$seed_dir" ]; then
                        # Files are in .../<model_name>/seed_XX/<model_name>/
                        source_dir="$seed_dir/$(basename "$model_dir")"
                        target_dir="$seed_dir"
                        if [ -d "$source_dir" ]; then
                            if [ "$DRY_RUN" = true ]; then
                                echo "  [DRY RUN] Would move files from $source_dir to $target_dir"
                                echo "    [DRY RUN] Would remove directory: $source_dir"
                            else
                                echo "  Moving files from $source_dir to $target_dir"
                                find "$source_dir" -maxdepth 1 -type f \( -name "*.json" -o -name "*.jsonl" \) -exec mv -t "$target_dir" {} +
                                rmdir "$source_dir" 2>/dev/null || true
                            fi
                        fi
                    fi
                done
            fi
        done
    fi
}

# Run the function on all result lm_harness dirs
for SEED in "${START_SEEDS[@]}"; do
    for LM_HARNESS_NAME in "lm_harness" "lm_harness_code"; do
        organize_result_files "outputs/baselines/qwen2.5/$SEED/$LM_HARNESS_NAME"
    done
done