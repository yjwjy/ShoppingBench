yaml_file=$1
export WANDB_PROJECT="qwen3-sft"

llamafactory-cli train $yaml_file
