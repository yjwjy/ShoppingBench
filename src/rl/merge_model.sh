python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/#your_experiment_name/actor \
    --target_dir outputs/merged_hf_model/#your_model_name