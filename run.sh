#!/bin/bash

task=$1 # e.g. "product", "shop", "voucher", "web"
config=$2 # e.g. "rollout", "ablation_python_execute", "ablation_react", "simpleqa_rollout"
model_name=$3 # optional model name to run

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs
mkdir -p data/simpleqa

# Validate data folder structure
echo "Validating data folder structure..."
required_files=(
    "data/synthesize_product_test.jsonl"
    "data/synthesize_shop_test.jsonl"
    "data/synthesize_voucher_test.jsonl"
    "data/synthesize_web_simpleqa_test.jsonl"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "Error: Missing required files in data directory:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    echo "Please place the required files in the data directory before running this script."
    exit 1
fi

echo "All required data files found."

if [ -n "$model_name" ]; then
    # Run with specified model name
    models_to_run="$model_name"
else
    # Run with all models
    models1="gpt-4.1"
    models2="o3-mini gpt-4o gpt-4o-mini gemini-2.5-flash claude-4-sonnet qwen-max deepseek-r1 deepseek-v3 qwen3-235b-a22b-instruct qwen3-235b-a22b qwen3-32b qwen3-14b qwen3-8b qwen3-4b"
    models3="gemma-3-27b-it gemma-3-12b-it gemma-3-4b-it"
    models_to_run="${models1}"
fi

for model in ${models_to_run}; do
    if [ -e config/${config}/${model}.json ]; then
        echo "----------"
        echo ${model}
        git checkout -- config/${config}/${model}.json

        # 1. modify the task
        cat config/${config}/${model}.json | sed "s/product/${task}/g" > .tmp; mv .tmp config/${config}/${model}.json

        # 2. modify the threads
        cat config/${config}/${model}.json | sed 's/\"threads\": 4/\"threads\": 1/g' > a; mv a config/${config}/${model}.json

        # 3. modify the synthesize file
        cat config/${config}/${model}.json | sed "s/synthesize_${task}.jsonl/synthesize_${task}_test.jsonl/g" > .tmp; mv .tmp config/${config}/${model}.json

        # 4. rollout
        nohup python src/agent/run_rollout.py config/${config}/${model}.json > logs/${config}_${task}_${model} 2>&1 &

        # 5. evaluate
        # python src/agent/run_evaluate.py config/${config}/${model}.json

        # 6. kill
        #ps aux | grep run_rollout.py | grep ${model} | awk '{print $2}' | xargs kill -9

        if [ -e data/${config}_${task}_${model}.jsonl ]; then
            # 7. empty reasoning_content and content
            a=`wc -l data/${config}_${task}_${model}.jsonl | awk '{print $1}'`
            b=`grep '"reasoning_content":"","content":""' data/${config}_${task}_${model}.jsonl | wc -l`
            div=`awk "BEGIN {print ${b} / ${a}}"`
            echo "samples: $a, empty content ratio: $div"
            #grep -v '"reasoning_content":"","content":""' data/${config}_${task}_${model}.jsonl > .tmp; mv .tmp data/${config}_${task}_${model}.jsonl

            # 8. null results
            a=`wc -l data/${config}_${task}_${model}.jsonl | awk '{print $1}'`
            b=`grep '"results":null' data/${config}_${task}_${model}.jsonl | wc -l`
            div=`awk "BEGIN {print ${b} / ${a}}"`
            echo "samples: $a, null results ratio: $div"
            #grep -v '"results":null' data/${config}_${task}_${model}.jsonl > .tmp; mv .tmp data/${config}_${task}_${model}.jsonl

            # 9. no terminate
            a=`wc -l data/${config}_${task}_${model}.jsonl | awk '{print $1}'`
            b=`grep -v '"name":"terminate"' data/${config}_${task}_${model}.jsonl | wc -l`
            div=`awk "BEGIN {print ${b} / ${a}}"`
            echo "samples: $a, no terminate ratio: $div"
            #grep '"name":"terminate"' data/${config}_${task}_${model}.jsonl > .tmp; mv .tmp data/${config}_${task}_${model}.jsonl
        fi
    fi
done
