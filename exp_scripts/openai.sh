#!/bin/bash

data_name=scibench
sub_data_names=("chemmc" "matter" "quan")


retrievers=("rrf4")
corpora=("merged")
models=("gpt-3.5-turbo" "gpt-4o" "o1")

api_key=""

# First conduct baseline without retrieval
for model in "${models[@]}"; do
    if [ "$sub_data_name" == "property_prediction" ]; then
        metrics="scibench_score retrieval_recall retrieval_precision"
    else
        metrics="em fingerprint bleu rouge-l retrieval_recall retrieval_precision"
    fi

    echo "Running baseline experiment with model: $model"
    python run_experiment.py --model_path placeholder \
                    --sub_data_name $sub_data_name \
                    --data_path $DATA_PATH \
                    --dataset_name $data_name \
                    --gpu_id $GPU_ID \
                    --gpu_util $GPU_UTIL \
                    --cached_dataset_dir $baseline_cache_dir \
                    --generator_model $model \
                    --generator_max_input_len 32000 \
                    --save_dir ./exp_results/$data_name/$sub_data_name/$model/baseline/ \
                    --corpus_name baseline \
                    --model_name $model \
                    --framework openai \
                    --api_key $api_key
done


# Retrieve for each model
echo "Running retrieval experiments"
for retriever in "${retrievers[@]}"; do
    for topk in "${topk[@]}"; do
        for corpus in "${corpora[@]}"; do
            for model in "${models[@]}"; do
                echo "Running experiment with model: $model, corpus: $corpus, topk: $topk"

                python run_experiment.py --model_path placeholder \
                            --data_path $DATA_PATH \
                            --dataset_name $data_name \
                            --gpu_id $GPU_ID \
                            --gpu_util $GPU_UTIL \
                            --cached_dataset_dir $RETRIEVAL_CACHE_DIR \
                            --generator_model $model \
                            --generator_max_input_len 32000 \
                            --save_dir ./exp_results/$data_name/$sub_data_name/$model/$retriever/$corpus/ \
                            --corpus_name $corpus \
                            --do_retrieval \
                            --topk $topk \
                            --model_name $model \
                            --framework openai \
                            --api_key $api_key
            done
        done
    done
done
