#!/bin/bash

data_name=scibench
sub_data_names=("atkins" "chemmc" "matter" "quan")

retrievers=("rrf4")
corpora=("merged")
models=("Llama-3.1-8B-Instruct" "Mistral-7B-Instruct-v0.2" "ChemLLM-7B-Chat" "DeepSeek-R1-Distill-Llama-8B" "Llama-3.1-70B-Instruct")

topk=(5 10)
GPU_ID=0,1,2,3,4,5,6,7
GPU_UTIL=0.9

# baseline
for model in "${models[@]}"; do
    echo "Running baseline experiment with model: $model"
    python run_experiment.py --model_path $model_path \
                            --data_path $DATA_PATH \
                            --dataset_name $data_name \
                            --gpu_id $GPU_ID \
                            --gpu_util $GPU_UTIL \
                            --cached_dataset_dir $baseline_cache_dir \
                            --generator_model LLM \
                            --generator_max_input_len 32000 \
                            --max_generate_tokens 10000 \
                            --save_dir ./exp_results/$data_name/$model/cot/ \
                            --model_name $model \
                            --corpus_name baseline \
done

# Retrieve for each model
echo "Running retrieval experiments"
for retriever in "${retrievers[@]}"; do
    for corpus in "${corpora[@]}"; do
        for model in "${models[@]}"; do
            echo "Running experiment with model: $model, retriever: $retriever, corpus: $corpus"

            python run_experiment.py --model_path $model_path \
                                --corpus_path $CORPUS_PATH \
                                --index_path $INDEX_PATH \
                                --data_path $DATA_PATH \
                                --dataset_name $data_name \
                                --gpu_id $GPU_ID \
                                --gpu_util $GPU_UTIL \
                                --cached_dataset_dir $RETRIEVAL_CACHE_DIR \
                                --generator_model LLM \
                                --generator_max_input_len $MAX_LEN \
                                --model_name $model \
                                --save_dir ./exp_results/$data_name/$sub_data_name/$model/$retriever/$corpus/ \
                                --num_experiments 1 \
                                --topk 5 \
                                --corpus_name $corpus \
                                --do_retrieval
        done
    done
done


