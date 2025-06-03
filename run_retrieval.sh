#!/bin/bash

data_name=mmlu_chem
sub_data_name=test

retriever=specter
corpus=uspto

python basic_rag.py --model_path $MODEL_PATH \
                    --sub_data_name $sub_data_name \
                    --corpus_path $CORPUS_PATH \
                    --index_path $INDEX_PATH \
                    --data_path $DATA_PATH \
                    --dataset_name $data_name \
                    --gpu_id 0,1 \
                    --topk $topk \
                    --retriever $retriever \
                    --do_retrieval \
                    --save_retrieval_only \
                    --save_dataset_dir $RETRIEVAL_CACHE_DIR
