#!/bin/bash

model_path=""
corpus_path=""
save_dir=""

python -m flashrag.retriever.index_builder \
    --retrieval_method $model \
    --model_path $model_path \
    --corpus_path $corpus_path \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 512 \
    --batch_size 256 \
    --pooling_method mean \
    --faiss_type Flat
