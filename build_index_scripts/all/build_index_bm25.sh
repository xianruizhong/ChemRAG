#!/bin/bash

corpus_path=""
save_dir=""

python -m flashrag.retriever.index_builder \
    --retrieval_method bm25 \
    --corpus_path $corpus_path \
    --bm25_backend pyserini \
    --save_dir $save_dir
