# ChemRAG: Benchmarking Retrieval-Augmented Generation for Chemistry

<div style='display:flex; gap: 0.25rem; flex-wrap: wrap; align-items: center;'>
  <a href='https://arxiv.org/abs/2505.07671'>
    <img src='https://img.shields.io/badge/Paper-PDF-red'>
  </a>
   <a href='https://chemrag.github.io'>
  <img src='https://img.shields.io/badge/Website-8A2BE2' alt='website'>
   </a>
   <a href='https://huggingface.co/ChemRAG/datasets'>
     <img src='https://img.shields.io/badge/🤗-Data--and--Index-FFD21B.svg' alt='data'>
   </a>
</div>

**ChemRAG** is a toolkit for systemaically studying RAG for chemistry tasks. We introduce ChemRAG-Toolkit, a modular framework that supports multiple retrievers and LLMs. The toolkit integrates various chemistry-specific corpora (**PubChem, PubMed, USPTO, Semantic Scholar, Chemistry Textbook, and Wikipedia**) and evaluates across a variety of tasks.

## Update
Our paper has been accepted by **COLM 2025**.

## Todo
1. Upload pubmed index.
2. Upload detailed experiment scripts.

## Data
1. Corpus Data: The processed text and index can be found at https://huggingface.co/ChemRAG/datasets.
2.  Question Data: The questions and ground-truth answers are under `data/{task_name}`, for example `data/ChemBench4K`.

## Installation
```
conda create --name chemrag python==3.10
conda activate chemrag

pip install -e .
pip install vllm==0.5.4
pip install pyserini
conda install conda-forge::faiss-gpu
```
Installing faiss-gpu could be tricky, you may also want to try the cpu version `conda install -c pytorch faiss-cpu`.

If you encounter `ImportError: libmkl_intel_lp64.so.1: cannot open shared object file: No such file or directory`, try to fix it by installing mlk
```
conda install mkl==2021
```

### Download a LLM from hf
You may want to download a model to a specific location:
```
from huggingface_hub import snapshot_download

model_name = "xxx"  # Replace with your desired model
local_path = "/path/to/save/model"  # Specify your desired local directory

download_dir = snapshot_download(repo_id=model_name, local_dir=local_path)
```

## Build your own index
If you want to build your own index, please use the script under `build_index_scripts`. 

## Retrieval
Retrieval cache is under `retrieval_cache`. If you want to retrieve by yourself, please refer to `run_retrieval.sh` for running retrieval given queries.

## Run Experiment
Please refer to `exp_scripts` folder for running the experiment.

## Citation
```
@misc{zhong2025benchmarkingretrievalaugmentedgenerationchemistry,
      title={Benchmarking Retrieval-Augmented Generation for Chemistry}, 
      author={Xianrui Zhong and Bowen Jin and Siru Ouyang and Yanzhen Shen and Qiao Jin and Yin Fang and Zhiyong Lu and Jiawei Han},
      year={2025},
      eprint={2505.07671},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.07671}, 
}
```