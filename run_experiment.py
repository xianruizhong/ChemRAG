import argparse
from collections import defaultdict
from flashrag.config import Config
from flashrag.dataset import Dataset
from flashrag.utils import get_dataset, get_generator
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate
import json
import pandas as pd
import os


def get_metrics(dataset_name, sub_dataset_name):
    if dataset_name == "scibench":
        return ["scibench_score", "retrieval_recall", "retrieval_precision"]
    elif dataset_name == "mol-instruct":
        if sub_dataset_name == "property_prediction":
            return ["scibench_score", "retrieval_recall", "retrieval_precision"]
        elif sub_dataset_name == "molecular_description_generation":
            return ["bleu", "rouge-l", "retrieval_recall", "retrieval_precision"]
        else:
            return ["em", "fingerprint", "bleu", "retrieval_recall", "retrieval_precision"]
    else:
        return ["em", "f1", "acc", "retrieval_recall", "retrieval_precision"]
    


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--corpus_path", type=str)
parser.add_argument("--index_path", type=str)
parser.add_argument("--data_path", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--gpu_id", type=str)
parser.add_argument("--gpu_util", type=float, default=0.9)
parser.add_argument("--topk", type=int, default=10)
parser.add_argument("--retriever", type=str, default='bm25')
parser.add_argument("--do_retrieval", action="store_true")
parser.add_argument("--cached_dataset_dir", type=str, default=None)
parser.add_argument("--save_dataset_dir", type=str, default=None)
parser.add_argument("--save_retrieval_only", action="store_true")
parser.add_argument("--framework", type=str, default='vllm', choices=['vllm', 'hf', 'openai'])
parser.add_argument("--generator_model", type=str, default='LLM')
parser.add_argument("--model_name", type=str)
parser.add_argument("--generator_max_input_len", type=int, default=128000)
parser.add_argument("--api_type", type=str, default='openai', choices=['openai', 'azure', 'server'])
parser.add_argument("--api_version", type=str, default='')
parser.add_argument("--api_key", type=str, default='')
parser.add_argument("--azure_endpoint", type=str, default='')
parser.add_argument("--num_experiments", type=int, default=1)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--max_generate_tokens", type=int, default=512)
parser.add_argument("--save_dir", type=str, default='output')
parser.add_argument("--use_all_corpus", action="store_true")
parser.add_argument("--corpus_name", type=str)
parser.add_argument("--cot", action="store_true")
parser.add_argument("--metrics", nargs='+', type=str, default=["em", "f1", "acc", "retrieval_recall", "retrieval_precision"])
parser.add_argument("--base_url", type=str, default='')
args = parser.parse_args()

print(f'topk: {args.topk}')
print(f'save_dir: {args.save_dir}')
print(f'metrics: {args.metrics}')
if 'deepseek' in args.model_name.lower():
    args.temperature = 0.6
    args.max_generate_tokens = 10000
if 'o1' in args.model_name.lower():
    args.temperature = 1.
    args.max_generate_tokens = 10000
openai_dict = {"api_key": args.api_key}
if args.api_type == "azure":
    openai_dict["api_type"] = "azure"
    openai_dict["azure_endpoint"] = args.azure_endpoint
    openai_dict["api_version"] = args.api_version
if args.api_type == "server":
    openai_dict["base_url"] = args.base_url
    openai_dict["api_type"] = "server"
    openai_dict["api_key"] = None

generation_dict = {
    "max_tokens": args.max_generate_tokens,
    "temperature": args.temperature,
}

model_config_dict = {
        "model2path": {"LLM": args.model_path},
        "framework": args.framework,
        "generator_model": args.generator_model,
        "gpu_memory_utilization": args.gpu_util,
        "gpu_id": args.gpu_id,
        "generator_max_input_len": args.generator_max_input_len,
        "model_name": args.model_name,
        "openai_setting": openai_dict,
        "generation_params": generation_dict,
        "do_retrieval": args.do_retrieval,
}

model_config = Config(config_dict=model_config_dict)

generator = get_generator(model_config)

task_dict = {
    "mmlu_chem": ["test"],
    "scibench": ["atkins", "chemmc", "matter", "quan"],
    "mol-instruct": ["molecular_description_generation", "description_guided_molecule_design", "property_prediction", "forward_reaction_prediction", "reagent_prediction", "retrosynthesis"],
    "ChemBench4K": ["Caption2mol", "Mol2caption", "Name_Conversion", "Product_Prediction", "Retrosynthesis", "Solvent_Prediction", "Temperature_Prediction", "Yield_Prediction"]
}


system_prompts = {
    'naive': "Answer the question directly.\nOnly give me the answer and do not output any other words.",
    'cot': "Answer the question. Provide the answer between [ANSWER] and [/ANSWER]. Between [ANSWER] and [/ANSWER], you should only output the answer without any additional text. Think step by step.",
    'rag': "Answer the question based on the given document.\nOnly give me the answer and do not output any other words.\nThe following are given documents.\n\n{reference}",
}

'''scibench_system_prompts = {
    'naive': "Answer the question directly.\nOnly give me the answer and do not output any other words.",
    'rag': "Answer the question based on the given document.\nOnly give me the answer and do not output any other words.\nThe following are given documents.\n\n{reference}",
}'''

molecule_design_prompts = {
    'naive': "Answer the question directly.\nOnly give me the answer and do not output any other words. Your designed molecule should be a valid SMILES string.",
    'cot': "Answer the question. Your designed molecule should be a valid SMILES string surrounded by [ANSWER] and [/ANSWER]. Think step by step.",
    'rag': "Answer the question based on the given document.\nOnly give me the answer and do not output any other words. Your designed molecule should be a valid SMILES string.\nThe following are given documents.\n\n{reference}",
}

mol_instruct_prompts = {
    'naive': "Answer the question directly.\nWhen generating a molecule, please generate a valid SMILES string surrounded by [ANSWER] and [/ANSWER].",
    'cot': "Answer the question. When generating a molecule, please generate a valid SMILES string surrounded by [ANSWER] and [/ANSWER]. Think step by step.",
    'rag': "Answer the question based on the given document.\nWhen generating a molecule, please generate a valid SMILES string surrounded by [ANSWER] and [/ANSWER].\nThe following are given documents.\n\n{reference}",
}

description_generation_prompts = {
    'naive': "Answer the question directly.\nYour generated description should be surrounded by [ANSWER] and [/ANSWER].",
    'cot': "Answer the question. Your generated description should be surrounded by [ANSWER] and [/ANSWER]. Think step by step.",
    'rag': "Answer the question based on the given document.\nYour generated description should be surrounded by [ANSWER] and [/ANSWER].\nThe following are given documents.\n\n{reference}",
}

scibench_system_prompts = {
    'naive': "Answer the question directly.\n**Conclude the answer by stating `The answer is therefore [ANSWER]`**\nOnly give me the answer and do not output any other words.",
    'cot': "Answer the question. **Conclude the answer by stating `The answer is therefore [ANSWER]`**\nOnly give me the answer and do not output any other words. Think step by step.",
    'rag': "Answer the question based on the given document.\n**Conclude the answer by stating `The answer is therefore [ANSWER]`**\nOnly give me the answer and do not output any other words.\nThe following are given documents.\n\n{reference}",
}


user_prompts = {
    'open': "Question: {question}\nAnswer:\n",
    'choices': "Question: {question}\nChoices: {choices}\nMake prediction from the given choices.\nAnswer:\n",
}

eval_result_list = []
sub_datasets = task_dict[args.dataset_name]
for sub_dataset in sub_datasets:
    print(f'Running {sub_dataset}...')
    metrics = get_metrics(args.dataset_name, sub_dataset)
    #print(f'sub_dataset: {sub_dataset}, metrics: {metrics}')
    if args.corpus_name == "baseline":
        cache_dir = f"./retrieval_cache/{args.dataset_name}/textbook_{sub_dataset}_bm25_50.json"
    else:
        cache_dir = f"./retrieval_cache/{args.dataset_name}/{args.corpus_name}_{sub_dataset}_{args.retriever}_50.json"
    save_dir = os.path.join(args.save_dir, sub_dataset)
    os.makedirs(save_dir, exist_ok=True)
    config_dict = {
        "data_dir": args.data_path,
        "index_path": args.index_path,
        "corpus_path": args.corpus_path,
        "model2path": {"LLM": args.model_path},
        "framework": args.framework,
        "generator_model": args.generator_model,
        "retrieval_method": args.retriever,
        "split": sub_dataset,
        "metrics": metrics,
        "retrieval_topk": args.topk,
        "save_intermediate_data": True,
        "gpu_memory_utilization": args.gpu_util,
        "dataset_name": args.dataset_name,
        "gpu_id": args.gpu_id,
        "do_retrieval": args.do_retrieval,
        "generator_max_input_len": args.generator_max_input_len,
        "model_name": args.model_name,
        "openai_setting": openai_dict,
        "generation_params": generation_dict,
        "save_dir": save_dir,
        "corpus_name": args.corpus_name,
    }

    config = Config(config_dict=config_dict)
    # print(config)

    with open(cache_dir, "r") as f:
        data = json.load(f)
    test_data = Dataset(config, data=data)

    if args.cot and args.do_retrieval:
        raise ValueError("COT and do_retrieval cannot be used together.")

    system_key = 'cot' if args.cot else 'rag' if args.do_retrieval else 'naive'
    user_key = 'choices' if len(test_data[0].choices) > 0  else 'open'


    if args.dataset_name == 'scibench':
        prompt_templete = PromptTemplate(
            config,
            system_prompt=scibench_system_prompts[system_key],
            user_prompt=user_prompts[user_key],
        )
    elif sub_dataset == 'property_prediction':
        prompt_templete = PromptTemplate(
            config,
            system_prompt=scibench_system_prompts[system_key],
            user_prompt=user_prompts[user_key],
        )
    elif sub_dataset == 'description_guided_molecule_design':
        prompt_templete = PromptTemplate(
            config,
            system_prompt=molecule_design_prompts[system_key],
            user_prompt=user_prompts[user_key],
        )
    elif sub_dataset == 'molecular_description_generation':
        prompt_templete = PromptTemplate(
            config,
            system_prompt=description_generation_prompts[system_key],
            user_prompt=user_prompts[user_key],
        )
    elif sub_dataset == 'mol-instruct':
        prompt_templete = PromptTemplate(
            config,
            system_prompt=mol_instruct_prompts[system_key],
            user_prompt=user_prompts[user_key],
        )
    else:
        prompt_templete = PromptTemplate(
                config,
                system_prompt=system_prompts[system_key],
                user_prompt=user_prompts[user_key],
            )
        
    pipeline = SequentialPipeline(config, prompt_template=prompt_templete, generator=generator,
                                inference_only=True if args.cached_dataset_dir is not None else False,
                                retrieval_only=args.save_retrieval_only)
    if args.save_retrieval_only:
        if args.save_dataset_dir is None:
            raise ValueError("Please specify the save_dataset_dir if you want to save the retrieval results only.")
        print("Running retrieval only.")
        output_dataset = pipeline.run_retriver(test_data, do_eval=True)
    else:
        if args.do_retrieval:
            output_dataset, eval_result = pipeline.run(test_data, do_eval=True)
        else:
            output_dataset, eval_result = pipeline.naive_run(test_data, do_eval=True)
    eval_result_list.append(eval_result)

    if args.save_dataset_dir is not None:
        output_dataset.save(args.save_dataset_dir)

all_results = defaultdict(list)

for eval_result in eval_result_list:
    for metric, scores in eval_result.items():
        all_results[metric].append(scores)

mean_results = {metric: sum(scores) / len(scores) for metric, scores in all_results.items()}

print(f'mean_results: {mean_results}')

with open(os.path.join(args.save_dir, "mean_results.json"), "w") as f:
    json.dump(mean_results, f)
