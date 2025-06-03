import argparse
import faiss
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np
import os

from flashrag.retriever.utils import load_model, load_corpus
from flashrag.retriever.encoder import EncoderDDP

def save_embedding(all_embeddings, embedding_save_path):
    memmap = np.memmap(embedding_save_path, shape=all_embeddings.shape, mode="w+", dtype=all_embeddings.dtype)
    length = all_embeddings.shape[0]
    # add in batch
    save_batch_size = 10000
    if length > save_batch_size:
        for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
            j = min(i + save_batch_size, length)
            memmap[i:j] = all_embeddings[i:j]
    else:
        memmap[:] = all_embeddings

def save_faiss_index(
        all_embeddings,
        faiss_type,
        index_save_path,
        faiss_gpu=False,
    ):
    # build index
    print("Creating index")
    dim = all_embeddings.shape[-1]
    faiss_index = faiss.index_factory(dim, faiss_type, faiss.METRIC_INNER_PRODUCT)

    if faiss_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        co.shard = True
        faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
        if not faiss_index.is_trained:
            faiss_index.train(all_embeddings)
        faiss_index.add(all_embeddings)
        faiss_index = faiss.index_gpu_to_cpu(faiss_index)
    else:
        if not faiss_index.is_trained:
            faiss_index.train(all_embeddings)
        faiss_index.add(all_embeddings)

    faiss.write_index(faiss_index, index_save_path)

def main():
    """
    Entry point for launching multi-GPU inference via torchrun.
    """
    parser = argparse.ArgumentParser(description="Creating embeddings")
    # Basic parameters
    parser.add_argument("--retrieval_method", type=str)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--corpus_path", type=str)
    parser.add_argument("--save_dir", default="embeddings/", type=str)

    # Parameters for building dense index
    parser.add_argument("--max_length", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--use_fp16", default=False, action="store_true")
    parser.add_argument("--pooling_method", type=str, default=None)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--faiss_type", type=str, default=None)
    parser.add_argument("--embedding_path", default=None, type=str)
    parser.add_argument("--faiss_gpu", default=False, action="store_true")
    parser.add_argument("--sentence_transformer", action="store_true", default=False)

    # Parameters for build multi-modal retriever index
    parser.add_argument("--index_modal", type=str, default="all", choices=["text", "image", "all"])
    
    args = parser.parse_args()

    # 1. Get rank/world_size from environment (set by torchrun)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])  # if needed

    # 2. Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )

    # 3. Set GPU device
    torch.cuda.set_device(local_rank)

    # 4. Make sure the directory exists (rank 0 does it)
    if rank == 0 and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 5. Load corpus / queries
    corpus = load_corpus(args.corpus_path)
    queries = [item["contents"] for item in tqdm(corpus, desc="Loading corpus for encoding")]

    # 6. Load model & tokenizer
    #    (each rank loads the same weights)
    model, tokenizer = load_model(
        model_path=args.model_path,
        use_fp16=args.use_fp16
    )
    model.cuda(local_rank)
    model.eval()

    # 7. Wrap model in DDP
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 8. Build your Encoder
    encoder = EncoderDDP(
        model_name=args.retrieval_method,
        tokenizer=tokenizer,
        model=ddp_model,
        pooling_method=args.pooling_method,
        max_length=args.max_length,
        instruction=args.instruction,
    )

    # 9. DistributedSampler to split queries among ranks
    sampler = DistributedSampler(
        queries,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # 10. Encode locally
    local_embs = encoder.encode(
        query_list=queries,
        batch_size=args.batch_size,
        is_query=False,
        distributed_sampler=sampler,
        rank=rank,
    )

    # 11. Gather
    local_embs = torch.from_numpy(local_embs).cuda(local_rank)

    local_size = torch.LongTensor([local_embs.shape[0]]).cuda(local_rank)
    gather_sizes = [torch.LongTensor([0]).cuda(local_rank) for _ in range(world_size)]
    dist.all_gather(gather_sizes, local_size)

    max_size = torch.max(torch.stack(gather_sizes)).item()
    padded_local = torch.zeros(max_size, local_embs.shape[1], device=local_rank, dtype=local_embs.dtype)
    padded_local[:local_embs.shape[0]] = local_embs

    gather_list = [
        torch.zeros(max_size, local_embs.shape[1], device=local_rank, dtype=local_embs.dtype)
        for _ in range(world_size)
    ]
    dist.all_gather(gather_list, padded_local)

    # Convert each rank's chunk
    all_embs_list = []
    for i, gathered in enumerate(gather_list):
        true_size = gather_sizes[i].item()
        all_embs_list.append(gathered[:true_size].cpu().numpy())
    all_embs = np.concatenate(all_embs_list, axis=0)

    # 12. Save on rank 0
    if rank == 0:
        all_embs = all_embs.astype(np.float32, order="C")
        index_save_path = os.path.join(args.save_dir, f"{args.retrieval_method}_{args.faiss_type}.index")
        save_faiss_index(all_embs, args.faiss_type, index_save_path, args.faiss_gpu)
        print(f"[Rank 0] Saved index to {index_save_path}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
