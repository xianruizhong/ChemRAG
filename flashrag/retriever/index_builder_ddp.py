# ddp_encode_worker.py
import os, json, torch, numpy as np, torch.distributed as dist
from torch.multiprocessing import spawn
from encoder import Encoder                          # your class

def worker(rank: int, world_size: int, cfg: dict, corpus: list, out_file: str):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    enc = Encoder(**cfg)
    enc.model.to(rank)
    enc.model = torch.nn.parallel.DistributedDataParallel(
        enc.model, device_ids=[rank], output_device=rank
    )                                                # optional for inference

    # shard the queries ---------------------------------
    local_queries = corpus[rank::world_size]
    local_emb = enc.encode(local_queries,
                           batch_size=cfg["batch_size"],
                           is_query=False)           # (m_i, d) numpy

    # ---------------------------------------------------
    # all-gather every shard to rank-0
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_emb)

    if rank == 0:
        full = np.concatenate(gathered, axis=0)
        np.save(out_file, full)                      # or return via queue

    dist.barrier()
    dist.destroy_process_group()

def launch(cfg, corpus, out_file="embeddings.npy"):
    world_size = torch.cuda.device_count()
    spawn(worker, args=(world_size, cfg, corpus, out_file),
          nprocs=world_size, join=True)
