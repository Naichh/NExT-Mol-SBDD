# test_dist.py
import os
import torch
import torch.distributed as dist
import time

def run():
    # 这个print应该能立即显示
    print(f"--- [PID:{os.getpid()}] Script started. ---", flush=True)

    # 初始化进程组，这是分布式通信的核心
    print(f"--- [PID:{os.getpid()}] Initializing process group... ---", flush=True)
    dist.init_process_group("nccl")
    print(f"--- [PID:{os.getpid()}] Process group initialized. ---", flush=True)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Hello from Rank {rank} of {world_size}! Distributed environment seems OK.", flush=True)

    # 创建一个张量并进行一次简单的通信（广播）
    tensor = torch.arange(2, device=rank) + 1 + 2 * rank
    print(f"Rank {rank} has initial tensor {tensor}", flush=True)

    # Rank 0 将它的张量广播给所有其他进程
    dist.broadcast(tensor, src=0)
    time.sleep(1) # 等待一下以确保打印顺序

    print(f"Rank {rank} has tensor {tensor} after broadcast.", flush=True)

    # 销毁进程组
    dist.destroy_process_group()
    print(f"Rank {rank} finished successfully.", flush=True)

if __name__ == "__main__":
    run()