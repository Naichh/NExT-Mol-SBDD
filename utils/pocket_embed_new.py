# preprocess_embeddings.py (方案二: Sharding版本)

import os
import torch
import io
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig

def process_pocket(pdb_path, model, device):
    # 这个函数和我们修复后的版本完全一样
    try:
        with open(pdb_path, 'r', errors='ignore') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"\n[!] WARNING (Shard {os.getenv('SHARD_ID', 0)}): File not found, skipping: {pdb_path}")
        return None
    lines = [line for line in lines if not line.startswith(('MODEL', 'ENDMDL'))]
    pdb_string_io = io.StringIO()
    pdb_string_io.write("MODEL        1\n")
    pdb_string_io.writelines(lines)
    pdb_string_io.write("ENDMDL\n")
    pdb_string_io.seek(0)
    protein = ESMProtein.from_pdb(pdb_string_io)
    protein_tensor = model.encode(protein).to(device)
    embedding_config = LogitsConfig(return_embeddings=True)
    with torch.no_grad():
        output = model.logits(protein_tensor, embedding_config)
    full_embedding = output.embeddings.squeeze(0).cpu()
    return full_embedding

def main(args):
    # 关键改动：根据分片ID设置设备
    # 确保每个进程使用不同的GPU
    device = torch.device("cuda")    
    print(f"[*] [分片 {args.shard_id}/{args.num_shards}] 正在加载 ESM-3 模型到 {device}...")
    model = ESM3.from_pretrained("esm3-sm-open-v1").to(device).eval()
    print(f"[*] [分片 {args.shard_id}] 模型加载成功。")

    dataset_root = Path(args.dataset_root)
    index_path = dataset_root / 'index.pkl'
    if not index_path.exists():
        raise FileNotFoundError(f"未找到主索引文件: {index_path}")

    with open(index_path, 'rb') as f:
        master_index = pickle.load(f)

    pocket_files_relative = sorted(list(set([item[0] for item in master_index if item[0] is not None])))
    print(f"[*] [分片 {args.shard_id}] 总共发现 {len(pocket_files_relative)} 个独立口袋。")

    # 关键改动：根据分片ID分割任务列表
    # 每个进程只处理属于自己的那一部分文件
    files_for_this_shard = [fn for i, fn in enumerate(pocket_files_relative) if i % args.num_shards == args.shard_id]
    
    # 过滤掉已经处理过的文件
    job_list = [fn for fn in files_for_this_shard if not (dataset_root / fn).with_suffix('.pt').exists()]
    print(f"[*] [分片 {args.shard_id}] 分配到 {len(files_for_this_shard)} 个任务, 其中 {len(job_list)} 个需要处理。")

    # 每个分片拥有自己的进度条
    for pocket_fn in tqdm(job_list, desc=f"分片 {args.shard_id} 正在处理", position=args.shard_id):
        pdb_path = dataset_root / pocket_fn
        output_path = (dataset_root / pocket_fn).with_suffix('.pt')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            embedding = process_pocket(str(pdb_path), model, device)
            if embedding is not None:
                torch.save(embedding, output_path)
        except Exception as e:
            error_msg = f"{pdb_path}\tSHARD_{args.shard_id}\t{e}\n"
            print(f"\n[!] [分片 {args.shard_id}] 处理 {pdb_path.name} 时发生错误: {e}")
            # 加锁以安全地写入同一个日志文件
            with open(dataset_root / "embedding_errors.log", "a") as error_log:
                error_log.write(error_msg)

    print(f"[*] [分片 {args.shard_id}] 已成功处理完毕。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为CrossDocked数据集提取embeddings (Sharding版本)。")
    parser.add_argument("--dataset_root", type=str, required=True, help="数据集的根目录")
    # device参数不再需要，由shard_id决定
    parser.add_argument("--num-shards", type=int, default=1, help="并行进程的总数")
    parser.add_argument("--shard-id", type=int, default=0, help="当前进程的ID (0-indexed)")
    args = parser.parse_args()
    main(args)