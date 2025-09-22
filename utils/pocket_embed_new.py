# pocket_embed_new.py (采纳您的建议，最终简洁稳健版)

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

def get_residue_ids_from_pdb(pdb_path):
    """
    从PDB文件中按顺序解析并返回一个唯一的(链ID, 残基序号)元组列表。
    这个简单、可靠的解析器是我们的“唯一真理”。
    """
    try:
        with open(pdb_path, 'r', errors='ignore') as f:
            lines = f.readlines()
        
        residue_keys = []
        seen_keys = set()
        for line in lines:
            if line.startswith("ATOM"):
                chain_id = line[21].strip()
                res_id = int(line[22:26].strip())
                key = (chain_id, res_id)
                if key not in seen_keys:
                    seen_keys.add(key)
                    residue_keys.append(key)
        return residue_keys
    except Exception as e:
        print(f"\n[!] 解析PDB文件 {pdb_path} 时出错: {e}")
        return None

def process_full_protein(protein_pdb_path, model, device):
    """
    处理一个完整的蛋白质PDB文件，返回切片后的核心embedding和从PDB直接解析的残基ID列表。
    """
    try:
        # 步骤 1: 我们自己从PDB文件解析出物理残基ID列表
        full_protein_res_keys = get_residue_ids_from_pdb(protein_pdb_path)
        if not full_protein_res_keys:
            return None, None

        # 步骤 2: 使用ESM SDK加载PDB并生成完整的embedding
        protein = ESMProtein.from_pdb(protein_pdb_path)
        protein_tensor = model.encode(protein).to(device)
        embedding_config = LogitsConfig(return_embeddings=True)
        with torch.no_grad():
            output = model.logits(protein_tensor, embedding_config)
        full_embedding = output.embeddings.squeeze(0).cpu()
        
        # --- 关键修正：根据您的建议，我们假设首尾是特殊字符并进行切片 ---
        if full_embedding.shape[0] == len(full_protein_res_keys) + 2:
            core_embedding = full_embedding[1:-1, :]
        elif full_embedding.shape[0] == len(full_protein_res_keys):
            # 如果长度恰好相等，也接受
            core_embedding = full_embedding
        else:
            # 如果长度不匹配且不是+2的关系，则这是一个真正的错误
            print(f"\n[!] WARNING: Embedding长度 ({full_embedding.shape[0]}) 与残基数 ({len(full_protein_res_keys)}) 的关系无法处理！ 文件: {protein_pdb_path}")
            return None, None
        
        return core_embedding, full_protein_res_keys

    except Exception as e:
        print(f"\n[!] 处理完整蛋白 {protein_pdb_path} 时出错: {e}")
        return None, None


def main(args):
    # 这部分代码与上一版相同，无需修改
    pocket_data_root = Path(args.dataset_root)
    full_protein_root = Path(args.full_protein_root)
    index_path = pocket_data_root / "index.pkl"
    device = torch.device("cuda")
    shard_info = f"[分片 {args.shard_id}/{args.num_shards}]"
    print(f"{shard_info} -> 口袋数据根目录: {pocket_data_root}")
    print(f"{shard_info} -> 完整蛋白根目录: {full_protein_root}")

    print(f"{shard_info} 正在加载 ESM-3 模型到 {device}...")
    model = ESM3.from_pretrained("esm3-sm-open-v1").to(device).eval()
    print(f"{shard_info} 模型加载成功。")

    if not index_path.exists():
        raise FileNotFoundError(f"未找到主索引文件: {index_path}")
    with open(index_path, 'rb') as f:
        master_index = pickle.load(f)

    pocket_to_protein_map = {item[0]: item[2] for item in master_index if item[0] is not None and item[2] is not None}
    all_pockets = sorted(list(pocket_to_protein_map.keys()))
    pockets_for_this_shard = [p for i, p in enumerate(all_pockets) if i % args.num_shards == args.shard_id]
    
    protein_to_pockets_map = {}
    for pocket_fn in pockets_for_this_shard:
        protein_fn = pocket_to_protein_map.get(pocket_fn)
        if protein_fn:
            if protein_fn not in protein_to_pockets_map:
                protein_to_pockets_map[protein_fn] = []
            protein_to_pockets_map[protein_fn].append(pocket_fn)
            
    print(f"{shard_info} 总共分配到 {len(pockets_for_this_shard)} 个口袋，涉及 {len(protein_to_pockets_map)} 个独立蛋白质。")

    for protein_fn, pocket_fns in tqdm(protein_to_pockets_map.items(), desc=f"{shard_info} 处理中", position=args.shard_id):
        all_done = all([(pocket_data_root / fn).with_suffix('.pt').exists() for fn in pocket_fns])
        if all_done:
            continue
            
        full_protein_path = full_protein_root / protein_fn
        # 注意这里接收的是切片后的 core_embedding
        core_embedding, full_res_keys = process_full_protein(str(full_protein_path), model, device)
        
        if core_embedding is None:
            continue

        res_key_to_idx_map = {res_key: i for i, res_key in enumerate(full_res_keys)}
        
        for pocket_fn in pocket_fns:
            output_path = (pocket_data_root / pocket_fn).with_suffix('.pt')
            if output_path.exists():
                continue

            # 依然使用我们自己可靠的解析器
            pocket_res_keys = get_residue_ids_from_pdb(pocket_data_root / pocket_fn)
            if pocket_res_keys is None:
                continue

            indices_to_select = [res_key_to_idx_map[res_key] for res_key in pocket_res_keys if res_key in res_key_to_idx_map]
            
            if not indices_to_select:
                print(f"\n{shard_info} WARNING: 对于口袋 {pocket_fn}, 未在完整蛋白中找到任何对应残基。")
                continue

            pocket_embedding = core_embedding[indices_to_select, :]
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pocket_embedding, output_path)

    print(f"{shard_info} 所有任务已成功处理完毕。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为CrossDocked数据集提取蛋白质embeddings (最终并行版)。")
    parser.add_argument("--dataset_root", type=str, required=True, help="指向 'crossdocked_pocket' 数据集的根目录")
    parser.add_argument("--full_protein_root", type=str, required=True, help="指向包含完整蛋白质的 'crossdocked_v1.1_rmsd1.0' 目录")
    parser.add_argument("--num-shards", type=int, default=1, help="并行进程的总数")
    parser.add_argument("--shard-id", type=int, default=0, help="当前进程的ID (0-indexed)")
    args = parser.parse_args()
    main(args)