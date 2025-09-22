# pocket_embed_new.py (回归事实的最终版)

import os
import torch
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

from Bio.PDB import PDBParser
from Bio.Data import PDBData

# 关键修正：我们只从最基础的模块导入ESM3
from esm.models.esm3 import ESM3

AA_3_TO_1 = PDBData.protein_letters_3to1

def get_sequence_and_res_keys_from_pdb(pdb_path):
    """使用Biopython从PDB文件中按顺序解析出标准的氨基酸序列和对应的(链ID, 残基序号)列表。"""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        sequence = ""
        residue_keys = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    res_name = residue.get_resname()
                    if res_name in AA_3_TO_1:
                        sequence += AA_3_TO_1[res_name]
                        res_id = residue.get_id()[1]
                        chain_id = chain.get_id()
                        residue_keys.append((chain_id, res_id))
        return sequence, residue_keys
    except Exception as e:
        print(f"\n[!] 使用Biopython解析PDB {pdb_path} 时出错: {e}")
        return None, None

def process_protein_by_sequence(sequence, res_keys, model, tokenizer, device):
    """通过手动tokenization处理蛋白质序列，返回核心embedding和残基ID列表。"""
    try:
        # 使用从模型实例中获取的分词器
        token_ids = tokenizer.batch_encode_plus([("", sequence)], return_tensors="pt")["input_ids"].to(device)

        # 将Token ID直接喂给模型，并指定提取最后一层的表征
        with torch.no_grad():
            output = model(token_ids, repr_layers=[12])
        
        full_embedding = output["representations"][12].squeeze(0).cpu()
        core_embedding = full_embedding[1:-1, :]

        if core_embedding.shape[0] != len(res_keys):
             print(f"\n[!] FATAL WARNING: 长度不匹配！Emb: {core_embedding.shape[0]}, Keys: {len(res_keys)}. File Path: {res_keys}")
             return None, None
             
        return core_embedding, res_keys

    except Exception as e:
        print(f"\n[!] Tokenization或模型推理时出错: {e}")
        return None, None

def main(args):
    # 路径和设备设置
    pocket_data_root = Path(args.dataset_root)
    full_protein_root = Path(args.full_protein_root)
    index_path = pocket_data_root / "index.pkl"
    device = torch.device("cuda")
    shard_info = f"[分片 {args.shard_id}/{args.num_shards}]"
    print(f"{shard_info} -> 口袋数据根目录: {pocket_data_root}")
    print(f"{shard_info} -> 完整蛋白根目录: {full_protein_root}")

    # --- 最终修正：回归到从模型实例获取分词器 ---
    print(f"{shard_info} 正在加载 ESM-3 模型和分词器...")
    model = ESM3.from_pretrained("esm3-sm-open-v1").to(device).eval()
    tokenizer = model.tokenizer # 这是被验证过的、在您环境中有效的方法
    print(f"{shard_info} 模型和分词器加载成功。")
    # --- 修正结束 ---

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
        
        sequence, full_res_keys = get_sequence_and_res_keys_from_pdb(str(full_protein_path))
        if not sequence:
            continue
            
        core_embedding, _ = process_protein_by_sequence(sequence, full_res_keys, model, tokenizer, device)
        
        if core_embedding is None:
            continue

        res_key_to_idx_map = {res_key: i for i, res_key in enumerate(full_res_keys)}
        
        for pocket_fn in pocket_fns:
            output_path = (pocket_data_root / pocket_fn).with_suffix('.pt')
            if output_path.exists():
                continue
            
            # 使用同样可靠的Biopython解析器获取口袋的残基ID
            _, pocket_res_keys = get_sequence_and_res_keys_from_pdb(pocket_data_root / pocket_fn)
            if not pocket_res_keys:
                continue

            indices_to_select = [res_key_to_idx_map[res_key] for res_key in pocket_res_keys if res_key in res_key_to_idx_map]
            
            if not indices_to_select:
                continue

            pocket_embedding = core_embedding[indices_to_select, :]
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pocket_embedding, output_path)

    print(f"{shard_info} 所有任务已成功处理完毕。")

if __name__ == "__main__":
    # 参数解析部分保持不变
    parser = argparse.ArgumentParser(description="为CrossDocked数据集提取蛋白质embeddings (最终序列版)。")
    parser.add_argument("--dataset_root", type=str, required=True, help="指向 'crossdocked_pocket' 数据集的根目录")
    parser.add_argument("--full_protein_root", type=str, required=True, help="指向包含完整蛋白质的 'crossdocked_v1.1_rmsd1.0' 目录")
    parser.add_argument("--num-shards", type=int, default=1, help="并行进程的总数")
    parser.add_argument("--shard-id", type=int, default=0, help="当前进程的ID (0-indexed)")
    args = parser.parse_args()
    main(args)