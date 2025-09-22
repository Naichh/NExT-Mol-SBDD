# pocket_embed_new.py (最终的、生产级的并行脚本)

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

from esm.pretrained import ESM3_sm_open_v0

AA_3_TO_1 = PDBData.protein_letters_3to1

def get_sequence_and_res_keys_from_pdb(pdb_path):
    """使用Biopython从PDB文件中按顺序解析出标准的氨基酸序列和对应的(链ID, 残基序号)列表。"""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        sequence, residue_keys = "", []
        for model in structure:
            for chain in model:
                for residue in chain:
                    res_name = residue.get_resname()
                    if res_name in AA_3_TO_1:
                        sequence += AA_3_TO_1[res_name]
                        residue_keys.append((chain.get_id(), residue.get_id()[1]))
        return sequence, residue_keys
    except Exception as e:
        print(f"\n[!] 解析PDB {pdb_path.name} 出错: {e}")
        return None, None

def process_protein(sequence, res_keys, model, tokenizer, device, max_len=1000, overlap=100):
    """通过智能分块处理蛋白质序列，保证不OOM。"""
    try:
        if len(sequence) <= max_len:
            tokens = tokenizer.encode(sequence)
            token_ids = torch.tensor(tokens, dtype=torch.int64).to(device).unsqueeze(0)
            with torch.no_grad():
                output = model.forward(sequence_tokens=token_ids)
            core_embedding = output.embeddings.squeeze(0).cpu()[1:-1, :]
        else:
            embedding_chunks = []
            start = 0
            while start < len(sequence):
                end = start + max_len
                chunk_seq = sequence[start:end]
                tokens = tokenizer.encode(chunk_seq)
                token_ids = torch.tensor(tokens, dtype=torch.int64).to(device).unsqueeze(0)
                with torch.no_grad():
                    output = model(sequence_tokens=token_ids)
                
                chunk_embedding = output.embeddings.squeeze(0).cpu()[1:-1, :]
                slice_start = overlap // 2 if start > 0 else 0
                embedding_chunks.append(chunk_embedding[slice_start:])
                start += (max_len - overlap)
                del token_ids, output, chunk_embedding
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            core_embedding = torch.cat(embedding_chunks, dim=0)[:len(sequence)]

        if core_embedding.shape[0] != len(res_keys):
             return None
        return core_embedding
    except Exception as e:
        print(f"\n[!] 推理时出错: {e}, 序列长度: {len(sequence)}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None

def main():
    parser = argparse.ArgumentParser(description="为CrossDocked数据集提取蛋白质embeddings (最终并行版)。")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--full_protein_root", type=str, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    args = parser.parse_args()

    # 每个进程使用由启动脚本指定的、唯一的GPU
    device = torch.device("cuda")
    shard_info = f"[分片 {args.shard_id}/{args.num_shards}]"
    
    # 每个进程在自己的GPU上加载一个完整的模型
    print(f"{shard_info} 正在加载 ESM-3 模型到 {device}...")
    model = ESM3_sm_open_v0(device)
    model.eval()
    sequence_tokenizer = model.tokenizers.sequence
    print(f"{shard_info} 模型加载成功。")

    pocket_data_root = Path(args.dataset_root)
    full_protein_root = Path(args.full_protein_root)
    index_path = pocket_data_root / "index.pkl"

    with open(index_path, 'rb') as f:
        master_index = pickle.load(f)

    pocket_to_protein_map = {item[0]: item[2] for item in master_index if item[0] is not None and item[2] is not None}
    all_proteins = sorted(list(set(pocket_to_protein_map.values())))
    
    proteins_for_this_process = all_proteins[args.shard_id::args.num_shards]
    
    print(f"{shard_info} 分配到 {len(proteins_for_this_process)} 个蛋白质进行处理。")

    protein_to_pockets_map = {}
    for pocket, protein in pocket_to_protein_map.items():
        if protein not in protein_to_pockets_map:
            protein_to_pockets_map[protein] = []
        protein_to_pockets_map[protein].append(pocket)

    for protein_fn in tqdm(proteins_for_this_process, desc=shard_info, position=args.shard_id):
        pocket_fns = protein_to_pockets_map.get(protein_fn, [])
        if all( (pocket_data_root / fn).with_suffix('.pt').exists() for fn in pocket_fns ):
            continue
            
        full_protein_path = full_protein_root / protein_fn
        sequence, full_res_keys = get_sequence_and_res_keys_from_pdb(str(full_protein_path))
        if not sequence: continue
        
        core_embedding = process_protein(sequence, full_res_keys, model, sequence_tokenizer, device)
        if core_embedding is None: continue

        res_key_to_idx_map = {res_key: i for i, res_key in enumerate(full_res_keys)}
        
        for pocket_fn in pocket_fns:
            output_path = (pocket_data_root / pocket_fn).with_suffix('.pt')
            if output_path.exists(): continue
            
            _, pocket_res_keys = get_sequence_and_res_keys_from_pdb(pocket_data_root / pocket_fn)
            if not pocket_res_keys: continue

            indices_to_select = [res_key_to_idx_map.get(res_key) for res_key in pocket_res_keys]
            indices_to_select = [idx for idx in indices_to_select if idx is not None]

            if not indices_to_select: continue
            
            pocket_embedding = core_embedding[indices_to_select, :]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pocket_embedding, output_path)

    print(f"{shard_info} 所有任务已成功处理完毕。")

if __name__ == "__main__":
    main()