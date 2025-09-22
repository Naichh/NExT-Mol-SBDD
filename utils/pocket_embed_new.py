# pocket_embed_new.py (使用模型并行修正版)

import os
import torch
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
import warnings

# 关键修改 #1: 导入 accelerate 的相关工具
from accelerate import dispatch_model, infer_auto_device_map

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

def process_protein_by_sequence(sequence, res_keys, model, tokenizer):
    """通过官方示例展示的、最直接的方式处理蛋白质序列。"""
    try:
        # 步骤 1: 使用分词器将序列字符串转换为Token ID
        tokens = tokenizer.encode(sequence)
        
        # 关键修改 #2: 将输入张量放到模型所在的第一个设备上 (通常是 'cuda:0')
        # 模型并行会自动处理后续在不同GPU间的张量传递
        token_ids = torch.tensor(tokens, dtype=torch.int64).to(next(model.parameters()).device).unsqueeze(0)

        # 步骤 2: 将Token ID直接喂给模型
        with torch.no_grad():
            output = model.forward(sequence_tokens=token_ids)
        
        # 步骤 3: 从ESMOutput对象中提取embedding
        full_embedding = output.embeddings.squeeze(0).cpu()

        # 步骤 4: 移除BOS/EOS token对应的embedding
        core_embedding = full_embedding[1:-1, :]

        # 步骤 5: 验证长度
        if core_embedding.shape[0] != len(res_keys):
             print(f"\n[!] FATAL WARNING: 长度不匹配！Emb: {core_embedding.shape[0]}, Keys: {len(res_keys)}.")
             return None, None
             
        return core_embedding, res_keys

    except Exception as e:
        print(f"\n[!] Tokenization或模型推理时出错: {e}")
        return None, None

def main(args):
    # 路径设置
    pocket_data_root = Path(args.dataset_root)
    full_protein_root = Path(args.full_protein_root)
    index_path = pocket_data_root / "index.pkl"
    
    # 关键修改 #3: 移除手动的设备和分片设置
    # device = torch.device("cuda")  <-- 删除这一行
    # shard_info = f"[分片 {args.shard_id}/{args.num_shards}]" <-- 不再需要分片
    print(f"-> 口袋数据根目录: {pocket_data_root}")
    print(f"-> 完整蛋白根目录: {full_protein_root}")

    # --- 模型加载修改 ---
    print(f"正在加载 ESM-3 模型和分词器...")
    # 1. 先在CPU上加载完整模型结构，避免在单个GPU上加载时就OOM
    model = ESM3_sm_open_v0().to('cpu') 
    
    # 2. 推断如何自动切分模型到所有可用的GPU上
    #    您可以根据 `nvidia-smi` 的情况调整 max_memory，留一些余量
    device_map = infer_auto_device_map(model, max_memory={i: "20GiB" for i in range(torch.cuda.device_count())}, no_split_module_classes=[])
    print("自动推断的设备分配图 (Device Map):")
    print(device_map)

    # 3. 根据分配图，将模型的不同部分部署到不同GPU上
    model = dispatch_model(model, device_map=device_map)
    model.eval()
    
    tokenizers = model.tokenizers      
    sequence_tokenizer = tokenizers.sequence 
    print(f"模型已成功以模型并行模式加载到 {torch.cuda.device_count()} 个GPU上。")
    # --- 模型加载修改结束 ---

    with open(index_path, 'rb') as f:
        master_index = pickle.load(f)

    pocket_to_protein_map = {item[0]: item[2] for item in master_index if item[0] is not None and item[2] is not None}
    
    # 关键修改 #4: 不再需要根据分片ID过滤数据，处理所有口袋
    all_pockets = sorted(list(pocket_to_protein_map.keys()))
    # pockets_for_this_shard = [p for i, p in enumerate(all_pockets) if i % args.num_shards == args.shard_id] <-- 删除
    
    protein_to_pockets_map = {}
    for pocket_fn in all_pockets: # <-- 使用 all_pockets
        protein_fn = pocket_to_protein_map.get(pocket_fn)
        if protein_fn:
            if protein_fn not in protein_to_pockets_map:
                protein_to_pockets_map[protein_fn] = []
            protein_to_pockets_map[protein_fn].append(pocket_fn)
            
    print(f"总共需要处理 {len(all_pockets)} 个口袋，涉及 {len(protein_to_pockets_map)} 个独立蛋白质。")

    # 关键修改 #5: tqdm的描述和position不再需要分片信息
    for protein_fn, pocket_fns in tqdm(protein_to_pockets_map.items(), desc="处理中"):
        all_done = all([(pocket_data_root / fn).with_suffix('.pt').exists() for fn in pocket_fns])
        if all_done:
            continue
            
        full_protein_path = full_protein_root / protein_fn
        
        sequence, full_res_keys = get_sequence_and_res_keys_from_pdb(str(full_protein_path))
        if not sequence:
            continue
            
        # 关键修改 #6: 不再传递 device 参数
        core_embedding, _ = process_protein_by_sequence(sequence, full_res_keys, model, sequence_tokenizer)
        
        if core_embedding is None:
            continue

        res_key_to_idx_map = {res_key: i for i, res_key in enumerate(full_res_keys)}
        
        for pocket_fn in pocket_fns:
            output_path = (pocket_data_root / pocket_fn).with_suffix('.pt')
            if output_path.exists():
                continue
            
            _, pocket_res_keys = get_sequence_and_res_keys_from_pdb(pocket_data_root / pocket_fn)
            if not pocket_res_keys:
                continue

            indices_to_select = [res_key_to_idx_map[res_key] for res_key in pocket_res_keys if res_key in res_key_to_idx_map]
            
            if not indices_to_select:
                continue

            pocket_embedding = core_embedding[indices_to_select, :]
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pocket_embedding, output_path)

    print("所有任务已成功处理完毕。")

if __name__ == "__main__":
    # 关键修改 #7: 移除并行相关的参数
    parser = argparse.ArgumentParser(description="为CrossDocked数据集提取蛋白质embeddings (模型并行版)。")
    parser.add_argument("--dataset_root", type=str, required=True, help="指向 'crossdocked_pocket' 数据集的根目录")
    parser.add_argument("--full_protein_root", type=str, required=True, help="指向包含完整蛋白质的 'crossdocked_v1.1_rmsd1.0' 目录")
    # parser.add_argument("--num-shards", type=int, default=1, help="并行进程的总数") <-- 删除
    # parser.add_argument("--shard-id", type=int, default=0, help="当前进程的ID (0-indexed)") <-- 删除
    args = parser.parse_args()
    main(args)