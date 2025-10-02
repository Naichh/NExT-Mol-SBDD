import os
import pickle
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import random
from collections import Counter, defaultdict
import warnings
from multiprocessing import Pool, Manager
import time

# --- 使用 esm 的底层库 Biotite 进行精确、一致的解析 ---
from biotite.structure.io.pdb import PDBFile
import biotite.structure as bs
from esm.utils.residue_constants import restype_3to1 as ESM_AA_3_TO_1

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 工作函数：这是每个并行进程要执行的核心任务 ---
def process_single_pocket(args):
    """
    处理单个口袋文件的函数，设计为在独立的进程中运行。
    """
    pocket_fn, dataset_root, device = args

    # 初始化该文件对应的统计数据
    stats = {
        'embedding_found': 0, 'embedding_missing': 0, 'unreadable_pt': 0,
        'nan_files': 0, 'inf_files': 0, 'zero_embeddings': 0,
        'shape_mismatch': 0, 'single_chain_pockets': 0, 'multi_chain_pockets': 0,
        'dimensions': Counter(), 'pocket_size': None, 'embedding_stats': None,
        'mismatch_example': None, 'missing_example': None
    }

    pt_path = (Path(dataset_root) / pocket_fn).with_suffix('.pt')

    if not pt_path.exists():
        stats['embedding_missing'] = 1
        stats['missing_example'] = pocket_fn
        return stats

    stats['embedding_found'] = 1
    pdb_path = pt_path.with_suffix('.pdb')

    # 1. PDB文件解析 (CPU任务)
    try:
        pdb_file = PDBFile.read(str(pdb_path))
        structure = pdb_file.get_structure(model=1)
        protein_atoms = structure[bs.filter_amino_acids(structure)]
        if protein_atoms.array_length() == 0:
            stats['unreadable_pt'] = 1
            return stats

        chain_ids = {res.chain_id[0] for res in bs.residue_iter(protein_atoms)}
        n_pdb = len(list(bs.residue_iter(protein_atoms)))

        if len(chain_ids) == 1:
            stats['single_chain_pockets'] = 1
        else:
            stats['multi_chain_pockets'] = 1
        stats['pocket_size'] = n_pdb
    except Exception:
        stats['unreadable_pt'] = 1
        return stats

    # 2. Embedding加载和分析 (GPU任务)
    try:
        # 将张量直接加载到指定的GPU设备
        embedding = torch.load(pt_path, map_location=device, weights_only=True)
        if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2 or embedding.shape[0] == 0:
            stats['unreadable_pt'] = 1
            return stats

        if torch.isnan(embedding).any(): stats['nan_files'] = 1
        if torch.isinf(embedding).any(): stats['inf_files'] = 1
        if torch.all(embedding == 0): stats['zero_embeddings'] = 1

        n_embed, d_embed = embedding.shape
        stats['dimensions'][d_embed] = 1

        if n_embed != n_pdb:
            stats['shape_mismatch'] = 1
            stats['mismatch_example'] = f"{pocket_fn} (Emb: {n_embed}, PDB: {n_pdb})"

        # 在GPU上计算统计值，然后转到CPU，避免GPU内存积累
        stats['embedding_stats'] = {
            'mean': embedding.mean().item(),
            'std': embedding.std().item(),
            'min': embedding.min().item(),
            'max': embedding.max().item()
        }
    except Exception:
        stats['unreadable_pt'] = 1

    return stats


def run_qa_on_set_parallel(pocket_fns: list, set_name: str, dataset_root: str, num_workers: int):
    """
    【已修改】使用多进程并行处理质量分析。
    """
    print("\n" + "#"*80)
    print(f"# 分析目标: {set_name} (应有 {len(pocket_fns)} 个口袋)")
    print(f"# 使用 {num_workers} 个CPU核心 和 {torch.cuda.device_count()} 张GPU并行处理...")
    print("#"*80)

    # --- 准备给每个进程的参数 ---
    num_gpus = torch.cuda.device_count()
    # 为每个任务分配一个GPU设备ID，实现负载均衡
    tasks = [(fn, dataset_root, f'cuda:{i % num_gpus}') for i, fn in enumerate(pocket_fns)]

    # --- 初始化一个空的总统计字典 ---
    aggregated_stats = {
        'total_pockets': len(pocket_fns), 'embedding_found': 0, 'embedding_missing': 0,
        'unreadable_pt': 0, 'nan_files': 0, 'inf_files': 0, 'zero_embeddings': 0,
        'shape_mismatch': 0, 'single_chain_pockets': 0, 'multi_chain_pockets': 0,
        'dimensions': Counter(), 'pocket_sizes': [],
        'embedding_means': [], 'embedding_stds': [], 'embedding_mins': [], 'embedding_maxs': [],
        'mismatch_examples': [], 'missing_examples': []
    }

    # --- 创建进程池并执行任务 ---
    with Pool(processes=num_workers) as pool:
        # 使用 imap_unordered 来获取结果，这样tqdm可以实时更新
        results_iterator = pool.imap_unordered(process_single_pocket, tasks)

        # 使用tqdm来包装迭代器，显示进度条
        for stats in tqdm(results_iterator, total=len(tasks), desc=f"分析 {set_name}"):
            # --- 汇总每个进程返回的结果 ---
            aggregated_stats['embedding_found'] += stats['embedding_found']
            aggregated_stats['embedding_missing'] += stats['embedding_missing']
            aggregated_stats['unreadable_pt'] += stats['unreadable_pt']
            aggregated_stats['nan_files'] += stats['nan_files']
            aggregated_stats['inf_files'] += stats['inf_files']
            aggregated_stats['zero_embeddings'] += stats['zero_embeddings']
            aggregated_stats['shape_mismatch'] += stats['shape_mismatch']
            aggregated_stats['single_chain_pockets'] += stats['single_chain_pockets']
            aggregated_stats['multi_chain_pockets'] += stats['multi_chain_pockets']
            aggregated_stats['dimensions'].update(stats['dimensions'])
            if stats['pocket_size'] is not None:
                aggregated_stats['pocket_sizes'].append(stats['pocket_size'])
            if stats['embedding_stats'] is not None:
                aggregated_stats['embedding_means'].append(stats['embedding_stats']['mean'])
                aggregated_stats['embedding_stds'].append(stats['embedding_stats']['std'])
                aggregated_stats['embedding_mins'].append(stats['embedding_stats']['min'])
                aggregated_stats['embedding_maxs'].append(stats['embedding_stats']['max'])
            if stats['mismatch_example'] and len(aggregated_stats['mismatch_examples']) < 5:
                aggregated_stats['mismatch_examples'].append(stats['mismatch_example'])
            if stats['missing_example'] and len(aggregated_stats['missing_examples']) < 5:
                aggregated_stats['missing_examples'].append(stats['missing_example'])

    print_report(set_name, aggregated_stats)
    return aggregated_stats

def print_report(set_name: str, stats: dict):
    """打印格式化的分析报告 (此函数无需修改)。"""
    # ... (原封不动地复制你原来的 print_report 函数)
    total = stats['total_pockets']
    found = stats['embedding_found']

    print("\n" + "="*60)
    print(f"               {set_name} - 质量分析报告")
    print("="*60)
    print(f"总计应有口袋: {total}")
    print(f"  - 实际找到 Embedding: {found}")
    print(f"  - 缺失 Embedding: {stats['embedding_missing']}")
    found_p = (found / total * 100) if total > 0 else 0
    print(f"  - 数据完整率: {found_p:.2f}%")
    print("-" * 60)

    parsable_pockets = stats['single_chain_pockets'] + stats['multi_chain_pockets']

    print("1. 口袋构成分析 (基于找到的Embedding):")
    single_p = (stats['single_chain_pockets'] / parsable_pockets * 100) if parsable_pockets > 0 else 0
    multi_p = (stats['multi_chain_pockets'] / parsable_pockets * 100) if parsable_pockets > 0 else 0
    print(f"  - 单链口袋: {stats['single_chain_pockets']} ({single_p:.2f}%)")
    print(f"  - 多链(界面)口袋: {stats['multi_chain_pockets']} ({multi_p:.2f}%)")

    print("\n2. Embedding 文件质量检查:")
    print(f"  - PDB或PT文件格式问题: {stats['unreadable_pt']}")
    print(f"  - 含有 NaN/inf: {stats['nan_files']} / {stats['inf_files']}")
    print(f"  - 全零 Embedding: {stats['zero_embeddings']}")

    print("\n3. 维度一致性检查:")
    print(f"  - 行数(N)与PDB残基数不匹配: {stats['shape_mismatch']}")

    print(f"\n4. Embedding 维度(D)统计: {dict(stats['dimensions'])}")

    print("\n5. 数值与口袋大小分布统计:")
    if stats['embedding_means']:
        print(f"  - 数值范围 (平均 Min/Max): {np.mean(stats['embedding_mins']):.2f} / {np.mean(stats['embedding_maxs']):.2f}")
        print(f"  - 数值分布 (平均 Mean/Std): {np.mean(stats['embedding_means']):.4f} / {np.mean(stats['embedding_stds']):.2f}")
    if stats['pocket_sizes']:
        print(f"  - 口袋大小 (Min/Max): {np.min(stats['pocket_sizes'])} / {np.max(stats['pocket_sizes'])}")
        print(f"  - 口袋大小 (平均/中位数): {np.mean(stats['pocket_sizes']):.1f} / {np.median(stats['pocket_sizes']):.1f}")

    print("="*60)

    if stats['missing_examples']:
        print("\n[!] 部分缺失的 Embedding 文件示例:")
        for fn in stats['missing_examples']: print(f"    - {fn}")
    if stats['mismatch_examples']:
        print("\n[!] 部分维度不匹配的文件示例:")
        for ex in stats['mismatch_examples']: print(f"    - {ex}")


if __name__ == "__main__":
    # --- 路径配置 ---
    DATASET_ROOT = "/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd/crossdocked_pocket"
    SPLIT_FILE = os.path.join(DATASET_ROOT, "split_by_name.pt")

    # --- 并行化配置 ---
    # 使用你CPU核心数的一半或全部，根据I/O性能调整
    NUM_WORKERS = 30 # 你有60个CPU，可以设置在 30 到 50 之间

    # 确保在使用多进程时，PyTorch的启动方式是 'fork' (在Linux上是默认的)
    # torch.multiprocessing.set_start_method('fork') # 通常不需要手动设置

    print(f"正在从 {SPLIT_FILE} 加载训练/测试集划分...")
    try:
        split = torch.load(SPLIT_FILE, weights_only=True)
        train_pockets = [p for p, l in split['train']]
        test_pockets = [p for p, l in split['test']]
        all_pockets = train_pockets + test_pockets
    except FileNotFoundError:
        print(f"错误: 找不到数据集划分文件: {SPLIT_FILE}")
        exit()
    except Exception as e:
        print(f"错误: 加载划分文件时出错: {e}")
        exit()

    start_time = time.time()

    train_stats = run_qa_on_set_parallel(train_pockets, "训练集 (Train Set)", DATASET_ROOT, NUM_WORKERS)
    test_stats = run_qa_on_set_parallel(test_pockets, "测试集 (Test Set)", DATASET_ROOT, NUM_WORKERS)
    all_stats = run_qa_on_set_parallel(all_pockets, "总合集 (Combined)", DATASET_ROOT, NUM_WORKERS)

    end_time = time.time()
    print(f"\n所有分析完成。总耗时: {end_time - start_time:.2f} 秒。")
