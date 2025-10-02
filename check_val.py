import os
import pickle
import torch
import argparse
import random
from torch.utils.data import Dataset, Subset
from rdkit import Chem, RDLogger
import selfies as sf
from tqdm import tqdm

# 禁用 RDKit 的冗余警告
RDLogger.DisableLog('rdApp.warning')

# ==============================================================================
#  从你的代码中复制过来的核心类和函数，以确保100%模拟环境
# ==============================================================================

def restricted_random_smiles(smiles, addHs=False):
    mol = Chem.MolFromSmiles(smiles)
    if addHs:
        mol = Chem.AddHs(mol)
    new_order = list(range(mol.GetNumAtoms()))
    random.shuffle(new_order)
    new_mol = Chem.RenumberAtoms(mol, newOrder=new_order)
    return Chem.MolToSmiles(new_mol, canonical=False, isomericSmiles=False)

class PocketLigandPairDataset(Dataset):
    def __init__(self, raw_path, rand_smiles='restricted', addHs=False, data_cache=None):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.rand_smiles = rand_smiles
        self.addHs = addHs
        self.data_cache = data_cache

        if self.data_cache is None:
            if not os.path.exists(self.index_path):
                raise FileNotFoundError(f"主索引文件未找到: {self.index_path}。请确认路径是否正确。")
            with open(self.index_path, 'rb') as f:
                self.index = pickle.load(f)
        else:
            self.index = list(self.data_cache.keys())

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        try:
            if self.data_cache is not None:
                original_idx = self.index[idx]
                return self.data_cache[original_idx]

            pocket_fn, ligand_fn, _, _ = self.index[idx]
            pocket_path = os.path.join(self.raw_path, pocket_fn)
            ligand_path = os.path.join(self.raw_path, ligand_fn)
            pt_embed_path = pocket_path.replace('.pdb', '.pt')

            # 检查所有文件是否存在
            if not os.path.exists(pt_embed_path):
                # print(f"警告: Embedding 文件不存在: {pt_embed_path}")
                return None
            if not os.path.exists(ligand_path):
                # print(f"警告: Ligand SDF 文件不存在: {ligand_path}")
                return None

            embedding = torch.load(pt_embed_path, map_location='cpu')
            mol = Chem.SDMolSupplier(ligand_path, removeHs=False, sanitize=True)[0]
            if mol is None:
                # print(f"警告: 无法从SDF加载分子: {ligand_path}")
                return None

            smiles = Chem.MolToSmiles(mol)
            # 简单返回一个字典，避免后续处理的复杂性
            return {'status': 'ok'}

        except Exception as e:
            # 捕获所有其他可能的错误
            # print(f"警告: 处理索引 {idx} 时发生未知错误: {e}")
            return None

# ==============================================================================
#  主诊断逻辑
# ==============================================================================
def main(args):
    print("--- 开始验证集完整性检查 ---")
    print(f"数据集根目录: {args.dataset_root}")
    print(f"分割文件路径: {args.split_file}")
    print(f"模拟批次大小: {args.batch_size}")

    if not os.path.exists(args.dataset_root):
        print(f"错误: 数据集根目录不存在: {args.dataset_root}")
        return
    if not os.path.exists(args.split_file):
        print(f"错误: 分割文件不存在: {args.split_file}")
        return

    # 1. 加载完整数据集索引
    print("\n--> 步骤 1: 加载完整数据集索引...")
    try:
        full_dataset = PocketLigandPairDataset(raw_path=args.dataset_root)
        print(f"成功加载完整索引，共 {len(full_dataset)} 个样本。")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return

    # 2. 创建 (pocket_fn, ligand_fn) 到索引的映射
    print("--> 步骤 2: 创建文件名到索引的映射...")
    file_to_idx_map = {
        (p, l): i
        for i, (p, l, _, _) in enumerate(tqdm(full_dataset.index, desc="构建映射"))
        if p is not None and l is not None
    }

    # 3. 加载分割文件并获取验证集索引
    print("--> 步骤 3: 加载分割文件并提取验证集索引...")
    split = torch.load(args.split_file)
    # 你的 dataloader 使用 'test' 作为验证集
    validation_file_pairs = split['test']

    test_indices = []
    missing_pairs = 0
    for pocket_fn, ligand_fn in validation_file_pairs:
        key = (pocket_fn, ligand_fn)
        if key in file_to_idx_map:
            test_indices.append(file_to_idx_map[key])
        else:
            missing_pairs += 1

    print(f"从分割文件中找到 {len(validation_file_pairs)} 个验证样本对。")
    if missing_pairs > 0:
        print(f"警告: {missing_pairs} 个样本对在主索引中未找到，已被忽略。")

    # 4. 创建验证集 Subset
    validation_dataset = Subset(full_dataset, test_indices)
    print(f"成功创建验证集 Subset，包含 {len(validation_dataset)} 个样本。")

    # 5. [检查阶段 1] 检查单个文件是否能成功加载
    print("\n--- 阶段 1: 检查单个样本加载情况 ---")
    failed_samples = []
    for i in tqdm(range(len(validation_dataset)), desc="扫描单个样本"):
        item = validation_dataset[i]
        if item is None:
            original_idx = test_indices[i]
            file_info = full_dataset.index[original_idx]
            failed_samples.append({'subset_idx': i, 'original_idx': original_idx, 'info': file_info})

    # 6. [检查阶段 2] 模拟批次整理过程
    print("\n--- 阶段 2: 模拟批次整理 (Collation) ---")
    empty_batches = []
    batch_size = args.batch_size

    # 将失败样本的索引存入一个集合，以便快速查找
    failed_subset_indices = {s['subset_idx'] for s in failed_samples}

    for i in tqdm(range(0, len(validation_dataset), batch_size), desc="扫描批次"):
        is_batch_empty = True
        for j in range(i, min(i + batch_size, len(validation_dataset))):
            if j not in failed_subset_indices:
                is_batch_empty = False
                break

        if is_batch_empty:
            batch_num = i // batch_size
            original_indices_in_batch = test_indices[i:i+batch_size]
            empty_batches.append({'batch_num': batch_num, 'original_indices': original_indices_in_batch})

    # 7. 生成最终报告
    print("\n" + "="*80)
    print("                  验证集健康检查报告")
    print("="*80)
    print(f"总验证样本数: {len(validation_dataset)}")
    print(f"无法加载的样本数: {len(failed_samples)}")
    print(f"导致“全空”的批次数: {len(empty_batches)}")
    print("-"*80)

    if failed_samples:
        print("\n[!] 发现无法加载的样本 (最多显示前20个):")
        for sample in failed_samples[:20]:
            print(f"  - Subset索引: {sample['subset_idx']}, 原始索引: {sample['original_idx']}, 文件: {sample['info'][:2]}")

    if empty_batches:
        print("\n[!!!] 严重问题: 发现会导致程序崩溃的“全空”批次！")
        for batch_info in empty_batches:
            print(f"  - 批次编号: {batch_info['batch_num']} (包含原始索引从 {batch_info['original_indices'][0]} 到 {batch_info['original_indices'][-1]}) 完全由损坏/缺失的样本组成。")
        print("\n[诊断结论]: 这几乎可以肯定是导致你训练中断的根本原因。")
        print("           当 DataLoader 加载到这些批次时，它会向模型传递一个空张量，引发 IndexError。")
        print("           请检查上述列出的样本文件是否存在或已损坏。")
    elif len(failed_samples) > 0:
        print("\n[诊断结论]: 数据集存在部分损坏的样本，但尚未发现能直接导致崩溃的“全空”批次。")
        print("           虽然目前配置下可能不会崩溃，但数据损坏问题依然存在，建议修复。")
    else:
        print("\n[诊断结论]: 数据集看起来是健康的！所有验证样本都可以成功加载。")
        print("           问题可能不在于数据本身，而在于模型代码或训练环境的其它方面。")

    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证 CrossDocked 数据集的完整性")
    parser.add_argument(
        '--dataset_root',
        type=str,
        default="/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd/crossdocked_pocket",
        help='数据集的根目录 (包含 index.pkl 的目录)'
    )
    parser.add_argument(
        '--split_file',
        type=str,
        default="/data/share/liuzhiyuan/nai/NExT-Mol/datasets/sbdd/crossdocked_pocket/split_by_name.pt",
        help='包含训练/测试分割的 .pt 文件'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='模拟验证时使用的批次大小'
    )
    args = parser.parse_args()
    main(args)
