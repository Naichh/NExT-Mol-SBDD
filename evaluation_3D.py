import argparse
import os
import pickle
import time
from tqdm import tqdm
from collections import defaultdict
import multiprocessing as mp
import math
import torch
from torch import optim
import lightning as L
import os
import subprocess
from transformers import AutoTokenizer
# from model.modeling_llama import LlamaForCausalLM, LlamaConfig
from transformers import LlamaForCausalLM, LlamaConfig
from peft import get_peft_model, LoraConfig
from model.help_funcs import AttrDict
from pathlib import Path
import selfies as sf
from rdkit import Chem
from tqdm import tqdm
from evaluation.eval_functions import get_2D_edm_metric, SBDD_validation
from torch.nn import CrossEntropyLoss
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict # Make sure this is imported
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch
import torch.nn as nn
from evaluation.eval_functions import run_full_evaluation
def worker_generate_3d(data_pair):

    selfies_string, pocket_path = data_pair
    try:
        smiles = sf.decoder(selfies_string)
        if not smiles:
            return None, None, selfies_string

        mol_3d = generate_3d_mol(smiles,'fast')

        if mol_3d is None:
            return None, None, selfies_string

        return mol_3d, pocket_path, selfies_string

    except Exception:
        return None, None, selfies_string

def generate_3d_mol(smiles: str, quality: str = 'fast') -> Chem.Mol:
        try:
            # 1. 从SMILES创建分子对象并加氢
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol = Chem.AddHs(mol, addCoords=True)

            # 2. 检查分子是否与MMFF94力场兼容
            # 如果分子包含MMFF94不支持的原子类型，此函数会抛出异常
            AllChem.MMFFGetMoleculeProperties(mol)

            # 3. 根据质量要求设置参数
            if quality == 'high':
                num_confs = 50  # 高质量模式下，进行更广泛的构象搜索
            else:  # 'fast' 模式
                num_confs = 5  # 快速模式下，进行有限的构象搜索

            params = AllChem.ETKDGv3()
            params.randomSeed = 0xf00d
            params.numThreads = 1  # 在多进程环境中，必须设置为1以保证线程安全

            # 4. 生成多个初始3D构象
            cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
            if len(cids) == 0:
                # 如果RDKit未能生成任何有效构象
                return None

            # 5. 使用MMFF94力场优化所有构象，并找到能量最低的那个
            # res 是一个元组列表，每个元组是 (是否收敛, 能量)
            res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1)

            # 筛选出成功收敛的构象
            converged_res = [(i, e) for i, (flag, e) in enumerate(res) if flag == 0]
            if not converged_res:
                # 如果没有一个构象成功收敛
                return None

            # 从成功收敛的构象中，找到能量最低的那个
            min_energy_idx, _ = min(converged_res, key=lambda x: x[1])

            # 6. 创建一个只包含最优构象的新分子对象并返回
            best_mol = Chem.Mol(mol)
            best_mol.RemoveAllConformers()
            best_conformer = mol.GetConformer(int(min_energy_idx))
            best_mol.AddConformer(best_conformer, assignId=True)

            return best_mol

        except Exception:
            # 捕获所有可能的异常 (如SMILES解析失败, MMFF不兼容, 构象生成失败等)
            # 并以安全的方式返回None，防止整个程序崩溃。
            return None
def main():
    """
    主函数，负责读取任务、并行处理、调用评估并保存结果。
    """
    input_file='/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/sampled_mol/20251006_203709_/epoch_0_3d_tasks.pkl'
    print("="*80)
    print("--- Starting Offline CPU Evaluation Script ---")
    print(f"--- Loading tasks from: {input_file} ---")
    print("="*80)

    # 1. 加载任务
    if not os.path.exists(input_file):
        print(f"[ERROR] Input file not found: {input_file}")
        return

    with open(input_file, 'rb') as f:
        tasks = pickle.load(f)

    print(f"--- Loaded {len(tasks)} tasks to process. ---")

    # 2. 并行生成3D构象
    print("\n--- Stage 1: Generating 3D Conformers in Parallel ---")
    start_time = time.time()

    # 准备 multiprocessing Pool 的输入
    pool_tasks = [(task['selfies'], task['pocket_path']) for task in tasks]

    # 获取CPU核心数
    slurm_cpus_str = os.getenv('SLURM_CPUS_PER_TASK')
    if slurm_cpus_str:
        cpu_cores = int(slurm_cpus_str)
        print(f"INFO: Detected Slurm allocation. Using {cpu_cores} allocated CPU cores.")
    else:
        # Fallback to mp.cpu_count() if not in a Slurm environment
        cpu_cores = mp.cpu_count()
        print(f"INFO: Not in a Slurm environment. Using total system CPU cores: {cpu_cores}.")
    num_processes = max(1, cpu_cores - 2)
    print(f"--- Using {num_processes} CPU cores for parallel processing. ---")

    generated_mols_3d_with_info = []

    with mp.Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(worker_generate_3d, pool_tasks)

        for i, result in enumerate(tqdm(results_iterator, total=len(pool_tasks), desc="Generating 3D conformers")):
            mol_3d, pocket_path, original_selfies = result
            if mol_3d:
                generated_mols_3d_with_info.append({
                    "mol_3d": mol_3d,
                    "pocket_path": pocket_path
                })

    duration = time.time() - start_time
    print(f"--- Finished 3D generation for {len(generated_mols_3d_with_info)} molecules in {duration:.2f} seconds. ---")

    # 3. 调用评估函数
    print("\n--- Stage 2: Running Full Evaluation (including Vina Docking) ---")

    if not generated_mols_3d_with_info:
        print("[WARNING] No valid 3D molecules were generated. Skipping evaluation.")
        return

    # 将结果按 pocket_path 分组，以匹配 run_full_evaluation 的输入格式
    grouped_mols_for_eval = defaultdict(list)
    for item in generated_mols_3d_with_info:
        grouped_mols_for_eval[item['pocket_path']].append(item['mol_3d'])

    # 调用您现有的评估接口
    eval_metrics = run_full_evaluation(grouped_mols=grouped_mols_for_eval)

    # 4. 保存结果
    print("\n--- Stage 3: Saving Evaluation Metrics ---")

    # 定义输出日志文件的路径
    output_log_path = os.path.join(os.path.dirname(input_file), "evaluation_results.log")

    with open(output_log_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"Evaluation Results for: {input_file}\n")
        f.write("="*60 + "\n")
        if eval_metrics:
            for key, value in eval_metrics.items():
                log_line = f"  - {key:<30}: {value:.4f}\n"
                print(log_line, end='')
                f.write(log_line)
        else:
            print("No metrics were generated.")
            f.write("No metrics were generated.\n")
        f.write("="*60 + "\n")

    print(f"\n--- SUCCESS: Evaluation complete. Results saved to: {output_log_path} ---")
    print("="*80)


if __name__ == '__main__':
    main()
