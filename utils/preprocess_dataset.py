import os
import subprocess
import torch
from tqdm import tqdm
import argparse
import AutoDockTools

def get_receptor_pdbqt_with_consistent_logic(pdb_path, pdbqt_path):
    """
    [最终修正版]
    带缓存的辅助函数，用于准备受体PDBQT文件。
    严格复现 TargetDiff 使用 subprocess.Popen 静默失败的逻辑。
    """
    # 脚本的逻辑是先删除再调用，所以这里不再需要检查文件是否存在
    
    pqr_path = pdbqt_path.replace('.pdbqt', '.pqr')
    
    # 步骤 1: PDB -> PQR
    pqr_command = ['pdb2pqr30', '--ff=AMBER', pdb_path, pqr_path]
    subprocess.Popen(pqr_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).communicate()

    if not os.path.exists(pqr_path):
        raise FileNotFoundError(f"PQR file was not generated for {pdb_path}. PDB file is likely corrupted.")

    # 步骤 2: PQR -> PDBQT
    prepare_receptor_script = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
    pdbqt_command = ['python3', prepare_receptor_script, '-r', pqr_path, '-o', pdbqt_path]
    subprocess.Popen(pdbqt_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).communicate()

    if not os.path.exists(pdbqt_path):
        raise FileNotFoundError(f"PDBQT file was not generated from {pqr_path}.")


def main(args):
    print(f"Loading split file from: {args.split_file}")
    split_data = torch.load(args.split_file)
    
    split_to_process = 'test'
    
    print(f"\n--- Force regenerating receptors for {split_to_process.upper()} Set ---")
    
    successful_count = 0
    failed_pockets = []

    unique_pockets = sorted(list(set([pair[0] for pair in split_data[split_to_process]])))
    
    for pocket_fn in tqdm(unique_pockets, desc=f"Regenerating PDBQTs for {split_to_process} set"):
        full_pdb_path = os.path.join(args.dataset_root, pocket_fn)
        
        if not os.path.exists(full_pdb_path):
            failed_pockets.append({'path': pocket_fn, 'reason': 'PDB file not found.'})
            continue

        # --- 核心逻辑：先删除，再重新生成 ---
        pqr_path = full_pdb_path.replace('.pdb', '.pqr')
        pdbqt_path = full_pdb_path.replace('.pdb', '.pdbqt')

        # 1. 删除旧文件 (如果存在)
        for f_path in [pqr_path, pdbqt_path]:
            try:
                os.remove(f_path)
            except FileNotFoundError:
                pass # 文件本就不存在，正好

        # 2. 调用函数重新生成
        try:
            get_receptor_pdbqt_with_consistent_logic(full_pdb_path, pdbqt_path)
            successful_count += 1
        except Exception as e:
            failed_pockets.append({'path': pocket_fn, 'reason': str(e)})

    # --- 生成最终报告 ---
    print("\n" + "="*80)
    print("--- Receptor Regeneration Report ---")
    print("="*80)
    
    num_total = len(unique_pockets)
    num_failed = len(failed_pockets)
    success_rate = (successful_count / num_total) * 100 if num_total > 0 else 0
    
    summary = (
        f"\n[{split_to_process.upper()} Set Summary]\n"
        f"  Total Unique Pockets to Process: {num_total}\n"
        f"  Successfully Regenerated: {successful_count} ({success_rate:.2f}%)\n"
        f"  Failed to Regenerate: {num_failed}\n"
    )
    print(summary)

    if failed_pockets:
        print("\n--- Failed Pockets Details ---")
        for item in failed_pockets:
            print(f"  Pocket: {item['path']}")
            print(f"  Reason: {item['reason']}")
            print("-" * 20)
            
    print("\nRegeneration process complete.")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Force regenerate PDBQT receptor files for the test set.")
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the CrossDocked dataset.')
    parser.add_argument('--split_file', type=str, required=True, help='Path to the split .pt file.')
    args = parser.parse_args()
    main(args)