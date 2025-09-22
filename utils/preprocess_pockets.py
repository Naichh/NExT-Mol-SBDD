import os
import torch
import subprocess
from tqdm import tqdm
import argparse
import AutoDockTools  # 用于定位脚本路径
import sys

def prepare_receptor_with_report(pdb_path):
    """
    尝试为一个PDB文件生成PDBQT，并返回成功与否及原因。
    """
    pqr_path = pdb_path.replace('.pdb', '.pqr')
    pdbqt_path = pdb_path.replace('.pdb', '.pdbqt')

    if os.path.exists(pdbqt_path):
        return True, "Already exists"

    try:
        pqr_command = ['pdb2pqr30', '--ff=AMBER', '--ignore-missing', pdb_path, pqr_path]
        pqr_result = subprocess.run(pqr_command, capture_output=True, text=True, timeout=300)
        
        if pqr_result.returncode != 0:
            error_message = pqr_result.stderr if pqr_result.stderr else "Unknown error from pdb2pqr."
            return False, error_message

        prepare_receptor_script = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
        pdbqt_command = ['python3', prepare_receptor_script, '-r', pqr_path, '-o', pdbqt_path]
        
        subprocess.run(pdbqt_command, check=True, capture_output=True, text=True, timeout=300)
        
        return True, "Successfully processed"

    except subprocess.CalledProcessError as e:
        return False, f"Subprocess Error during PDBQT creation: {e.stderr}"
    except subprocess.TimeoutExpired:
        return False, "Processing timed out"
    except Exception as e:
        return False, f"An unexpected error occurred: {str(e)}"

def main(args):
    os.makedirs(args.log_dir, exist_ok=True)
    
    print(f"Loading split file from: {args.split_file}")
    split_data = torch.load(args.split_file)
    
    split_to_process = 'test'
    
    print(f"\n--- Analyzing {split_to_process.upper()} Set ---")
    
    successful_pockets = set()
    failed_pockets = []

    unique_pockets = sorted(list(set([pair[0] for pair in split_data[split_to_process]])))
    
    for pocket_fn in tqdm(unique_pockets, desc=f"Testing {split_to_process} pockets"):
        full_pdb_path = os.path.join(args.dataset_root, pocket_fn)
        
        if not os.path.exists(full_pdb_path):
            failed_pockets.append({'path': pocket_fn, 'reason': 'PDB file not found.'})
            continue

        is_success, reason = prepare_receptor_with_report(full_pdb_path)
        
        if is_success:
            successful_pockets.add(pocket_fn)
        else:
            failed_pockets.append({'path': pocket_fn, 'reason': reason.strip()})
    
    # --- 生成报告 ---
    print("\n" + "="*80)
    print("--- Receptor Preprocessing Report ---")
    print("="*80)
    
    report_content = ""
    num_total = len(unique_pockets)
    num_success = len(successful_pockets)
    num_failed = len(failed_pockets)
    success_rate = (num_success / num_total) * 100 if num_total > 0 else 0
    
    summary = (
        f"\n[{split_to_process.upper()} Set Summary]\n"
        f"  Total Unique Pockets Analyzed: {num_total}\n"
        f"  Successfully Processable: {num_success} ({success_rate:.2f}%)\n"
        f"  Failed to Process: {num_failed}\n"
    )
    print(summary)
    report_content += summary

    report_file = os.path.join(args.log_dir, 'receptor_preprocessing_report_test_only.txt')
    with open(report_file, 'w') as f:
        f.write(report_content)
        f.write("\n--- Failed Pockets Details ---\n")
        if not failed_pockets:
            f.write("None\n")
        for item in failed_pockets:
            f.write(f"Pocket: {item['path']}\nReason: {item['reason']}\n---\n")
            
    print(f"\nDetailed failure report saved to: {report_file}")
    print("Analysis complete. No split files were created or modified.")
    print("="*80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze and report the validity of CrossDocked pocket files for the test set.")
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the CrossDocked dataset.')
    parser.add_argument('--split_file', type=str, required=True, help='Path to the split .pt file.')
    parser.add_argument('--log_dir', type=str, default="/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/scripts/logs", help="Directory to save the report file.")
    
    args = parser.parse_args()
    main(args)