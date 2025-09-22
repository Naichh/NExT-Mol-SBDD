import os
from tqdm import tqdm
import time

# --- 配置 ---
# 设置您的数据集根目录
ROOT_DIR = "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket"
# -----------

def check_progress():
    print(f"[*] Starting to scan for pocket PDB files in: {ROOT_DIR}")
    start_time = time.time()
    
    # 1. 递归查找所有 *_pocket*.pdb 文件
    pdb_files = []
    for root, _, files in os.walk(ROOT_DIR):
        for file in files:
            # 确保我们只检查 pocket PDB 文件，避免统计其他无关的 pdb
            if file.endswith(".pdb") and "_pocket" in file:
                pdb_files.append(os.path.join(root, file))
                
    if not pdb_files:
        print("\n[!] Error: No pocket PDB files found. Please check the ROOT_DIR path.")
        return

    total_pdb_count = len(pdb_files)
    print(f"[*] Found {total_pdb_count} total pocket PDB files.")
    
    # 2. 检查每个 PDB 是否有对应的 .pt 文件
    found_count = 0
    missing_count = 0
    missing_files_list = []
    
    for pdb_path in tqdm(pdb_files, desc="Checking for corresponding .pt files"):
        # 构建对应的 .pt 文件路径
        # os.path.splitext('/path/to/file.pdb') -> ('/path/to/file', '.pdb')
        base_path = os.path.splitext(pdb_path)[0]
        pt_path = base_path + ".pt"
        
        if os.path.exists(pt_path):
            found_count += 1
        else:
            missing_count += 1
            missing_files_list.append(pdb_path)
            
    end_time = time.time()
    
    # 3. 打印最终的报告
    print("\n" + "="*40)
    print("      Embedding File Verification Report")
    print("="*40)
    print(f"Scan Duration: {end_time - start_time:.2f} seconds")
    print(f"Total pocket PDB files scanned: {total_pdb_count}")
    print("-" * 40)
    print(f"✅ PDBs with a corresponding .pt file: {found_count} ({found_count/total_pdb_count:.2%})")
    print(f"❌ PDBs MISSING a corresponding .pt file: {missing_count} ({missing_count/total_pdb_count:.2%})")
    print("="*40 + "\n")

    # 4. 如果有缺失文件，打印一部分列表
    if missing_files_list:
        print(f"[!] Listing up to the first 100 PDB files that are missing their .pt embedding:")
        for i, missing_pdb in enumerate(missing_files_list[:100]):
            print(f"  - {missing_pdb}")
        if len(missing_files_list) > 100:
            print(f"  ... and {len(missing_files_list) - 100} more missing files.")
    else:
        print("[✔] Excellent! All pocket PDB files have a corresponding .pt embedding.")

if __name__ == "__main__":
    check_progress()