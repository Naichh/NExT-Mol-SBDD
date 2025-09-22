import os
from tqdm import tqdm
import time

# --- 配置 ---
# 设置您的数据集根目录
ROOT_DIR = "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket"
# --- 新增：输出文件名配置 ---
OUTPUT_FILE = "missing_files.txt"
# --------------------------

def find_and_export_missing():
    """
    遍历所有 pocket PDB 文件，找出既没有 .pt 也没有 .pkl 文件的样本，并将完整列表导出到 txt 文件。
    """
    print(f"[*] Starting scan for PDBs missing BOTH .pt and .pkl files in: {ROOT_DIR}")
    start_time = time.time()
    
    # 1. 收集所有相关的 PDB 文件路径
    pdb_files = []
    for root, _, files in os.walk(ROOT_DIR):
        for file in files:
            if file.endswith(".pdb") and "_pocket" in file:
                pdb_files.append(os.path.join(root, file))
                
    if not pdb_files:
        print("\n[!] Error: No pocket PDB files found. Please check the ROOT_DIR path.")
        return

    total_pdb_count = len(pdb_files)
    print(f"[*] Found {total_pdb_count} total pocket PDB files to check.")
    
    # 2. 检查每个 PDB，看是否两种格式的 embedding 都缺失
    missing_both_list = []
    
    for pdb_path in tqdm(pdb_files, desc="Checking for missing embeddings"):
        base_path = os.path.splitext(pdb_path)[0]
        pt_path = base_path + ".pt"
        pkl_path = base_path + ".pkl"
        
        if not os.path.exists(pt_path) and not os.path.exists(pkl_path):
            missing_both_list.append(pdb_path)
            
    end_time = time.time()
    
    # 3. 打印摘要报告到终端
    print("\n" + "="*50)
    print("      Completely Missing Embedding Files Report")
    print("="*50)
    print(f"Scan Duration: {end_time - start_time:.2f} seconds")
    print(f"Total pocket PDB files scanned: {total_pdb_count}")
    print("-" * 50)
    print(f"❌ Found {len(missing_both_list)} files missing BOTH .pt and .pkl embeddings.")
    print("="*50)

    # --- 4. 新增：将完整列表写入到 TXT 文件 ---
    if missing_both_list:
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                for path in missing_both_list:
                    f.write(path + '\n')
            print(f"\n[✔] Success! A complete list of {len(missing_both_list)} missing file paths has been saved to: {os.path.abspath(OUTPUT_FILE)}")
        except Exception as e:
            print(f"\n[!] Error: Failed to write to output file {OUTPUT_FILE}. Reason: {e}")
            
        # 仍然在终端打印一部分样本，方便快速预览
        print(f"\n[INFO] Listing up to the first 100 completely missing files in the console:")
        for i, missing_pdb in enumerate(missing_both_list[:100]):
            print(f"  - {missing_pdb}")
        if len(missing_both_list) > 100:
            print(f"  ... and {len(missing_both_list) - 100} more (see .txt file for the full list).")
    else:
        print("\n[✔] Great news! No PDB files are missing both formats.")

if __name__ == "__main__":
    find_and_export_missing()