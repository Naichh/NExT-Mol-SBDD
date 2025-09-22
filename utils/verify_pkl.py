import os
import pickle
from tqdm import tqdm

# --- 配置 ---
# 设置您的数据集根目录
ROOT_DIR = "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket"
# -----------

def verify_files():
    print(f"[*] Starting verification in: {ROOT_DIR}")
    
    # 收集所有 .pkl 文件的路径
    pkl_files = []
    for root, _, files in os.walk(ROOT_DIR):
        for file in files:
            if file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))

    if not pkl_files:
        print("[!] No .pkl files found. Please check your ROOT_DIR.")
        return

    print(f"[*] Found {len(pkl_files)} .pkl files to verify.")
    
    corrupted_files = []
    
    # 使用 tqdm 创建进度条
    for pkl_path in tqdm(pkl_files, desc="Verifying files"):
        try:
            with open(pkl_path, 'rb') as f:
                # 只需要尝试加载，不需要使用加载的数据
                pickle.load(f)
        except Exception as e:
            # 如果加载失败，记录文件名和错误
            print(f"\n[!!!] CORRUPTED FILE DETECTED: {pkl_path}")
            print(f"      Reason: {e}")
            corrupted_files.append(pkl_path)
    
    if not corrupted_files:
        print("\n[✔] All .pkl files verified successfully!")
    else:
        print(f"\n[!] Verification complete. Found {len(corrupted_files)} corrupted files:")
        for path in corrupted_files:
            print(f"  - {path}")

if __name__ == "__main__":
    verify_files()