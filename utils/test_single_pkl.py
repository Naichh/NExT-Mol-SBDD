import pickle
import torch
import numpy as np
import traceback

# --- 配置 ---
# 我们将直接测试那个导致训练崩溃的文件
# 根据你之前的 `ls` 命令，这个 .pkl 文件是存在的
pkl_file_to_test = "/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/sbdd/crossdocked_pocket/BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3_pocket10.pkl"
# -----------

def test_readability():
    print("="*50)
    print(f"[*] Attempting to load a single PKL file in the current environment...")
    print(f"[*] File: {pkl_file_to_test}")
    print("="*50)

    try:
        # 尝试打开并用 pickle 加载文件
        with open(pkl_file_to_test, 'rb') as f:
            data = pickle.load(f)
        
        # 检查加载出的数据类型
        print(f"[✔] Successfully loaded data with pickle!")
        print(f"    - Data type: {type(data)}")
        
        # 如果是 numpy 数组，就尝试转换为 tensor
        if isinstance(data, np.ndarray):
            print(f"    - NumPy array shape: {data.shape}")
            tensor = torch.from_numpy(data)
            print(f"[✔] Successfully converted to PyTorch tensor!")
            print(f"    - Tensor shape: {tensor.shape}")
            print(f"    - Tensor dtype: {tensor.dtype}")
        
        print("\n[CONCLUSION] The file is readable in this environment.")

    except Exception as e:
        print("\n[!!!] FAILED to load the PKL file.")
        print(f"[ERROR TYPE] {type(e).__name__}")
        print(f"[ERROR DETAILS] {e}")
        print("\n--- Full Traceback ---")
        traceback.print_exc()
        print("----------------------")
        print("\n[CONCLUSION] The file is NOT readable in this environment.")
        print("             This confirms the environment incompatibility issue.")

if __name__ == "__main__":
    test_readability()