

import pandas as pd
from pathlib import Path
#import py3Dmol
import warnings
from pocket_embed_new import process_pocket,_handle_alt_loc_df,standardize_pdb
# 忽略pandas读取固定宽度文件时的性能警告
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
import os
import torch
import tempfile
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import io
import warnings

# 忽略pandas读取固定宽度文件时的性能警告
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Assuming ESM-3 and other dependencies are installed
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig

print("库导入成功！")
# --- 请修改为您的文件路径 ---
# 假设您已将test_set解压到data目录下
pdb_path = Path("/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/data/cross_docked_new/crossdocked_pocket/CARP_CRYPA_90_419_catalytic_0/3wz8_A_rec_5hct_61p_lig_tt_min_0_pocket10.pdb")

if not pdb_path.exists():
    print(f"错误：文件未找到！请检查路径：{pdb_path}")
else:
    print(f"成功找到文件：{pdb_path.name}")
# with open(pdb_path) as f:
#     for line in f:
#         if line.startswith("ATOM") or line.startswith("HETATM"):
#             print(line[54:60])

# --- Improvement 1: Load the model ONCE ---
print("[*] Loading ESM-3 model... (this may take a moment)")
device = torch.device(1)
# Use bfloat16 for faster inference and less memory usage if available
model = ESM3.from_pretrained("esm3-sm-open-v1").to(device).eval()
print("[*] Model loaded successfully.")

# --- Improvement 2: Discover all files using your index.pkl ---

output_path ='/mnt/rna01/liuzhiyuan/zyliu/nai/NExT-Mol/test_scripts'



    
embedding = process_pocket(str(pdb_path), model, device)
if embedding is not None:
        torch.save(embedding, output_path)


print("[*] All pockets processed successfully.")

